# py
from os.path import join, exists
from os import makedirs, remove
import csv
import copy
from argparse import ArgumentParser

# libraries imports
import nibabel as nib
import numpy as np
from PIL import Image
import cv2

# project imports
from dataset.data_loader import DataLoader
from utils.deformation_utils import deform2D
from config import config_donors, config_dev, config_database


# ******************************************************************************************************************** #
#
# This file uses the computed latent SVF from the ST algorithm to deform the initial stained sections (LFB, HE) at
# the original high-resolution (possibly with some --downsample_factor).
# It uses the SVF computed using the ST algorithm specified at --reg_algorithm
#
# ******************************************************************************************************************** #


####################
# Input parameters #
####################
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='P57-16', choices=list(config_donors.file_dict.keys()))
arg_parser.add_argument('--reg_algorithm', default='ST3_L1_RegNet_NN2',
                        choices=['Linear', 'RegNet', 'ST3_L1_RegNet_NN2',  'ST3_L1_RegNet_NN4'])
arg_parser.add_argument('--downsample_factor', default=2, type=int)
arg_parser.add_argument('--bid', default=None, nargs='+')

arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
reg_algorithm = arguments.reg_algorithm
downsample_factor = arguments.downsample_factor
initial_bid_list = arguments.bid


BT_DB_blocks = config_database.get_lin_dict(SUBJECT)
BT_DB_slices = config_database.get_nonlin_dict(SUBJECT)
data_dict = {SUBJECT: BT_DB_slices}
config_file_data = config_donors.file_dict[SUBJECT]
parameter_dict_MRI = config_dev.CONFIG_DICT['MRI']

REG_BLOCKS_DIR = join(BT_DB_blocks['RESULTS_DIR'], 'results', 'reg_blocks')
SLICES_DIR = join(BT_DB_blocks['RESULTS_DIR'], 'results', 'slices')

base_dir = join(config_file_data.BASE_DIR, 'NonlinearRegistration')
data_path_dict = {
    'RegNet': {'path': join(base_dir, 'Registration', 'RegNet'), 'stain': ['MRI_LFB', 'MRI_HE']},
    'ST3_L1_RegNet_NN2': {'path': join(base_dir, 'ST3', 'l1', 'NN2'), 'stain': ['LFB', 'HE']},
    'ST3_L1_RegNet_NN4': {'path': join(base_dir, 'ST3', 'l1', 'NN4'), 'stain': ['LFB', 'HE']}
}

BLOCK_res = config_file_data.BLOCK_res
HISTO_res = config_file_data.HISTO_res

bid_list = initial_bid_list if initial_bid_list is not None else config_file_data.initial_bid_list
bid_list = {SUBJECT: bid_list}

data_loader = DataLoader(data_dict, included_blocks=bid_list)
subject_loader = data_loader.subject_dict[SUBJECT]
initial_bid_list = initial_bid_list if initial_bid_list is not None else config_file_data.initial_bid_list

missing_slices = []
for it_bid, bid in enumerate(initial_bid_list):

    print('')
    print('Block ID: ' + bid)

    block_loader = subject_loader.block_dict[bid]
    SBJ_BLOCK = BT_DB_blocks['SUBJECT'] + '_' + bid
    BLOCK_DIR = join(SLICES_DIR, bid)
    HISTO_DIR = join(BT_DB_blocks['SLIDES_DIR'], SBJ_BLOCK)
    AFFINE = block_loader.vox2ras0

    # try:
    for stain in data_path_dict[reg_algorithm]['stain']:
        if stain !='LFB': continue
        print(' ' + stain)
        print('   Slice number:', end=' ', flush=True)

        results_dir_sbj = join(data_path_dict[reg_algorithm]['path'], bid)
        proxy_flow = nib.load(join(results_dir_sbj, stain + '.totalflow.nii.gz'))
        flowshape = proxy_flow.shape

        if not exists(join(results_dir_sbj, stain)):
            makedirs(join(results_dir_sbj, stain))

        resize_factor = BLOCK_res / (HISTO_res*downsample_factor)
        resized_shape = tuple([int(np.round(a * resize_factor)) for a in flowshape[1:3]])
        resize_factor = [a / b for a, b in zip(resized_shape, flowshape[1:3])]

        if stain == data_path_dict[reg_algorithm]['stain'][0]:
            aux = np.asarray([(ipf - 1) / (2 * ipf) for ipf in resize_factor] + [0])
            vox2ras0 = copy.copy(AFFINE)
            vox2ras0[:3, 0] = vox2ras0[:3, 0] / resize_factor[0]
            vox2ras0[:3, 1] = vox2ras0[:3, 1] / resize_factor[1]
            vox2ras0[:3, 3] = vox2ras0[:3, 3] - np.dot(AFFINE[:3, :3], aux.T)


        np.save(join(results_dir_sbj, stain, 'vox2ras.npy'), vox2ras0)

        flow = np.asarray(proxy_flow.dataobj)
        for slice in block_loader:
            sid = int(slice.sid)
            if sid not in [4,8,12,16,20,24,28,32]: continue
            sid_2str = "{:02d}".format(sid)
            print('#' + str(sid), end=' ', flush=True)
            slice_num = sid - 1

            # if exists(join(results_dir_sbj, stain, 'slice_' + sid_2str + '_' + str(downsample_factor) + 'D.jpg')):
            #     continue


            #### Read image and deform
            filename = SBJ_BLOCK + '_' + stain + '_' + sid_2str + '.jpg'
            filename_mask = SBJ_BLOCK + '_' + stain + '_' + sid_2str + '.mask.png'

            H_filepath = join(HISTO_DIR, stain, filename)
            H_mask_filepath = join(HISTO_DIR, stain, filename_mask)

            try:
                H = cv2.imread(H_filepath)
                M = cv2.imread(H_mask_filepath, flags=0)
                M = (M / np.max(M)) > 0.5
            except:
                missing_slices += [filename]
                continue

            if SUBJECT == 'P57-16' and bid == 'B4.1': #block B4.1 was flipped (up-down, so to find matching blocks we need to do that)
                Mup = np.zeros_like(M)
                Mup[:-7500] = M[7500:]
                Hup = np.zeros_like(H)
                Hup[:-7500] = H[7500:]
                M = np.flipud(Mup)
                H = np.flipud(Hup)

                del Mup
                del Hup

            resized_shape = tuple([int(i / (HISTO_res*downsample_factor) * BLOCK_res) for i in flow.shape[1:3]])
            factor = resized_shape[1]/flow.shape[2]
            field_i = factor*cv2.resize(flow[0, ..., slice_num], (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
            factor = resized_shape[0]/flow.shape[1]
            field_j = factor*cv2.resize(flow[1, ..., slice_num], (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)

            field = np.zeros((2,) + field_i.shape)
            field[0] = field_i
            field[1] = field_j

            del field_i, field_j

            if downsample_factor > 1:
                resized_shape = tuple([int(i / downsample_factor) for i in H.shape])
                M = (255 * M).astype('uint8')
                H = cv2.resize(H, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
                M = cv2.resize(M, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
                M = (M / np.max(M)) > 0.5

            H = cv2.cvtColor(H, cv2.COLOR_BGR2RGB)
            H_masked = H
            H_masked = np.zeros_like(H, dtype='uint8')
            for it_c in range(3):
                H_sl = H[..., it_c]
                H_sl[~M] = 0
                H_masked[..., it_c] = H_sl

            del H, M

            H_masked = deform2D(H_masked, field, mode='linear')
            H_masked = H_masked.astype('uint8')
            img = Image.fromarray(H_masked, mode='RGB')
            img.save(join(results_dir_sbj, stain, 'slice_' + sid_2str + '_' + str(downsample_factor) + 'D.jpg'), quality=90)

            del H_masked
print(missing_slices)
