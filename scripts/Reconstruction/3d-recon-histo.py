# py
from os.path import join, exists
from os import makedirs, remove
from argparse import ArgumentParser

# libraries imports
import nibabel as nib
import numpy as np
import cv2

# project imports
from dataset.data_loader import DataLoader
from utils.deformation_utils import deform2D
from config import config_donors, config_dev, config_database
from utils import recon_utils

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
arg_parser.add_argument('--bid', default=None, nargs='+')
arg_parser.add_argument('--reg_algorithm', default='ST3_L1_RegNet_NN2',
                        choices=['Linear', 'SbR', 'RegNet', 'ST3_L1_RegNet_NN4', 'ST3_L1_RegNet_NN2'])

arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
reg_algorithm = arguments.reg_algorithm
initial_bid_list = arguments.bid

BT_DB_blocks = config_database.get_lin_dict(SUBJECT)
BT_DB_slices = config_database.get_nonlin_dict(SUBJECT)
data_dict = {SUBJECT: BT_DB_slices}
config_file_data = config_donors.file_dict[SUBJECT]
parameter_dict_MRI = config_dev.CONFIG_DICT['MRI']

RESULTS_DIR = join(BT_DB_blocks['RESULTS_DIR'], 'results')
REG_BLOCKS_DIR = join(RESULTS_DIR, 'reg_blocks')
SLICES_DIR = join(RESULTS_DIR, 'slices')
SLICES_AFFINE_DIR = join(RESULTS_DIR, 'slices_affine')

base_dir = join(config_file_data.BASE_DIR, 'NonlinearRegistration')
data_path_dict = {
    'RegNet': {'path': join(base_dir, 'Registration'), 'stain': ['MRI_LFB', 'MRI_HE']},
    'ST3_L1_RegNet_NN2': {'path': join(base_dir, 'ST3', 'l1', 'NN2'), 'stain': ['LFB', 'HE']},
    'ST3_L1_RegNet_NN4': {'path': join(base_dir, 'ST3', 'l1', 'NN4'), 'stain': ['LFB', 'HE']}
}

BLOCK_res = config_file_data.BLOCK_res
HISTO_res = config_file_data.HISTO_res

bid_list = initial_bid_list if initial_bid_list is not None else config_file_data.initial_bid_list
bid_list = {SUBJECT: bid_list}

data_loader = DataLoader(data_dict, included_blocks=bid_list)
subject_loader = data_loader.subject_dict[SUBJECT]

for it_bid, bid in enumerate(bid_list[SUBJECT]):

    print('')
    print('Block ID: ' + bid)

    block_loader = subject_loader.block_dict[bid]

    SBJ_BLOCK = BT_DB_blocks['SUBJECT'] + '_' + bid
    BLOCK_DIR = join(SLICES_DIR, bid)
    HISTO_DIR = join(BT_DB_blocks['SLIDES_DIR'], SBJ_BLOCK)
    LABELS_DIR = join(BT_DB_blocks['HISTO_LABELS_DIR'], bid)
    AFFINE = block_loader.vox2ras0

    rotation_angle = recon_utils.get_rotation_angle(SUBJECT, bid)
    for stain in data_path_dict[reg_algorithm]['stain']:

        print(' - ' + stain)

        results_dir_sbj = join(data_path_dict[reg_algorithm]['path'], bid)
        if not exists(join(results_dir_sbj, stain)):
            makedirs(join(results_dir_sbj, stain))

        # if exists(join(results_dir_sbj, stain + '_IMAGE' + '.nii.gz')):
        #     print(' - ' + stain + ' already processd')
        #     continue

        print('   Slice number:', end=' ', flush=True)

        proxy_flow = nib.load(join(results_dir_sbj, stain + '.totalflow.nii.gz'))
        flow = np.asarray(proxy_flow.dataobj)

        image_volume = np.zeros(flow.shape[1:] + (3,), dtype='uint8')
        label_proxy = nib.load(join(results_dir_sbj, 'LABELS.nii.gz'))
        for slice in block_loader:

            sid = int(slice.sid)
            sid_2str = "{:02d}".format(sid)
            slice_num = sid - 1

            print('#' + str(sid), end=' ', flush=True)

            # Read original images
            filename = SBJ_BLOCK + '_' + stain + '_' + sid_2str + '.jpg'
            filename_mask = SBJ_BLOCK + '_' + stain + '_' + sid_2str + '.mask.png'

            H_filepath = join(HISTO_DIR, stain, filename)
            H_mask_filepath = join(HISTO_DIR, stain, filename_mask)
            H = cv2.imread(H_filepath)
            M = cv2.imread(H_mask_filepath, flags=0)
            if M is None or H is None: continue
            M = (M / np.max(M)) > 0.5

            resized_shape = tuple([int(i * HISTO_res / BLOCK_res) for i in M.shape])
            M = (255 * M).astype('uint8')
            H = cv2.resize(H, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
            M = cv2.resize(M, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
            H = cv2.cvtColor(H, cv2.COLOR_BGR2RGB)
            H_masked = np.zeros_like(H, dtype='uint8')

            M = (M / np.max(M)) > 0.5

            for it_c in range(3):
                H_sl = H[..., it_c]
                H_sl[~M] = 0
                H_masked[..., it_c] = H_sl

            del H
            del M

            image_volume[..., slice_num, :] = deform2D(H_masked, flow[..., slice_num], mode='linear')

            # if slice_num >= label_proxy.shape[-1]:
            #     M_filepath = join(HISTO_DIR, stain, filename_mask)
            #     M = cv2.imread(M_filepath, flags=0)
            #     if M is None: continue
            #     M = (255 * M).astype('uint8')
            #     M = cv2.resize(M, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_NEAREST)
            #     M = (M / np.max(M)) > 0.5
            #
            # else:
            #     L = np.asarray(label_proxy.dataobj[..., slice_num])
            #     L = recon_utils.preprocess_label_image(L, resized_shape, SUBJECT, bid, rotation_angle, order=0)
            #     M = np.ones(L.shape, dtype='bool')
            #     for l in [0, 50, 383, 620, 688, 328]:
            #         M[L==l] = False


        img = nib.Nifti1Image(image_volume, AFFINE)
        nib.save(img, join(results_dir_sbj, stain + '_IMAGE' + '.nii.gz'))

        del image_volume

        print('')

