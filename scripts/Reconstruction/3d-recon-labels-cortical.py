# py
import pdb
import subprocess
from argparse import ArgumentParser

# libraries imports
import cv2
import nibabel as nib
import numpy as np

# project imports
from dataset.data_loader import DataLoader
from utils.deformation_utils import deform2D
from config import config_donors, config_dev, config_database
from utils.recon_utils import *


# ******************************************************************************************************************** #
#
# This file uses the computed latent SVF from the ST algorithm to deform the initial labeled
# sections at the processing resolution.
# It uses the SVF computed using the ST algorithm specified at --reg_algorithm
#
# ******************************************************************************************************************** #


####################
# Input parameters #
####################

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='change_orientation', choices=list(config_donors.file_dict.keys()))
arg_parser.add_argument('--reg_algorithm', default='ST3_L1_RegNet_NN2',
                        choices=['Linear', 'RegNet', 'ST3_L1_RegNet_NN2',  'ST3_L1_RegNet_NN4'])
arg_parser.add_argument('--disable_one_hot_flag', action='store_false')
arg_parser.add_argument('--bid', default=None, nargs='+')

arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
ONE_HOT_FLAG = arguments.disable_one_hot_flag
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
    'RegNet': {'path': join(base_dir, 'Registration', 'RegNet'), 'stain': 'MRI_LFB'},
    'ST3_L1_RegNet_NN2': {'path': join(base_dir, 'ST3', 'l1', 'NN2'), 'stain': 'LFB'},
    'ST3_L1_RegNet_NN4': {'path': join(base_dir, 'ST3', 'l1', 'NN4'), 'stain': 'LFB'}
}

BLOCK_res = config_file_data.BLOCK_res
HISTO_res = config_file_data.HISTO_res
downsampleFactorHistoLabels = config_file_data.downsampleFactorHistoLabels

bid_list = initial_bid_list if initial_bid_list is not None else config_file_data.initial_bid_list
bid_list = {SUBJECT: bid_list}

data_loader = DataLoader(data_dict, included_blocks=bid_list)
subject_loader = data_loader.subject_dict[SUBJECT]

for bid in bid_list[SUBJECT]:

    print('')
    print('Block ID: ' + bid)
    if 'C' in bid or 'B' in bid:
        continue
    block_loader = subject_loader.block_dict[bid]
    sbj_block = BT_DB_blocks['SUBJECT'] + '_' + bid
    BLOCK_DIR = join(SLICES_DIR, bid)
    AFFINE = block_loader.vox2ras0

    results_dir_sbj = join(data_path_dict[reg_algorithm]['path'], bid)
    LABELS_DIR = join(results_dir_sbj, 'MRI_aparc')

    rotation_angle = get_rotation_angle(SUBJECT, bid)

    if reg_algorithm == 'Linear':
        nslices = np.max([int(k.sid) for k in block_loader])
        flow = []
        for it_slice, slice in enumerate(block_loader):

            sid = int(slice.sid)
            slice_num = sid - 1

            refFile = join(BLOCK_DIR, 'MRI', 'images', 'slice_' + "{:04d}".format(sid) + '.png')
            affineFile = join(BLOCK_DIR, data_path_dict[reg_algorithm]['stain'], 'affine', 'slice_' + "{:04d}".format(sid) + '.aff')
            dummyFileNifti = 'images_tmpw.nii.gz'

            subprocess.call([config_dev.TRANSFORMcmd, '-ref', refFile, '-disp', affineFile, dummyFileNifti],
                            stdout=subprocess.DEVNULL)

            proxy_aff = nib.load(dummyFileNifti)
            affine_field = np.asarray(proxy_aff.dataobj)[:, :, 0, 0, :]

            flow.append(np.transpose(affine_field, axes=[2, 1, 0]))

        flow = np.stack(flow, axis=-1)
        flow = flow[::-1]#(j,i) --> (i,j)

    else:
        proxy_flow = nib.load(join(results_dir_sbj, data_path_dict[reg_algorithm]['stain'] + '.totalflow.nii.gz'))
        flow = np.asarray(proxy_flow.dataobj)

    labels_volume = np.zeros(flow.shape[1:], dtype=np.uint16)
    missing_slices = []
    print('   Slice number:', end=' ', flush=True)
    for slice in block_loader:
        sid = int(slice.sid)
        slice_num = sid - 1
        sid_2str = "{:02d}".format(sid)
        if SUBJECT == 'P57-16' and bid == 'B5.1' and np.mod(slice_num, 4):
            print("[WARNING] Problem with B5.1 LABELS and volume dimensions. The label file adHocs seems to be padded compared to the original!")
            continue

        print('#' + str(sid), end=' ', flush=True)

        label_image = get_label_image(sid, sbj_block, LABELS_DIR)
        # if np.sum(label_image == 2008) > 0:
        #     label_image[label_image == 2029] = 2008
        #
        #     file1 = join(LABELS_DIR, sbj_block + '_' + sid_2str + '.nii.gz')
        #     print(file1)
        #     proxy = nib.load(file1)
        #     img = nib.Nifti1Image(label_image, proxy.affine)
        #     nib.save(img, file1)

        if label_image is None:
            print('- missing', end=' ', flush=True)
            missing_slices.append(slice_num)
            continue

        if sbj_block == 'EX9-19_P1.3' or sbj_block == 'P85-18_P3.2':
            if sid == 1:
                ishape = label_image.shape

            if sid not in [1,5,9,11,13,17,21,25,28,33,37]:
                label_image = label_image[:ishape[0], :ishape[1]]

        resized_shape = tuple([int(i * HISTO_res * downsampleFactorHistoLabels / BLOCK_res) for i in label_image.shape])

        if ONE_HOT_FLAG:
            unique_labels = np.unique(label_image)
            image_onehot_def = np.zeros(labels_volume.shape[:-1] + (len(unique_labels),), dtype='float32')
            for it_ul, ul in enumerate(unique_labels):
                image_onehot = np.zeros_like(label_image).astype('float32')
                image_onehot[label_image == ul] = 1

                image_onehot = cv2.resize(image_onehot, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
                # image_onehot = preprocess_label_image(image_onehot, resized_shape, SUBJECT, bid, rotation_angle, order=1)
                image_onehot_def[..., it_ul] = deform2D(image_onehot, flow[..., slice_num], mode='bilinear')

            image_def_unordered = np.argmax(image_onehot_def, axis=-1).astype('uint16')
            for it_ul, ul in enumerate(unique_labels): labels_volume[image_def_unordered == it_ul, slice_num] = ul


        else:
            image = preprocess_label_image(label_image, resized_shape, SUBJECT, bid, rotation_angle, order=0)
            labels_volume[..., slice_num] = deform2D(image, flow[..., slice_num], mode='nearest')

    labels_volume = postprocess_label_image(labels_volume, SUBJECT, bid, missing_slices)

    img = nib.Nifti1Image(labels_volume, AFFINE)
    nib.save(img, join(results_dir_sbj, 'LABELS.cortical.nii.gz'))
