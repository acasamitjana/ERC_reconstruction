# py
import pdb
import subprocess
from argparse import ArgumentParser

# libraries imports
import nibabel as nib
import numpy as np
from PIL import Image

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
arg_parser.add_argument('--final_labels', action='store_true')
arg_parser.add_argument('--bid', default=None, nargs='+')

arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
ONE_HOT_FLAG = arguments.disable_one_hot_flag
FINAL_LABELS_FLAG = arguments.final_labels
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

    block_loader = subject_loader.block_dict[bid]
    sbj_block = BT_DB_blocks['SUBJECT'] + '_' + bid
    BLOCK_DIR = join(SLICES_DIR, bid)
    if FINAL_LABELS_FLAG and ('C' not in bid and 'B' not in bid):
        LABELS_DIR = join(BT_DB_blocks['FINAL_HISTO_LABELS_DIR'], bid)
    else:
        LABELS_DIR = join(BT_DB_blocks['HISTO_LABELS_DIR'], bid)

    AFFINE = block_loader.vox2ras0

    results_dir_sbj = join(data_path_dict[reg_algorithm]['path'], bid)

    rotation_angle = get_rotation_angle(SUBJECT, bid)
    print(rotation_angle)
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
    orig_volume = np.zeros(flow.shape[1:])
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

        # Get orig shape
        filename = BT_DB_blocks['SUBJECT'] + '_' + bid + '_LFB_' + sid_2str
        HISTO_DIR = join(BT_DB_blocks['SLIDES_DIR'], BT_DB_blocks['SUBJECT'] + '_' + bid)
        histo_mask_filepath = join(HISTO_DIR, 'LFB', filename + '.mask.png')
        orig_image = cv2.imread(histo_mask_filepath)
        orig_shape = orig_image.shape[:2]

        label_image = get_label_image(sid, sbj_block, LABELS_DIR)
        if label_image is None:
            missing_slices.append(slice_num)
            continue

        if sbj_block == 'P58-16_B3.1':
            label_image = label_image[:, 158:-180]
        elif sbj_block == 'P58-16_B4.1':
            label_image = label_image[:, 34:-28]

        resized_shape = tuple([int(i * HISTO_res / BLOCK_res) for i in orig_shape])
        if ONE_HOT_FLAG:
            unique_labels = np.unique(label_image)
            image_onehot_def = np.zeros(labels_volume.shape[:-1] + (len(unique_labels),), dtype='float32')
            for it_ul, ul in enumerate(unique_labels):
                image_onehot = np.zeros_like(label_image).astype('float32')
                image_onehot[label_image == ul] = 1

                if not (FINAL_LABELS_FLAG and ('C' not in bid and 'B' not in bid)):
                    image_onehot = preprocess_label_image(image_onehot, SUBJECT, bid, rotation_angle, order=1,
                                                          orig_shape=orig_shape, resized_shape=resized_shape)
                elif SUBJECT == 'P41-16' and bid not in ['A1.2', 'A2.2', 'A3.2', 'P1.3', 'P2.4', 'P3.3', 'P4.1']:
                    image_onehot = preprocess_label_image(image_onehot, SUBJECT, bid, rotation_angle, order=1,
                                                          orig_shape=orig_shape, resized_shape=resized_shape)
                else:
                    image_onehot = cv2.resize(image_onehot, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)

                image_onehot_def[..., it_ul] = deform2D(image_onehot, flow[..., slice_num], mode='linear')

            image_def_unordered = np.argmax(image_onehot_def, axis=-1).astype('uint16')
            for it_ul, ul in enumerate(unique_labels): labels_volume[image_def_unordered == it_ul, slice_num] = ul


        else:
            if not (FINAL_LABELS_FLAG and ('C' not in bid and 'B' not in bid)):
                label_image = preprocess_label_image(label_image, SUBJECT, bid, rotation_angle, order=0,
                                                     orig_shape=orig_shape, resized_shape=resized_shape)
            elif SUBJECT == 'P41-16' and bid not in ['A1.2', 'A2.2', 'A3.2', 'P1.3', 'P2.4', 'P3.3', 'P4.1']:
                image_onehot = preprocess_label_image(label_image, SUBJECT, bid, rotation_angle, order=0,
                                                      orig_shape=orig_shape, resized_shape=resized_shape)
            else:
                label_image = cv2.resize(label_image, (resized_shape[1], resized_shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
            labels_volume[..., slice_num] = deform2D(label_image, flow[..., slice_num], mode='nearest')


    labels_volume = postprocess_label_image(labels_volume, SUBJECT, bid, missing_slices)

    img = nib.Nifti1Image(labels_volume, AFFINE)
    nib.save(img, join(results_dir_sbj, 'LABELS.nii.gz'))

print('')