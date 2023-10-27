# py
import pdb
from os import makedirs
from argparse import ArgumentParser

# libraries imports
from scipy.ndimage import distance_transform_edt

# project imports
from dataset.data_loader import DataLoader
from utils.deformation_utils import deform2D
from config import config_donors, config_dev, config_database
from utils.recon_utils import *


# ******************************************************************************************************************** #
#
# This file merges the manually labeled sections with the results of 5.propagate_aparc. A final step removes the
# corpus callosum labels with the nearest neighbor.
# It uses the SVF computed using the ST algorithm specified at --reg_algorithm
#
# ******************************************************************************************************************** #


print('\n\n\n\n\n')
print('# ----------------------- #')
print('# Merging cortical labels #')
print('# ----------------------- #')
print('\n\n')

####################
# Input parameters #
####################

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='change_orientation', choices=list(config_donors.file_dict.keys()))
arg_parser.add_argument('--reg_algorithm', default='ST3_L1_RegNet_NN2',
                        choices=['Linear', 'RegNet', 'ST3_L1_RegNet_NN2',  'ST3_L1_RegNet_NN4'])
arg_parser.add_argument('--disable_one_hot_flag', action='store_false')
arg_parser.add_argument('--cortical_flag', action='store_true')
arg_parser.add_argument('--downsample_factor', default=1, type=int)
arg_parser.add_argument('--bid', default=None, nargs='+')

arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
ONE_HOT_FLAG = arguments.disable_one_hot_flag
cortical_flag = arguments.cortical_flag
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
initial_bid_list = initial_bid_list if initial_bid_list is not None else config_file_data.initial_bid_list

print('Step 1: Merge cortical and subcortical labels')
for it_bid, bid in enumerate(initial_bid_list):
    print('')
    print('Block ID: ' + bid)

    block_loader = subject_loader.block_dict[bid]
    sbj_block = BT_DB_blocks['SUBJECT'] + '_' + bid
    BLOCK_DIR = join(SLICES_DIR, bid)

    CORTICAL_LABELS_DIR = join(data_path_dict[reg_algorithm]['path'], bid, 'MRI_aparc')
    LABELS_DIR = join(BT_DB_blocks['HISTO_LABELS_DIR'], bid)
    FINAL_LABELS_DIR = join(BT_DB_blocks['FINAL_HISTO_LABELS_DIR'], bid)

    results_dir_sbj = join(data_path_dict[reg_algorithm]['path'], bid)
    proxy_flow = nib.load(join(results_dir_sbj, 'LFB.totalflow.nii.gz'))
    flow = np.asarray(proxy_flow.dataobj)

    if not exists(join(results_dir_sbj, 'LABELS')):
        makedirs(join(results_dir_sbj, 'LABELS'))

    labels_volume = np.zeros(flow.shape[1:])
    missing_slices = []
    missing_slices_cort = []
    rotation_angle = get_rotation_angle(SUBJECT, bid)
    print('Slice number:', end=' ', flush=True)
    for slice in block_loader:
        sid = int(slice.sid)
        # if sid != 5: continue
        slice_num = sid - 1
        it_z = sid - 1
        sid_2str = "{:02d}".format(sid)
        sid_4str = "{:04d}".format(sid)
        print('#' + str(sid), end=' ', flush=True)

        # Get orig shape
        filename = BT_DB_blocks['SUBJECT'] + '_' + bid + '_LFB_' + sid_2str
        HISTO_DIR = join(BT_DB_blocks['SLIDES_DIR'], BT_DB_blocks['SUBJECT'] + '_' + bid)
        histo_mask_filepath = join(HISTO_DIR, 'LFB', filename + '.mask.png')
        orig_image = cv2.imread(histo_mask_filepath)
        orig_shape = orig_image.shape[:2]

        # Read cortical label
        cortical_label_image = get_label_image(sid, sbj_block, CORTICAL_LABELS_DIR)
        if cortical_label_image is None:
            missing_slices_cort.append(slice_num)
            continue

        if any([l in np.arange(1000, 1036) for l in np.unique(cortical_label_image)]):
            for l in np.arange(1000, 1036):
                cortical_label_image[cortical_label_image==l] += 1000

        # cortical_label_image = preprocess_label_image(cortical_label_image, SUBJECT, bid, -rotation_angle,
        #                                               resized_shape=None, orig_shape=orig_shape, order=0, inverse=True) # inverse rotate the image to match the other one.
        cortical_mask = cortical_label_image > 0


        # Read whole brain label
        label_image = get_label_image(sid, sbj_block, LABELS_DIR)
        if label_image is None:
            missing_slices.append(slice_num)
            continue
        label_image = preprocess_label_image(label_image, SUBJECT, bid, rotation_angle,
                                             orig_shape=orig_shape, resized_shape=None, order=0)

        if SUBJECT == 'P57-16' and bid == 'B4.1':  # block B4.1 was flipped (up-down, so to find matching blocks we need to do that)
            resized_shape = tuple([int(i * downsampleFactorHistoLabels) for i in label_image.shape])
            L = cv2.resize(label_image, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_NEAREST)
            L = imrotate(L, -90, resize=True, order=0)  # rotate
            Lup = np.zeros_like(L)
            Lup[:-7500] = L[7500:]
            Lup = np.flipud(Lup)  # are 90 degrees rotated all labels
            Lup = imrotate(Lup, 90, resize=True, order=0)  # rotate
            label_image = cv2.resize(Lup, (label_image.shape[1], label_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            label_image = np.fliplr(label_image)

            del L
            del Lup

        label_image = cv2.resize(label_image, (cortical_mask.shape[1], cortical_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        cortical_mask[label_image != 3] = 0
        cortical_label_image[~cortical_mask] = 0
        label_image[cortical_mask] = cortical_label_image[cortical_mask]

        empty_voxels = np.where(label_image == 3)
        if empty_voxels:

            # Read CORTICAL LABEL
            cortical_label_image = get_label_image(sid, sbj_block, CORTICAL_LABELS_DIR)
            if cortical_label_image is None:
                missing_slices_cort.append(slice_num)
                continue
            if any([l in np.arange(1000, 1036) for l in np.unique(cortical_label_image)]):
                for l in np.arange(1000, 1036):
                    cortical_label_image[cortical_label_image == l] += 1000
            # cortical_label_image = preprocess_label_image(cortical_label_image, None, None, SUBJECT, bid, -rotation_angle,
            #                                               order=0, inverse=True)
            cortical_mask = cortical_label_image > 0

            winningDist = 1e10 * np.ones(empty_voxels[0].shape)
            labelDist = np.zeros(empty_voxels[0].shape)
            for lab in np.unique(cortical_label_image):
                if lab == 0: continue
                mask_l = cortical_label_image == lab
                dist = distance_transform_edt(~mask_l)
                for it_ev in range(empty_voxels[0].shape[0]):
                    if dist[empty_voxels[0][it_ev], empty_voxels[1][it_ev]] < winningDist[it_ev]:
                        winningDist[it_ev] = dist[empty_voxels[0][it_ev], empty_voxels[1][it_ev]]
                        labelDist[it_ev] = lab

            for it_ev in range(empty_voxels[0].shape[0]):
                label_image[empty_voxels[0][it_ev], empty_voxels[1][it_ev]] = labelDist[it_ev]


        file1 = join(LABELS_DIR, sbj_block + '_' + "{:02d}".format(sid) + '.nii.gz')
        proxy_lab = nib.load(file1)
        if not exists(join(FINAL_LABELS_DIR)):
            makedirs(join(FINAL_LABELS_DIR))
        img = nib.Nifti1Image(label_image, proxy_lab.affine)
        nib.save(img, join(FINAL_LABELS_DIR, sbj_block + '_' + "{:02d}".format(sid) + '.nii.gz'))

print('')
print('Step 2: correct corpus callosum labels:')
for it_bid, bid in enumerate(initial_bid_list):
    print('')
    print('Block ID: ' + bid)

    block_loader = subject_loader.block_dict[bid]
    sbj_block = BT_DB_blocks['SUBJECT'] + '_' + bid
    BLOCK_DIR = join(SLICES_DIR, bid)

    CORTICAL_LABELS_DIR = join(data_path_dict[reg_algorithm]['path'], bid, 'MRI_aparc')
    LABELS_DIR = join(BT_DB_blocks['HISTO_LABELS_DIR'], bid)
    FINAL_LABELS_DIR = join(BT_DB_blocks['FINAL_HISTO_LABELS_DIR'], bid)

    results_dir_sbj = join(data_path_dict[reg_algorithm]['path'], bid)
    proxy_flow = nib.load(join(results_dir_sbj, 'LFB.totalflow.nii.gz'))
    flow = np.asarray(proxy_flow.dataobj)

    if not exists(join(results_dir_sbj, 'LABELS')):
        makedirs(join(results_dir_sbj, 'LABELS'))

    labels_volume = np.zeros(flow.shape[1:])
    missing_slices = []
    missing_slices_cort = []
    rotation_angle = get_rotation_angle(SUBJECT, bid)
    print('   Slice number:', end=' ', flush=True)
    for slice in block_loader:
        sid = int(slice.sid)
        slice_num = sid - 1
        it_z = sid - 1

        # if sid != 5: continue
        print('#' + str(sid), end=' ', flush=True)

        if not exists(join(FINAL_LABELS_DIR, sbj_block + '_' + "{:02d}".format(sid) + '.nii.gz')): continue

        proxy_lab = nib.load(join(FINAL_LABELS_DIR, sbj_block + '_' + "{:02d}".format(sid) + '.nii.gz'))
        label_image = np.array(proxy_lab.dataobj)

        # Correct corpus_callosum
        corpus_callosum = np.where(label_image == 2004)
        if corpus_callosum[0].shape[0] != 0:

            # Read CORTICAL LABEL
            cortical_label_image = get_label_image(sid, sbj_block, CORTICAL_LABELS_DIR)
            if cortical_label_image is None:
                missing_slices_cort.append(slice_num)
                continue
            if any([l in np.arange(1000, 1036) for l in np.unique(cortical_label_image)]):
                for l in np.arange(1000, 1036):
                    cortical_label_image[cortical_label_image == l] += 1000
            # cortical_label_image = preprocess_label_image(cortical_label_image, None, None, SUBJECT, bid, -rotation_angle,
            #                                               order=0, inverse=True)
            cortical_mask = cortical_label_image > 0

            winningDist = 1e10 * np.ones(corpus_callosum[0].shape)
            labelDist = np.zeros(corpus_callosum[0].shape)
            for lab in [2002, 2023, 2010]:
                if lab == 0: continue
                mask_l = cortical_label_image == lab
                if np.sum(mask_l) == 0: continue
                dist = distance_transform_edt(~mask_l)
                for it_ev in range(corpus_callosum[0].shape[0]):
                    if dist[corpus_callosum[0][it_ev], corpus_callosum[1][it_ev]] < winningDist[it_ev]:
                        winningDist[it_ev] = dist[corpus_callosum[0][it_ev], corpus_callosum[1][it_ev]]
                        labelDist[it_ev] = lab

            for it_ev in range(corpus_callosum[0].shape[0]):
                label_image[corpus_callosum[0][it_ev], corpus_callosum[1][it_ev]] = labelDist[it_ev]

            img = nib.Nifti1Image(label_image, proxy_lab.affine)
            nib.save(img, join(FINAL_LABELS_DIR, sbj_block + '_' + "{:02d}".format(sid) + '.nii.gz'))

        # Correct empty voxels
        empty_voxels = np.where(label_image == 3)
        if empty_voxels[0].shape[0] == 0:

            # Read CORTICAL LABEL
            cortical_label_image = get_label_image(sid, sbj_block, CORTICAL_LABELS_DIR)
            if cortical_label_image is None:
                missing_slices_cort.append(slice_num)
                continue
            if any([l in np.arange(1000, 1036) for l in np.unique(cortical_label_image)]):
                for l in np.arange(1000, 1036):
                    cortical_label_image[cortical_label_image == l] += 1000
            # cortical_label_image = preprocess_label_image(cortical_label_image, None, None, SUBJECT, bid, -rotation_angle,
            #                                               order=0, inverse=True)
            cortical_mask = cortical_label_image > 0

            winningDist = 1e10 * np.ones(empty_voxels[0].shape)
            labelDist = np.zeros(empty_voxels[0].shape)
            for lab in np.unique(cortical_label_image):
                if lab == 0: continue
                mask_l = cortical_label_image == lab
                dist = distance_transform_edt(~mask_l)
                for it_ev in range(empty_voxels[0].shape[0]):
                    if dist[empty_voxels[0][it_ev], empty_voxels[1][it_ev]] < winningDist[it_ev]:
                        winningDist[it_ev] = dist[empty_voxels[0][it_ev], empty_voxels[1][it_ev]]
                        labelDist[it_ev] = lab

            for it_ev in range(empty_voxels[0].shape[0]):
                label_image[empty_voxels[0][it_ev], empty_voxels[1][it_ev]] = labelDist[it_ev]

            img = nib.Nifti1Image(label_image, proxy_lab.affine)
            nib.save(img, join(FINAL_LABELS_DIR, sbj_block + '_' + "{:02d}".format(sid) + '.nii.gz'))

print('\n[FI - Merge cortical labels] Ended successfully.')
