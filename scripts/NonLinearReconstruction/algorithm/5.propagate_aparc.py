# py
import pdb
from os.path import join, exists
from os import makedirs, remove
import subprocess
from argparse import ArgumentParser

# libraries imports
import ssl # just to fix matlab engine collision
import nibabel as nib
import numpy as np
from skimage.transform import rotate as imrotate
from skimage import measure
from skimage.morphology import binary_erosion
import cv2
import matlab.engine
from PIL import Image
from scipy.io import savemat

# project imports
from dataset.data_loader import DataLoader
from utils.deformation_utils import deform2D
from setup_repo import *
from utils.recon_utils import read_slice_info, get_label_image, get_rotation_angle, preprocess_label_image
from config import config_donors, config_database
from utils.visualization import slices



# ******************************************************************************************************************** #
#
# This file uses the (inverse) computed latent SVF from the ST algorithm to deform the APARC files to the section space.
# Then, it propagates the labels and does some morphological operations trying to correct for obvious unconsistencies.
# It uses the SVF computed using the ST algorithm specified at --reg_algorithm
#
# ******************************************************************************************************************** #


print('\n\n\n\n\n')
print('# ------------------------- #')
print('# Propagate cortical labels #')
print('# ------------------------- #')
print('\n\n')


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
reg_algorithm = arguments.reg_algorithm
initial_bid_list = arguments.bid


BT_DB_blocks = config_database.get_lin_dict(SUBJECT)
BT_DB_slices = config_database.get_nonlin_dict(SUBJECT)
config_file_data = config_donors.file_dict[SUBJECT]
data_dict = {SUBJECT: BT_DB_slices}

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
slice_dict, mapping_dict = read_slice_info(SLICES_DIR)

for bid in bid_list[SUBJECT]:
    rotation_angle = get_rotation_angle(SUBJECT, bid)

    print('')
    print('Block ID: ' + bid)
    if 'B' in bid or 'C' in bid:
        continue

    block_loader = subject_loader.block_dict[bid]
    sbj_block = BT_DB_blocks['SUBJECT'] + '_' + bid
    BLOCK_DIR = join(SLICES_DIR, bid)
    HISTO_DIR = join(BT_DB_blocks['SLIDES_DIR'], BT_DB_blocks['SUBJECT'] + '_' + bid)
    LABELS_DIR = join(BT_DB_blocks['HISTO_LABELS_DIR'], bid)
    AFFINE = block_loader.vox2ras0

    results_dir_sbj = join(data_path_dict[reg_algorithm]['path'], bid)

    # proxy_flow = nib.load(join(results_dir_sbj, data_path_dict[reg_algorithm]['stain'] + '.flow_reverse.nii.gz'))
    # flow = np.asarray(proxy_flow.dataobj)



    results_dir_sbj_MRI = join(results_dir_sbj, 'MRI_aparc')
    if not exists(results_dir_sbj_MRI): makedirs(results_dir_sbj_MRI)
    results_dir_sbj_MRI_tmp = join(results_dir_sbj_MRI, 'tmp')
    if not exists(results_dir_sbj_MRI_tmp): makedirs(results_dir_sbj_MRI_tmp)

    proxy_tmp = nib.load(join(results_dir_sbj, data_path_dict[reg_algorithm]['stain'] + '.nii.gz'))
    affine = proxy_tmp.affine

    proxy = nib.load(join(REG_BLOCKS_DIR, sbj_block + '_volume.aparc.reg.mgz'))
    mri_vox2ras0 = proxy.affine
    mri_vol = np.asarray(proxy.dataobj)

    proxy = nib.load(join(REG_BLOCKS_DIR, sbj_block + '_volume.mri.mask.reg.closing.mgz'))
    mri_mask = np.asarray(proxy.dataobj)



    if any([l in np.arange(1000, 1036) for l in np.unique(mri_vol)]):
        for l in np.arange(1000, 1036):
            mri_vol[mri_vol == l] += 1000


    #allocate label 2000
    if 2000 in np.unique(mri_vol):
        unknown_label = (mri_vol == 2000).astype('uint8')

        # Get the largest blob
        all_blobs, num_blobs = measure.label(unknown_label, connectivity=1, return_num=True)
        all_blobs_count = np.bincount(all_blobs[unknown_label > 0])
        from utils import image_utils
        from scipy.ndimage import distance_transform_edt

        for it_abc, abc in enumerate(all_blobs_count):

            if abc == 0: continue
            tmp_label = all_blobs == it_abc

            mask_crop, crop_coord = image_utils.crop_label(tmp_label, margin=30)
            mri_crop = image_utils.apply_crop(mri_vol, crop_coord)
            unknown_labelmap = np.where(mri_crop == 2000)

            winningDist = 1e10 * np.ones(unknown_labelmap[0].shape)
            labelDist = np.zeros(unknown_labelmap[0].shape)
            for lab in np.unique(mri_crop):
                if lab <= 2000: continue
                mask_l = mri_crop == lab
                if np.sum(mask_l) == 0: continue
                dist = distance_transform_edt(~mask_l)
                for it_ev in range(unknown_labelmap[0].shape[0]):
                    if dist[unknown_labelmap[0][it_ev], unknown_labelmap[1][it_ev], unknown_labelmap[2][it_ev]] < winningDist[it_ev]:
                        winningDist[it_ev] = dist[unknown_labelmap[0][it_ev], unknown_labelmap[1][it_ev], unknown_labelmap[2][it_ev]]
                        labelDist[it_ev] = lab

            for it_ev in range(unknown_labelmap[0].shape[0]):
                mri_crop[unknown_labelmap[0][it_ev], unknown_labelmap[1][it_ev], unknown_labelmap[2][it_ev]] = labelDist[it_ev]

            mri_vol[
                crop_coord[0][0]: crop_coord[0][1],
                crop_coord[1][0]: crop_coord[1][1],
                crop_coord[2][0]: crop_coord[2][1]
            ] = mri_crop

    # If we run out of mask for whatever reason, we copy from previous slices
    # Not the prettiest, but it works. TODO: improve?
    aux = np.squeeze(np.sum(np.sum(mri_mask, axis=0), axis=0))
    ok = np.where(aux > 5000)[0]
    nok = np.where(aux < 5000)[0]
    for j in nok:
        dist = np.abs(j - ok)
        idx = np.argmin(dist)
        ref = ok[idx]
        mri_mask[..., j] = mri_mask[..., ref]

    mri_mask_bool = mri_mask > 0.5
    mri_vol[~mri_mask_bool] = 0
    mri_vol = np.double(mri_vol)

    angle = -mapping_dict[bid][0]     # MRI->to histo, while the angle was computed the other way around.
    if bid == 'B2.1' and 'P57-16' in SUBJECT:
        angle += 35

    if mapping_dict[bid][1]:
        mri_vol = np.fliplr(mri_vol)  # flip
        mri_mask = np.fliplr(mri_mask)  # flip

    if bid == 'B4.1' and 'P57-16' in SUBJECT:
        mri_vol = np.flipud(mri_vol)  # flip
        mri_mask = np.flipud(mri_mask)  # flip

    mri_vol_list = []
    mri_mask_list = []
    mri_mask = mri_mask.astype('float')
    for it_z in range(mri_vol.shape[2]):
        mri_vol_list.append(imrotate(mri_vol[..., it_z], angle, resize=True,order=0))  # rotate
        mri_mask_list.append(imrotate(mri_mask[..., it_z], angle, resize=True,order=0))  # rotate

    mri_vol = np.transpose(np.asarray(mri_vol_list), [1, 2, 0])
    mri_mask = np.transpose(np.asarray(mri_mask_list).astype('uint8'), [1, 2, 0])

    if mapping_dict[bid][2]:
        mri_vol = mri_vol[..., ::-1]  # change order of the most posterior and most lateral parts of the cerebrum/cerebellum
        mri_mask = mri_mask[..., ::-1]  # idem


    print('   Slice number:', end=' ', flush=True)
    for it_slice, slice in enumerate(block_loader):
        sid = int(slice.sid)
        # if sid != 6: continue
        print('#' + str(sid), end=' ', flush=True)
        # if sid != 17: continue
        slice_num = sid - 1
        sid_2str = "{:02d}".format(sid)
        sid_4str = "{:04d}".format(sid)


        # Get orig shape
        filename = BT_DB_blocks['SUBJECT'] + '_' + bid + '_LFB_' + sid_2str
        HISTO_DIR = join(BT_DB_blocks['SLIDES_DIR'], BT_DB_blocks['SUBJECT'] + '_' + bid)
        histo_mask_filepath = join(HISTO_DIR, 'LFB', filename + '.mask.png')
        orig_image = cv2.imread(histo_mask_filepath)
        orig_shape = orig_image.shape[:2]


        ## Read image and deform
        field = np.load(join(results_dir_sbj, 'flow_reverse', 'slice_' + sid_4str + '.npy'))
        Maparc = mri_vol[..., it_slice].astype('int16')
        image = deform2D(Maparc, field, mode='nearest').astype('int')


        ## Read LFB labels
        label_image = get_label_image(sid, sbj_block, LABELS_DIR)
        if label_image is None: continue
        label_image = preprocess_label_image(label_image, SUBJECT, bid, rotation_angle,
                                             orig_shape=orig_shape, resized_shape=None, order=0,
                                             sid_2str=sid_2str, BT_DB_blocks=BT_DB_blocks)

        res_shape = label_image.shape
        mask_image = label_image == 3


        # Erode the cortical mask a bit for the first pass
        strel_mask = np.ones((20, 20))
        strel_labels = np.ones((35, 35))
        mask_image_ero = binary_erosion(mask_image, strel_mask)
        mask_image_low = cv2.resize(mask_image_ero.astype('float'),
                                    (image.shape[1], image.shape[0]),
                                    interpolation=cv2.INTER_LINEAR) > 0.5
        image[~mask_image_low] = 0


        # First pass: keep all blobs greater than a threshold
        unique_aparc = [0] + list(filter(lambda x: x > 1000, np.unique(image)))
        image_onehot = np.zeros(res_shape + (len(unique_aparc),))

        for it_label, label in enumerate(unique_aparc):
            tmp_label = (image == label).astype('uint8')

            if label != 0:
                # Get the largest blob
                all_blobs, num_blobs = measure.label(tmp_label, connectivity=1, return_num=True)
                all_blobs_count = np.bincount(all_blobs[tmp_label > 0])
                if np.max(all_blobs_count) < 400:
                    uab = np.argmax(all_blobs_count)
                    idx = np.where(all_blobs != uab)
                    tmp_label[idx] = 0

                else:
                    for it_abc, abc in enumerate(all_blobs_count):
                        if abc < 200:
                            idx = np.where(all_blobs == it_abc)
                            tmp_label[idx] = 0

            # Resize
            tmp_onehot = cv2.resize(tmp_label, (res_shape[1], res_shape[0]), interpolation=cv2.INTER_LINEAR)
            image_onehot[..., it_label] = tmp_onehot  # binary_opening(tmp_onehot, strel_labels)#

        if unique_aparc:
            # First pass with the hugely eroded mask
            savemat(join(results_dir_sbj_MRI_tmp, BT_DB_blocks['SUBJECT'] + '_' + bid + '_' + sid_2str + '.mat'),
                    {'image': image_onehot, 'mask': mask_image, 'labels': unique_aparc, 'vox2ras': affine})

            eng = matlab.engine.start_matlab()
            eng.addpath(join(REPO_DIR, 'utils'))
            eng.highres_fmm(bid, BT_DB_blocks['SUBJECT'], sid_2str, results_dir_sbj, nargout=0)
            remove(join(results_dir_sbj_MRI_tmp, BT_DB_blocks['SUBJECT'] + '_' + bid + '_' + sid_2str + '.mat'))

            # Second pass to fill the rest of the mask
            proxy = nib.load(join(results_dir_sbj_MRI, BT_DB_blocks['SUBJECT'] + '_' + bid + '_' + sid_2str + '.nii.gz'))
            image_second = np.squeeze(np.asarray(proxy.dataobj))

            mask_image_low = cv2.resize(mask_image.astype('float'), (image.shape[1], image.shape[0]),  interpolation=cv2.INTER_LINEAR) > 0.5
            image_onehot = np.zeros(res_shape + (len(unique_aparc),))
            for it_label, label in enumerate(unique_aparc):
                if label in unique_aparc and label != 0:
                    tmp_label = (image_second == label).astype('uint8')
                else:
                    tmp_label = cv2.resize((image == label).astype('uint8'), (res_shape[1], res_shape[0]), interpolation=cv2.INTER_LINEAR)

                # Resize
                image_onehot[..., it_label] = binary_erosion(tmp_label, strel_labels)

            savemat(join(results_dir_sbj_MRI_tmp, BT_DB_blocks['SUBJECT'] + '_' + bid + '_' + sid_2str + '.mat'),
                    {'image': image_onehot, 'mask': mask_image, 'labels': unique_aparc, 'vox2ras': affine})

            eng = matlab.engine.start_matlab()
            eng.addpath(join(REPO_DIR, 'utils'))
            eng.highres_fmm(bid, BT_DB_blocks['SUBJECT'], sid_2str,  results_dir_sbj, nargout=0)
            remove(join(results_dir_sbj_MRI_tmp, BT_DB_blocks['SUBJECT'] + '_' + bid + '_' + sid_2str + '.mat'))



        proxy = nib.load(join(results_dir_sbj_MRI, BT_DB_blocks['SUBJECT'] + '_' + bid + '_' + sid_2str + '.nii.gz'))
        data = np.array(proxy.dataobj)

        file1 = join(LABELS_DIR, sbj_block + '_' + "{:02d}".format(sid) + '.nii.gz')
        proxy_lab = nib.load(file1)
        img = nib.Nifti1Image(data, proxy_lab.affine)
        nib.save(img,join(results_dir_sbj_MRI, BT_DB_blocks['SUBJECT'] + '_' + bid + '_' + sid_2str + '.nii.gz'))

