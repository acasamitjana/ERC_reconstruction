from os.path import join, exists
import subprocess
import os
import pdb

from joblib import delayed, Parallel
from scipy.ndimage import distance_transform_edt, binary_erosion

import nibabel as nib
import numpy as np

from config import config_donors, config_database
from argparse import ArgumentParser
from utils import recon_utils, image_utils

DATA_DIR = '/home/acasamitjana/Data/ERC_segmentation'

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='P57-16', choices=['P57-16', 'P41-16', 'P58-16', 'P85-18', 'EX9-19'])
arg_parser.add_argument('--res', default=0.2, type=float)
arg_parser.add_argument('--nc', type=int, default=3, choices=[2, 3], help='Number of contrasts')
arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
arg_parser.add_argument('--nn', type=int, default=4, help='Number of neighbours')
arg_parser.add_argument('--observations', default='RegNet', choices=['RegNet', 'NiftyReg'])


arguments = arg_parser.parse_args()
subject = arguments.subject
target_res = arguments.res

ncontrasts = arguments.nc
nneighbours = arguments.nn
cost = arguments.cost
observations = arguments.observations

BT_DB = config_database.get_nonlin_dict(subject)
BT_DB_MRI = config_database.get_lin_dict(subject)
config_file_data = config_donors.file_dict[subject]

algorithm_dir = join(config_file_data.BASE_DIR, 'NonlinearRegistration')
blockspath = join(algorithm_dir, 'ST' + str(ncontrasts), cost, 'NN' + str(nneighbours))

outdir = join(blockspath, 'mosaic_' + str(target_res))
outdirtmp = join(outdir, 'tmp')
if not exists(outdirtmp): os.makedirs(outdirtmp)

labelsfile = join(outdir, 'LABELS.nii.gz')
labelsfile_filled_tmp = join(outdir, 'LABELS.filled.tmp.nii.gz')
labelsfile_filled = join(outdir, 'LABELS.filled.nii.gz')

asegfile = join(BT_DB_MRI['BASE_DIR'], 'MRI', 'aseg.rasoriented.nii.gz')
aseg = np.asarray(nib.load(asegfile).dataobj)

print('Fill the gaps')
if True:#not exists(labelsfile_filled):

    mask_dict = {'cr': join(outdir, 'upsampledMRIMaskCR.nii.gz'),
                 'crV': join(outdir, 'upsampledAseg.nii.gz'),
                 'cl': join(outdir, 'upsampledMRIMaskCLL.nii.gz'),
                 'bs': join(outdir, 'upsampledMRIMaskBS.nii.gz'),
                 }
    # Compute masks independently for each brain part
    for brainstruct, mrfile in mask_dict.items():

        if False:#exists(join(outdirtmp, brainstruct + '.filled.nii.gz')):
            continue

        proxy = nib.load(labelsfile)
        labels = np.asarray(proxy.dataobj)
        proxy = nib.load(mrfile)
        mask = np.asarray(proxy.dataobj)
        if brainstruct == 'crV':
            mask = np.logical_or(mask == 4, mask == 43)
            labels_missing_mask = labels[mask] == 0
            labels_missing = np.zeros(labels_missing_mask.shape, dtype='uint16')
            labels_missing[labels_missing_mask] = 383
            labels_filled = np.zeros_like(labels, dtype='uint16')
            labels_filled[mask] = labels_missing
            img = nib.Nifti1Image(labels_filled, proxy.affine)
            nib.save(img, join(outdirtmp, brainstruct + '.filled.nii.gz'))
            continue

        mask = mask > 0.5
        mask = binary_erosion(mask, np.ones((3,3,3)))
        mask_cr, crop_coord = image_utils.crop_label(mask, margin=10)

        labels_cr = image_utils.apply_crop(labels, crop_coord)
        labels_done = labels_cr[mask_cr]
        labels_missing = labels_cr[mask_cr] == 0

        num_voxels = np.sum(labels_missing)

        num_labels = 0
        label_dict = {}

        unique_labels = np.unique(labels_done)
        unique_labels = unique_labels.astype('int')
        unique_labels = list(filter(lambda x: x != 0, unique_labels))

        labels_onehot_filtered = np.zeros((num_voxels, len(unique_labels) + 1), dtype=np.int8)

        if brainstruct == 'cr':
            sigma = [1, 1, 4]

        elif brainstruct == 'cl':
            sigma = [6, 3, 3]

        else:
            sigma = [4, 10, 4]

        num_cores = 2
        results = Parallel(n_jobs=num_cores)(
            delayed(recon_utils.one_hot_encoding_gaussian)(labels_done, mask, labels_missing, ul, sigma)
            for ul in unique_labels)

        for it_ul, ul in enumerate(unique_labels):
            print(ul, end=' ', flush=True)
            if ul not in label_dict.keys():
                label_dict[ul] = num_labels
                it_label = num_labels
                num_labels += 1
            else:
                it_label = label_dict[ul]

        print('Saving filtered labels for ' + brainstruct + ' ...')
        labels_onehot_filtered = np.asarray(results).T
        labels_cat_mask = np.sum(labels_onehot_filtered, axis=-1) > 0
        labels_cat = np.argmax(labels_onehot_filtered, axis=-1)
        labels_cat = labels_cat.astype('uint16')
        del labels_onehot_filtered

        labels_cat_lut = np.zeros_like(labels_cat, dtype='uint16')
        for label, channel in label_dict.items(): labels_cat_lut[np.where(labels_cat == channel)] = int(label)
        labels_cat_lut[~labels_cat_mask] = 0
        labels_cat_gaps = np.zeros_like(labels_missing, dtype='uint16')
        labels_cat_gaps[labels_missing] = labels_cat_lut

        del labels_cat_lut, labels_cat

        lfb_labels_gaps = np.zeros(mask_cr.shape, dtype='uint16')
        lfb_labels_gaps[mask_cr] = labels_cat_gaps
        lfb_labels_gaps = lfb_labels_gaps.astype('uint16')

        del labels_cat_gaps

        labels_gaps_cr = np.zeros_like(labels_cr)
        labels_gaps_cr[labels_cr == 0] = lfb_labels_gaps[labels_cr == 0]


        labels_filled = np.zeros_like(labels, dtype='uint16')
        labels_filled[crop_coord[0][0]: crop_coord[0][1], crop_coord[1][0]: crop_coord[1][1], crop_coord[2][0]: crop_coord[2][1]] = labels_gaps_cr.astype('uint16')

        img = nib.Nifti1Image(labels_filled, proxy.affine)
        nib.save(img, join(outdirtmp, brainstruct + '.filled.nii.gz'))


    if True:#not exists(labelsfile_filled_tmp):
        # Group all masks into a single filled volume; here, probably a few voxels will remain still unlabeled.
        proxy_cr = nib.load(join(outdirtmp, 'cr.filled.nii.gz'))
        proxy_crV = nib.load(join(outdirtmp, 'crV.filled.nii.gz'))
        proxy_cl = nib.load(join(outdirtmp, 'cl.filled.nii.gz'))
        proxy_bs = nib.load(join(outdirtmp, 'bs.filled.nii.gz'))

        proxy = nib.load(labelsfile)
        labels = np.asarray(proxy.dataobj)

        mask_cr = np.array(proxy_cr.dataobj)
        labels[mask_cr > 0] = mask_cr[mask_cr > 0]
        del mask_cr

        mask_crV = np.array(proxy_crV.dataobj)
        labels[mask_crV > 0] = mask_crV[mask_crV > 0]
        del mask_crV

        mask_cl = np.array(proxy_cl.dataobj)
        labels[mask_cl > 0] = mask_cl[mask_cl > 0]
        del mask_cl

        mask_bs = np.array(proxy_bs.dataobj)
        labels[mask_bs > 0] = mask_bs[mask_bs > 0]
        del mask_bs

        img = nib.Nifti1Image(labels.astype('uint16'), proxy.affine)
        nib.save(img, labelsfile_filled_tmp)

    else:

        proxy = nib.load(labelsfile_filled_tmp)
        labels = np.array(proxy.dataobj)

    proxy_brain = nib.load(join(outdir, 'upsampledMRIMask.nii.gz'))
    mask_brain = np.array(proxy_brain.dataobj)
    mask_missing = labels == 0

    idx = distance_transform_edt(mask_missing, return_distances=False, return_indices=True)

    mask_missing[mask_brain == 0] = 0
    idx_missing = idx[:, mask_missing]

    labels_missing = np.zeros((int(np.sum(mask_missing)), ), dtype='uint16')

    for it_v in range(int(np.sum(mask_missing))):
        vox = idx_missing[..., it_v]
        labels_missing[it_v] = labels[vox[0], vox[1], vox[2]]

    labels[mask_missing] = labels_missing

    img = nib.Nifti1Image(labels.astype('uint16'), proxy.affine)
    nib.save(img, labelsfile_filled)

output_dir = join(DATA_DIR, subject)
if not exists(output_dir): os.makedirs(output_dir)
subprocess.call(['ln', '-s', labelsfile, join(output_dir, str(target_res) + 'mm.nii.gz')])
subprocess.call(['ln', '-s', labelsfile_filled, join(output_dir, str(target_res) + 'mm.filled.nii.gz')])

