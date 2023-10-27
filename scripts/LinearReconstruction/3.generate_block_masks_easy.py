# py
from os.path import join, exists
from os import listdir
import copy
from argparse import ArgumentParser
import pdb

# libraries imports
import nibabel as nib
import numpy as np


# project imports
from dataset.data_loader_linear import DataLoader
from config import config_donors, config_database
from src import algorithm_helpers
from utils import io_utils
from utils.image_utils import align_with_identity_vox2ras0

#  *********************************  #
#
# This script creates block masks using a distance transform to fill the MRI masks and creates a volume to relate MR
# coordinates to histology blocks (i.e., the indices mask)
#
#  *********************************  #


print('--------------------------------')
print('--- Generate MRI-Block Masks ---')
print('--------------------------------')
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='P57-16', choices=['P57-16',  'P41-16', 'P58-16', 'P85-18', 'EX9-19'])
arg_parser.add_argument('--bid',  default=None, nargs='+')
arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
bid_list = arguments.bid

BT_DB = config_database.get_lin_dict(SUBJECT)
config_file_data = config_donors.file_dict[SUBJECT]
BASE_DIR = join(config_file_data.BASE_DIR, 'RigidRegistration')
RESULTS_DIR = join(BASE_DIR, 'results')
TMP_DIR = join(RESULTS_DIR, 'tmp')
MASKS_DIR = join(RESULTS_DIR, 'masks')

io_utils.create_results_dir(RESULTS_DIR, subdirs=['tmp', 'mri'])

BLOCK_FILES = listdir(MASKS_DIR)
BLOCK_FILES = list(filter(lambda x: '_volume.LFB.mask.reg.shift.mgz' in x, BLOCK_FILES))

data_loader_full = DataLoader(BT_DB)
MRI_MASK = data_loader_full.load_MRI_mask() > 0.5
MRI_AFFINE = data_loader_full.MRI_affine

# if not exists(join(RESULTS_DIR, 'BLOCK_MASK_MOSAIC_INNER.mgz')):
#     blockID, blocks_mask = generate_inner_masks()
#     BLOCK_MASK_MOSAIC_INNER = np.zeros_like(MRI_MASK, dtype=np.int)
#     BLOCK_MASK_MOSAIC_INNER[MRI_MASK] = blockID
#     img = nib.Nifti1Image(BLOCK_MASK_MOSAIC_INNER, MRI_AFFINE)
#     nib.save(img, join(RESULTS_DIR, 'BLOCK_MASK_MOSAIC_INNER.mgz'))
if not exists(join(RESULTS_DIR, 'BLOCK_MASK_MOSAIC.rasoriented.update.nii.gz')):
    print('Generate block indices ...')
    blockID, blocks_mask = algorithm_helpers.generate_MRI_masks(data_loader_full, BLOCK_FILES, MASKS_DIR)
    BLOCK_MASK_MOSAIC = np.zeros_like(MRI_MASK, dtype=np.int)
    BLOCK_MASK_MOSAIC[MRI_MASK] = blockID
    img = nib.Nifti1Image(BLOCK_MASK_MOSAIC, MRI_AFFINE)
    nib.save(img, join(RESULTS_DIR, 'BLOCK_MASK_MOSAIC.mgz'))

    v, vr = align_with_identity_vox2ras0(BLOCK_MASK_MOSAIC.astype('uint16'), MRI_AFFINE)
    img = nib.Nifti1Image(v, vr)
    nib.save(img, join(RESULTS_DIR, 'BLOCK_MASK_MOSAIC.rasoriented.update.nii.gz'))

else:
    proxy = nib.load(join(RESULTS_DIR, 'BLOCK_MASK_MOSAIC.rasoriented.update.nii.gz'))
    BLOCK_MASK_MOSAIC = np.asarray(proxy.dataobj)

# Save MRI virtual masks (each block separately) in results
print('Saving virtual masks ...')
print(bid_list)
for it_block_filepath, block_filepath in enumerate(BLOCK_FILES):
    bid = block_filepath.split('_')[1]
    if bid_list is None or bid in bid_list:
        print(bid)
        proxy = nib.load(join(MASKS_DIR,block_filepath))
        affine = MRI_AFFINE

        mask = (BLOCK_MASK_MOSAIC == it_block_filepath + 1).astype(np.int)

        img = nib.Nifti1Image(mask.astype(np.float), affine)
        nib.save(img, join(MASKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.mri.mask.mgz'))

print('Done.\n')
