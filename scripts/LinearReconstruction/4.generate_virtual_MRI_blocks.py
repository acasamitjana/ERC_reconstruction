# py
from os.path import join
from argparse import ArgumentParser
import subprocess
import copy

# libraries imports
import nibabel as nib
import numpy as np
import torch
from scipy import io
from skimage.morphology import binary_closing
import cv2

# project imports
from utils.io_utils import create_results_dir
from src import datasets
from dataset.data_loader_linear import DataLoader, DataLoaderDownsample
from config import config_donors, config_database

#  *********************************  #
#
# This script creates virtual MRI blocks in order to be utilised for registration. For each block, a volume of N slices
# (same number of histology section) is created at the block resolution
#
#  *********************************  #


####################################
############ PARAMETERS ############
####################################

# Create a data loader
print('-----------------------------------')
print('--- Generate virtual-MRI blocks ---')
print('-----------------------------------')

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='P57-16', choices=['P57-16',  'P41-16', 'P58-16', 'P85-18', 'EX9-19'])
arg_parser.add_argument('--bid',  default=None, nargs='+')
arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
BID_LIST = arguments.bid

BT_DB = config_database.get_lin_dict(SUBJECT)
BT_DB_full = copy.copy(BT_DB)
config_file_data = config_donors.file_dict[SUBJECT]
BASE_DIR = join(config_file_data.BASE_DIR, 'RigidRegistration')
RESULTS_DIR = join(BASE_DIR, 'results')
MASKS_DIR = join(RESULTS_DIR, 'masks')
TMP_DIR = join(RESULTS_DIR, 'tmp')
REG_BLOCKS_DIR = join(RESULTS_DIR, 'reg_blocks')
create_results_dir(RESULTS_DIR, subdirs=['masks', 'reg_blocks'])

torch_dtype = torch.float
device = torch.device("cuda:0")

data_loader = DataLoaderDownsample(BT_DB)
data_loader_full = DataLoader(BT_DB_full)

Nblocks = len(data_loader_full)
Nslides_per_block = {}

vol_shape = data_loader.vol_shape
vol_affine = data_loader.MRI_affine
subject_dict = data_loader.subject_dict
slice_id_list = data_loader.slice_id_list

dataset = datasets.BlockRegistrationBTDataset(data_loader)

print('Saving virtual blocks ...')
print('   ', end=' ')
for it_sbj, bid in enumerate(data_loader.subject_dict.keys()):
    print(bid + ',', end=' ', flush=True)

    if BID_LIST is not None:
        if bid not in BID_LIST: continue

    TARGET_RES_XY = 0.1
    TARGET_RES_Z = 1

    input_block_file = join(REG_BLOCKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.LFB.reg.mgz')
    input_mask_file = join(MASKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.mri.mask.mgz')
    block_mosaic = join(RESULTS_DIR, 'BLOCK_MASK_MOSAIC.mgz')
    tmp_block_mosaic = join(RESULTS_DIR, 'BLOCK_MASK_MOSAIC_RES.mgz')

    output_block_file_upsampled = join(REG_BLOCKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.LFB.reg.upsampled.mgz')
    output_mri_file = join(REG_BLOCKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.mri.reg.mgz')
    output_aparc_file = join(REG_BLOCKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.aparc.reg.mgz')
    output_mask_file = join(REG_BLOCKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.mri.mask.reg.mgz')
    output_mask_file_closing = join(REG_BLOCKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.mri.mask.reg.closing.mgz')
    proxy_block = nib.load(input_block_file)
    shape_block = proxy_block.shape
    data_block = np.asarray(proxy_block.dataobj)
    data_block = (255 * data_block).astype('uint8')
    vox2ras0 = copy.copy(proxy_block.affine)
    block_res = np.linalg.norm(vox2ras0,2,0)

    mapping = io.loadmat(join(BT_DB['SLIDES_DIR'], BT_DB['SUBJECT'] + '_' + bid, 'LFB',
                              BT_DB['SUBJECT'] + '_' + bid + '_LFB_mapping.mat'))['mapping'][0]

    # upsample LFB block to 0.1 mm in plane
    inplane_factor = [b/TARGET_RES_XY for b in block_res[:2]]
    inplane_shape = tuple([int(np.round(a*b)) for a, b in zip(shape_block[:2], inplane_factor)])
    inplane_factor = [a/b for a, b in zip(inplane_shape, shape_block[:2])]
    nz = shape_block[2]
    mri_vol = np.zeros(inplane_shape + (nz,), dtype='uint8')

    for j in range(nz):
        z = TARGET_RES_Z * (j - 1) + 1
        mri_vol[..., z] = cv2.resize(data_block[..., z], (inplane_shape[1], inplane_shape[0]), interpolation=cv2.INTER_LINEAR)

    aux = np.asarray([(ipf-1) / (2*ipf) for ipf in inplane_factor] + [0])
    affine = vox2ras0
    affine[:3,0] = affine[:3,0] / inplane_factor[0]
    affine[:3,1] = affine[:3,1] / inplane_factor[1]
    affine[:3,2] = affine[:3,2] / TARGET_RES_Z
    affine[:3,3] = affine[:3,3] - np.dot(vox2ras0[:3,:3], aux.T)

    img = nib.Nifti1Image((mri_vol / 255).astype(np.float), affine)
    nib.save(img, output_block_file_upsampled)

    # reslice-like mri
    subprocess.call([
        'mri_convert', data_loader_full.MRI_file, output_mri_file, '-odt', 'float', '-rl', output_block_file_upsampled
    ], stdout=subprocess.DEVNULL)

    # reslice-like seg
    subprocess.call([
        'mri_convert', data_loader_full.MRI_aparc_file, output_aparc_file, '-odt', 'float', '-rt', 'nearest', '-rl', output_block_file_upsampled
    ], stdout=subprocess.DEVNULL)

    # reslice-like mask
    subprocess.call([
        'mri_convert', input_mask_file, output_mask_file, '-odt', 'float', '-rt', 'nearest', '-rl',
        output_block_file_upsampled
    ], stdout=subprocess.DEVNULL)

    # reslice-like block mosaic to constraint the closing (e.g., useful to avoid closing gyrus)
    subprocess.call([
        'mri_convert', block_mosaic, tmp_block_mosaic, '-odt', 'float', '-rt', 'nearest', '-rl',
        output_block_file_upsampled
    ], stdout=subprocess.DEVNULL)

    # closing of size 13,13 in plane (i.e. 1.3 mm)
    proxy = nib.load(output_mask_file)
    data = np.asarray(proxy.dataobj) > 0.5
    se_xy = np.ones((13, 13))
    for it_z in range(data.shape[2]):
        data[..., it_z] = binary_closing(data[..., it_z], se_xy)

    proxy = nib.load(tmp_block_mosaic)
    data_tmp = np.asarray(proxy.dataobj) > 0

    new_data = np.zeros_like(data)
    new_data[data_tmp] = data[data_tmp]
    img = nib.Nifti1Image(new_data.astype(np.int), proxy.affine)
    nib.save(img, join(REG_BLOCKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.mri.mask.reg.closing.mgz'))

print('Done.\n')




