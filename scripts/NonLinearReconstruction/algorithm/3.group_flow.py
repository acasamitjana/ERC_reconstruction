# py
import pdb

import os
from os.path import join, exists
import subprocess
from argparse import ArgumentParser

# libraries imports
import nibabel as nib
import numpy as np

# project imports
from config import config_database, config_donors, config_dev
from dataset.data_loader import DataLoader
from utils.deformation_utils import interpolate2D
from src import algorithm_helpers

# ******************************************************************************************************************** #
#
# This script is used to group _reg_aladin_ registration field (affine) and any nonlinear field:
#    - Registraion baseline: RegNet, NR
#    - Algorithm: SPX-LX
# Then we could use the total flow to deform RGB images, labels at different resolutions.
#
# ******************************************************************************************************************** #

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='change_orientation', choices=list(config_donors.file_dict.keys()))
arg_parser.add_argument('--reg_algorithm', default='ST3_L1_RegNet_NN2', choices=['Linear', 'SbR', 'RegNet',
                                                                                 'ST3_L1_RegNet_NN4',
                                                                                 'ST3_L1_RegNet_NN2'])
arg_parser.add_argument('--bid', default=None, nargs='+')

arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
reg_algorithm = arguments.reg_algorithm
initial_bid_list = arguments.bid

NiftyRegPath = '/home/acasamitjana/Software_MI/niftyreg-git/build/'
TRANSFORMcmd = NiftyRegPath + 'reg-apps' + '/reg_transform'

BT_DB_blocks = config_database.get_lin_dict(SUBJECT)
BT_DB_slices = config_database.get_nonlin_dict(SUBJECT)
data_dict = {SUBJECT: BT_DB_slices}
SLICES_DIR = join(BT_DB_blocks['RESULTS_DIR'], 'results', 'slices')
config_file_data = config_donors.file_dict[SUBJECT]

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
    BLOCK_DIR = join(SLICES_DIR, bid)
    AFFINE = block_loader.vox2ras0
    nslices = np.max([int(sl.sid) for sl in block_loader])

    # try:
    for stain in data_path_dict[reg_algorithm]['stain']:
        print(' ' + stain)

        results_dir_sbj = join(data_path_dict[reg_algorithm]['path'], bid)
        # if exists(join(results_dir_sbj, stain + '.totalflow.nii.gz')): continue

        proxy_flow = nib.load(join(results_dir_sbj, stain + '.flow.nii.gz'))
        flow = np.asarray(proxy_flow.dataobj)

        field = np.zeros(flow.shape[:-1] + (nslices, ))
        print('   Slice number:', end=' ', flush=True)
        for it_slice, slice in enumerate(block_loader):
            sid = int(slice.sid)
            print('#' + str(sid), end=' ', flush=True)
            slice_num = sid - 1

            #
            refFile = join(BLOCK_DIR, 'MRI', 'images', 'slice_' + "{:04d}".format(sid) + '.png')
            affineFile = join(BLOCK_DIR, stain, 'affine', 'slice_' + "{:04d}".format(sid) + '.aff')
            dummyFileNifti = '/tmp/' + SUBJECT + slice.sid + 'images_tmp_groupflow.nii.gz'

            subprocess.call([TRANSFORMcmd, '-ref', refFile, '-disp', affineFile, dummyFileNifti], stdout=subprocess.DEVNULL)

            proxy_aff = nib.load(dummyFileNifti)
            affine_field = np.asarray(proxy_aff.dataobj)[:, :, 0, 0, :]
            affine_field = np.transpose(affine_field, axes=[2, 1, 0])

            #
            II, JJ = np.meshgrid(np.arange(0, flow.shape[1]), np.arange(0, flow.shape[2]), indexing='ij')
            II2 = II + flow[0,..., it_slice]
            JJ2 = JJ + flow[1,..., it_slice]
            inci = interpolate2D(affine_field[1], np.stack((II2, JJ2), axis=-1))
            incj = interpolate2D(affine_field[0], np.stack((II2, JJ2), axis=-1))
            JJ3 = JJ2 + incj
            II3 = II2 + inci
            field_j = JJ3 - JJ
            field_i = II3 - II

            field[0,..., sid-1] = field_i
            field[1,..., sid-1] = field_j

        img = nib.Nifti1Image(field, AFFINE)
        nib.save(img, join(results_dir_sbj, stain + '.totalflow.nii.gz'))

