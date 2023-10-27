# imports
import pdb
from os.path import join, exists
from os import makedirs
import time
from argparse import ArgumentParser

# third party imports
import numpy as np
import torch
import nibabel as nib

# project imports
from config import config_database
from dataset.data_loader import DataLoader
from config import config_dev, config_donors
from src.algorithm_helpers import initialize_graph_RegNet, initialize_graph_SbR, get_dataset, get_model, \
    initialize_algorithm_masks, get_weightsfile

# ******************************************************************************************************************** #
#
# This file initialises the observational graph for a given subject (--subject) by computing intermodal and intramodal
# (up to -nn neighbours). Optionally you could run all blocks or specify a list of blocks in --bid.
#
# ******************************************************************************************************************** #


print('')
print('--------------------------')
print('---- Initialize Graph ----')
print('--------------------------')
print('')

####################
# Input parameters #
####################
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--model', default='bidir', choices=['standard', 'bidir'])
arg_parser.add_argument('--subject', default='P57-16', choices=list(config_donors.file_dict.keys()))
arg_parser.add_argument('--nn', default=4, type=int)
arg_parser.add_argument('--bid', default=None, nargs='+')
arg_parser.add_argument('--force', action='store_true')
arg_parser.add_argument('--modality', default=['MRI_LFB', 'MRI_HE', 'LFB_HE', 'MRI', 'LFB', 'HE'], nargs='+',
                        help='used to contrain the graph initialization to given registration modalities')

arguments = arg_parser.parse_args()
model_type = arguments.model
N_NEIGHBOURS = arguments.nn
SUBJECT = arguments.subject
initial_bid_list = arguments.bid
modality = arguments.modality
force_flag = arguments.force

# Parameters and directories
BT_DB_blocks = config_database.get_lin_dict(SUBJECT)
BT_DB = config_database.get_nonlin_dict(SUBJECT)
data_dict = {SUBJECT: BT_DB}

config_file_data = config_donors.file_dict[SUBJECT]
if initial_bid_list is None: initial_bid_list = config_file_data.initial_bid_list
initial_bid_list = {SUBJECT: initial_bid_list}
print(initial_bid_list)

CONFIG_DICT = config_dev.CONFIG_DICT
parameter_dict_MRI = CONFIG_DICT['MRI']

use_gpu = False#parameter_dict_MRI['USE_GPU'] and torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

kwargs_generator = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}

results_dir = join(config_file_data.BASE_DIR, 'NonlinearRegistration', 'Registration')
if not exists(results_dir): makedirs(results_dir)
tempdir = join(results_dir, 'tmp')
if not exists(tempdir): makedirs(tempdir)

###################
# Tree parameters #
###################
prefix_list = ['MRI_LFB', 'MRI_HE', 'LFB_HE']
num_neighbours_list = [0]*3
for it_n in range(N_NEIGHBOURS):
    prefix_list.extend(['MRI', 'LFB', 'HE'])
    num_neighbours_list.extend([it_n+1]*3)

####################
# Run registration #
####################

print('[SUBJECT ' + str(SUBJECT) + '] Processing.')
not_processed_files = []
for it_network in range(3 * (1 + N_NEIGHBOURS)):

    stain = prefix_list[it_network]
    num_neighbours = num_neighbours_list[it_network]
    parameter_dict = CONFIG_DICT[stain]


    if stain not in modality:
        continue

    data_loader = DataLoader(data_dict, included_blocks=initial_bid_list)
    subject_loader = data_loader.subject_dict[SUBJECT]
    nblocks = len(data_loader)

    print('Registering: ' + stain + ' with ' + str(num_neighbours)  + ' neighbours.')

    for it_block, block in enumerate(subject_loader):

        if SUBJECT + block.id == 'P41-16P4.1' and stain in ['MRI_LFB', 'MRI_HE']:
            parameter_dict['RESULTS_DIR'] = '/home/acasamitjana/Results/Registration/BUNGEE_Tools/Registration/NCC9_RegNet/R1_S0.1/DownFactor_4/' + stain
        else:
            parameter_dict['RESULTS_DIR'] = '/home/acasamitjana/Results/Registration/BUNGEE_Tools/Registration/NCC9_RegNet/R1_S1/DownFactor_4/' + stain

        print('    - Block: ' + block.id, end='. ', flush=True)

        # Block parameters
        nslices = len(block)
        results_dir_sbj = join(results_dir, block.id)
        if not exists(results_dir_sbj):
            makedirs(results_dir_sbj)

        filename = stain + '.' + str(num_neighbours) + 'N'
        if not force_flag or exists(join(results_dir_sbj, filename + '.velocity_field.nii.gz')) and arguments.bid is None:
            continue

        # Load dataset
        if stain in ['MRI_LFB', 'MRI_HE', 'LFB_HE']:
            dataset, sampler = get_dataset(block=block, parameter_dict=parameter_dict, registration_type='intermodal',
                                           image_shape=data_loader.image_shape, mdil=False)

        elif stain in ['MRI', 'LFB', 'HE']:
            dataset, sampler = get_dataset(block=block, parameter_dict=parameter_dict, registration_type='intramodal',
                                           image_shape=data_loader.image_shape, num_neighbours=num_neighbours,
                                           fix_neighbors=True, mdil=False)


        else:
            raise ValueError("Please, specify a valid stain [MRI, LFB, HE, and combinations]")

        input_shape = dataset.image_shape
        generator = torch.utils.data.DataLoader(
            dataset,
            batch_size=parameter_dict['BATCH_SIZE'],
            shuffle=False,
            sampler=sampler,
            **kwargs_generator
        )

        # Update affine (due to padding/cropping)
        affine = block.vox2ras0
        diff_shape = [input_shape[it_d] - block.image_shape[it_d] for it_d in range(2)]
        if sum(np.abs(diff_shape)) > 0:
            tx = np.eye(4)
            tx[0, -1] = -diff_shape[0] // 2
            tx[1, -1] = -diff_shape[1] // 2
            affine = affine @ tx

        if SUBJECT == 'P57.16' and (block.id == 'P8.1' or block.id == 'C5.1'):
            flip_z = np.eye(4)
            flip_z[2, 2] = -1
            flip_z[2, 3] = nslices - 1
            affine = np.dot(affine, flip_z)

        if force_flag or (stain in ['MRI', 'LFB', 'HE'] and num_neighbours == 1 and not exists(join(results_dir_sbj, stain + '.mask.nii.gz'))):
            dataset_masks, sampler_masks = get_dataset(block=block, parameter_dict=parameter_dict, num_neighbours=0,
                                                       fix_neighbors=0, registration_type='intramodal',
                                                       image_shape=data_loader.image_shape, mdil=False,
                                                       )

            ref_masks = initialize_algorithm_masks(dataset_masks, sampler_masks)
            img = nib.Nifti1Image(ref_masks, affine)
            nib.save(img, join(results_dir_sbj, stain + '.mask.nii.gz'))

        ################
        # Registration #
        ################
        t_init = time.time()

        if stain in ['MRI_LFB', 'MRI_HE', 'LFB_HE'] and SUBJECT + block.id not in ['P41-16_P4.1']:
            weightsfile = get_weightsfile(parameter_dict['RESULTS_DIR'] + '_SbR_' + model_type, SUBJECT, block.id)
            model = get_model('sbr', input_shape, parameter_dict, device, weightsfile=weightsfile)
            output_results = initialize_graph_SbR(model, generator, parameter_dict, device)

        elif stain in ['MRI', 'LFB', 'HE']:
            weightsfile = get_weightsfile(parameter_dict['RESULTS_DIR'] + '_' + model_type, SUBJECT, block.id)
            #weightsfile = join(parameter_dict['RESULTS_DIR'] + '_' + model_type, 'checkpoints', 'model_checkpoint.FI.pth')
            model = get_model('regnet', input_shape, parameter_dict, device, weightsfile=weightsfile)
            output_results = initialize_graph_RegNet(model, generator, parameter_dict, device)

        else:
            weightsfile = get_weightsfile(parameter_dict['RESULTS_DIR'] + '_' + model_type, SUBJECT, block.id)
            #weightsfile = join(parameter_dict['RESULTS_DIR'] + '_' + model_type, 'checkpoints', 'model_checkpoint.FI.pth')
            model = get_model('regnet', input_shape, parameter_dict, device, weightsfile=weightsfile)
            output_results = initialize_graph_RegNet(model, generator, parameter_dict, device)
            # raise ValueError("Please, specify a valid stain [MRI, LFB, HE, and combinations]")

        registered_image, registered_mask, velocity_field, displacement_field = output_results

        print('Networks elapsed time: ' + str(np.round(time.time() - t_init, 2)), end='', flush=True)

        # Save output forward tree
        img = nib.Nifti1Image(output_results[0], affine)
        nib.save(img, join(results_dir_sbj, filename + '.nii.gz'))

        img = nib.Nifti1Image(output_results[1], affine)
        nib.save(img, join(results_dir_sbj, filename + '.mask.nii.gz'))

        img = nib.Nifti1Image(output_results[2], affine)
        nib.save(img, join(results_dir_sbj, filename + '.velocity_field.nii.gz'))
        print('')

print('Not processed files: ', end='', flush=True)
print(not_processed_files)

print('[SUBJECT '+ str(SUBJECT) + '] Graph initialized DONE.')
print('')

