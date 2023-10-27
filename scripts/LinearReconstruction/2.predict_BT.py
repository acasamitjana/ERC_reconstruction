# py
from os.path import join
import numpy as np
import copy
import subprocess
from argparse import ArgumentParser

# libraries imports
import torch
import nibabel as nib

# project imports
from dataset.data_loader_linear import DataLoader, DataLoaderDownsample
from config import config_donors, config_database
from utils.io_utils import create_results_dir
from src import datasets, models, layers, test



#  *********************************  #
#
# This script is used to update image headers (and possibly predict intermediate steps blocks for debugging) by the
# trained model specified by `subject'. It requires:
# It requires the following data from the MATLAB pipeline:
#    - A trained model in databaseConfig.BT_LINEAR[subject]['RESULTS_DIR']
#    - Manually update training_params to match those from training
# It outputs the following:
#    - Update headers at the original resolution (training was at lower resolution, prediction at original)
#    - The resulting histo blocks with the updated headers.
#
#  *********************************  #

print('-----------------------------------')
print('--- Predict Linear Registration ---')
print('-----------------------------------')

##################
# I/O parameters #
##################

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='P57-16', choices=['P57-16', 'P41-16', 'P58-16', 'P85-18', 'EX9-19'])
arg_parser.add_argument('--bid', nargs='+')
arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
BID_LIST = arguments.bid

BT_DB = config_database.get_lin_dict(SUBJECT)

config_file_data = config_donors.file_dict[SUBJECT]
BASE_DIR = join(config_file_data.BASE_DIR, 'RigidRegistration')
RESULTS_DIR = join(BASE_DIR, 'results')
MASKS_DIR = join(RESULTS_DIR, 'masks')
REG_BLOCKS_DIR = join(RESULTS_DIR, 'reg_blocks')
create_results_dir(RESULTS_DIR, subdirs=['masks', 'reg_blocks'])

kwargs_training = {'log_interval': 1}  # Number of steps
device = torch.device("cuda:0")

# Training
torch_dtype = torch.float
training_params = {
    'n_iterations': 9,
    'schedule': [0, 1, 0, 3,   0,   4,   0,   4,   5],
    'cp_spacing': 20
}

###############
# Data Loader #
###############
data_loader = DataLoaderDownsample(BT_DB)
data_loader_full = DataLoader(BT_DB)

Nblocks = len(data_loader)
Nslides_per_block = {}

vol_shape = data_loader.vol_shape
vol_affine = data_loader.MRI_affine
subject_dict = data_loader.subject_dict
slice_id_list = data_loader.slice_id_list

dataset = datasets.BlockRegistrationBTDataset(data_loader)

header_orig = {}
for it_bid, bid in enumerate(dataset.data_loader.subject_dict.keys()):
    _, _, header_orig[bid] = dataset[bid]

##############
# Prediction #
##############
epoch_weights = 'FI'
weightsfile = 'model_checkpoint.FI.pth'
results_filepath_list = []
for it_opt_level in range(training_params['n_iterations']):
    results_dir_schedule = join(BASE_DIR, str(it_opt_level+1))
    results_filepath = join(results_dir_schedule, 'results', 'training_results.csv')
    results_filepath_list.append(results_filepath)
    # visualization.plot_results(results_filepath, keys=['loss', 'loss_intensities', 'loss_overlap', 'loss_scale'])

print('Performance plots ...')
# visualization.plot_multiple_results(results_filepath_list,
#                                     keys=['loss', 'loss_intensities', 'loss_overlap', 'loss_scale'])

for it_opt_level in range(training_params['n_iterations']):

    results_dir_schedule = join(BASE_DIR, str(it_opt_level+1))
    checkpoint = torch.load(join(results_dir_schedule, 'checkpoints', weightsfile))

    print('\n')
    print('####### ITERATION NUMBER ' + str(it_opt_level + 1) +
          '. Model ' + str(training_params['schedule'][it_opt_level]) + '. #######')
    print('\n')

    # Models
    init_model_dict = {}
    init_optimizer_dict = {}
    warper = layers.SpatialTransformerAffine(vol_shape, padding_mode = 'zeros', torch_dtype=torch_dtype)
    model_dict = {'warper': warper.to(device)}
    optimizer_dict = {}

    if training_params['schedule'][
        it_opt_level] == 0:  # Global scaling. Parameters per structure: Angle (3), translation (3),
        shared_scaling = torch.nn.Parameter(torch.zeros(2, 1))
        for it_model in range(3):
            model = models.Instance3DGlobalScalingModel(vol_shape, vol_affine, shared_scaling, device=device)
            model = model.to(device)
            init_model_dict[it_model] = model

        for bid, block in subject_dict.items():
            if block.structure_id == 'Cerebrum':
                init_model_dict[0].normvecs.append(block.get_normvec())
                init_model_dict[0].crs.append(block.get_cr())
                model_dict[bid] = init_model_dict[0]

            elif block.structure_id == 'Cerebellum':
                init_model_dict[1].normvecs.append(block.get_normvec())
                init_model_dict[1].crs.append(block.get_cr())
                model_dict[bid] = init_model_dict[1]

            elif block.structure_id == 'Brainstem':
                init_model_dict[2].normvecs.append(block.get_normvec())
                init_model_dict[2].crs.append(block.get_cr())
                model_dict[bid] = init_model_dict[2]

            else:
                raise ValueError('No structure named ' + block.structure_id)

    elif training_params['schedule'][
        it_opt_level] == 1:  # Global scaling. Parameters per slice: angle (1, in-plane), translation (3),
        shared_scaling = torch.nn.Parameter(torch.zeros(2, 1))
        for it_model in slice_id_list:
            if it_model not in init_model_dict.keys():
                model = models.InstanceGlobalScalingModel(vol_shape, vol_affine, shared_scaling, device=device)
                init_model_dict[it_model] = model.to(device)

        for bid, block in subject_dict.items():
            init_model_dict[block.slice_id].normvecs.append(block.get_normvec())
            init_model_dict[block.slice_id].crs.append(block.get_cr())
            model_dict[bid] = init_model_dict[block.slice_id]


    elif training_params['schedule'][
        it_opt_level] == 2:  # Global scaling. Parameters per block: angle (1, in-plane), translation (3).
        shared_scaling = torch.nn.Parameter(torch.zeros(2, 1))
        for bid, block in subject_dict.items():
            model = models.InstanceGlobalScalingModel(vol_shape, vol_affine, shared_scaling, device=device)
            model = model.to(device)
            model.normvecs.append(block.get_normvec())
            model.crs.append(block.get_cr())
            model_dict[bid] = model


    elif training_params['schedule'][
        it_opt_level] == 3:  # Parameters per block: Angle (1, in-plane), translation (3), scaling (2)
        for bid, block in subject_dict.items():
            model = models.InstanceIndividualScalesModel(vol_shape, vol_affine, device=device)
            model = model.to(device)
            model.normvecs.append(block.get_normvec())
            model.crs.append(block.get_cr())
            model_dict[bid] = model

    elif training_params['schedule'][
        it_opt_level] == 4:  # Parameters per block: Angle (3), translation (3), scaling (2)
        for bid, block in subject_dict.items():
            model = models.InstanceIndividualSimilarityModel(vol_shape, vol_affine, device=device)
            model.normvecs.append(block.get_normvec())
            model.crs.append(block.get_cr())
            model = model.to(device)
            model_dict[bid] = model

    # Eugenio: added mode 5, which is the same as 4 but with nonlinear transforms for every slice
    elif training_params['schedule'][
        it_opt_level] == 5:  # Parameters per block: Angle (3), translation (3), scaling (2) + nonlinear per slice
        for bid, block in subject_dict.items():
            block_shape = block.vol_shape

            model = models.InstanceIndividualSimilarityModelWithNonlinear(vol_shape, vol_affine, block_shape,
                                                                          cp_spacing=training_params['cp_spacing'],
                                                                          device=device,
                                                                          torch_dtype=torch_dtype)
            model.normvecs.append(block.get_normvec())
            model.crs.append(block.get_cr())
            model = model.to(device)
            model_dict[bid] = model

    else:
        raise RuntimeError('Mode must be between 0 and 5')

    for model_key, model in model_dict.items():
        if model_key == 'S':
            continue

        model.load_state_dict(checkpoint['state_dict_' + model_key])

    Testing = test.LinearTest(dataset, device)
    # predict(model_dict, iteration=it_opt_level+1, **kwargs_training)
    if training_params['schedule'][it_opt_level] == 5:
        Testing.update_headers_images(model_dict)
    else:
        Testing.update_headers(model_dict)

# image_utils.filter_3d(dataset, kernel_sigma=[5,5,5])
print('Saving registered images and masks ...')
print('   ', end=' ')
for it_sbj, bid in enumerate(data_loader.subject_dict.keys()):

    if BID_LIST is not None:
        if bid not in BID_LIST: continue

    print(bid + ',', end=' ', flush=True)

    block_file = join(REG_BLOCKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.LFB.reg.mgz')
    block_mask = join(MASKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.LFB.mask.reg.downsampled.mgz')
    block_mask_upsampled = join(MASKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.LFB.mask.reg.mgz')
    block_mask_mriconvert = join(MASKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.LFB.mask.reg.mriconvert.mgz')
    block_mask_shift = join(MASKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.LFB.mask.reg.shift.mgz')

    sbj = data_loader.subject_dict[bid]
    block_shape = sbj.vol_shape

    # header = copy.copy(dataset.data_loader.subject_dict[bid]._affine)
    image_data = dataset.images_dict[bid]
    mask_data = dataset.mask_dict[bid]

    proxy = nib.load(block_file)
    # image_data = np.array(proxy.dataobj)
    header = proxy.affine
    # proxy = nib.load(block_mask)
    # mask_data = (image_data>0.1).astype('uint8')#np.array(proxy.dataobj)

    mri_affine = data_loader_full.MRI_affine
    mri_res = np.linalg.norm(mri_affine, axis=0)[:3]
    block_res = np.linalg.norm(header, axis=0)[:3]
    resize_factor = [m / b for m, b in zip(mri_res, block_res)]


    img = nib.Nifti1Image(image_data, header)
    nib.save(img, block_file)

    img = nib.Nifti1Image(mask_data, header)
    nib.save(img, block_mask)

    vs = [b*r for b,r in zip(block_res, resize_factor)]
    subprocess.call([
        'mri_convert', block_mask, block_mask_mriconvert, '-vs', str(vs[0]), str(vs[1]), str(vs[2])
    ], stdout=subprocess.DEVNULL)

    # # Shift
    proxy = nib.load(block_mask_mriconvert)
    proxy_affine = proxy.affine
    aux = np.asarray([0, 0, -(resize_factor[-1] - 1) / (2 * resize_factor[2])])
    proxy_affine[:3, 3] = proxy_affine[:3, 3] + np.dot(header[:3,:3], aux.T)
    data = np.asarray(proxy.dataobj)
    img = nib.Nifti1Image(data, proxy_affine)
    nib.save(img, block_mask_shift)


print('Done.\n')

