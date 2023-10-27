from os.path import join, exists
from os import makedirs
import time
import pdb
from argparse import ArgumentParser

import numpy as np
import torch
import nibabel as nib

from src.training import LinearTraining
from utils.io_utils import create_results_dir, ExperimentWriter
from utils.deformation_utils import MakeGrayMosaicWithoutNormalization
from src.callbacks import History, ModelCheckpoint, PrinterCallback, ToCSVCallback, LRDecay
from src import losses, models, datasets, layers
from dataset.data_loader_linear import DataLoaderDownsample
from config import config_donors, config_database


#  *********************************  #
#
# This script is used to linearly align all histology blocks to the ex-vivo MRI volume. It uses a hierarchical model
# similar to the one used by Mancini et al., (Scientific Reports, 2020)
# It requires the following data from the MATLAB pipeline:
#    - All the histology blocks (createHistoBlocks.m).
#    - The downsampled version of MRI and histo blocks (typically at 1mm, also from MATLAB code)
#    - MRI masks separately for each brain part (Cerebrum, cerebellum and brainsetm)
# It outputs the following:
#    - Trained parameters for each training iteration
#    - The resulting blocks/vox2ras0 matrices if SAVE_BLOCKS=True
#
#  *********************************  #

##################
# I/O parameters #
##################

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='change_orientation', choices=['change_orientation', 'P41-16', 'P58-16', 'P85-18', 'EX9-19'])
arg_parser.add_argument('--save_blocks', action='store_true', help='Save blocks and affine matrices in the end. '
                                                                   'Use predict_BT for that (if false)')

arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
SAVE_BLOCKS = arguments.save_blocks
BT_DB = config_database.get_lin_dict(SUBJECT)

config_file_data = config_donors.file_dict[SUBJECT]
RESULTS_DIR = join(config_file_data.BASE_DIR, 'RigidRegistration')
if not exists(RESULTS_DIR):
    makedirs(RESULTS_DIR)

kwargs_training = {'log_interval': 1}  # Number of steps
kwargs_generator = {'num_workers': 1, 'pin_memory': True}
device = torch.device("cuda:0")#torch.device("cpu")#

create_results_dir(RESULTS_DIR)
experimentWriter = ExperimentWriter(join(RESULTS_DIR, 'experiment.txt'), attach=False)

##############
# Parameters #
##############
# Data
torch_dtype = torch.float

downsampledVoxSize = 1# resolution of the downsampled blocks
reference_resolution = 1# resolution in plane of the reference volume

# Training
training_params = config_file_data.training_params
# training_params = {
#     'starting_iteration': 1,
#     'n_iterations': 9,
#     # Eugenio: new schedule
#     'schedule': [0,   2,   0,   3,   0,   4,   0,   4,   5],
#     # lbfgs run
#     'nepochs': [10, 10, 5, 10, 5, 10, 5, 10, 40],#, 20, 50, 100, 50, 100, 50, 100, 50, 100, 50],
#     # LBFGS
#     'learning_rate': [0.01,  0.005, 0.01, 0.005, 0.05, 0.005, 0.05, 0.05, 0.2],
#     'mask_weight': [5] * 9, # global overlap weight; this multiplies all the structure specific ones
#     'mask_cll_weight': [1] * 9,  # cerebellum overlap weight
#     'mask_cr_bs_weight': [1] * 9,  # cerebrum overlap weight
#     'image_weight': [1] * 9, # intensity loss weight
#     'scale_weight': [0.05] * 9,  # scaling loss weight
#     # Eugenio: we maybe want to change alpha/beta again in the future if we want to more harshly penalize overlaps (compared with gaps)
#     'alpha_weight': [1] * 9, # [2] * 11,  # parameter in overlap weight (for overlaps)
#     'beta_weight': [1] * 9,  # parameter in overlap weight (for gaps)
#     # Eugenio: select your loss. Global NCC seems to be doing a good job with the intensities
#     'loss_intensity': losses.NCC_Loss(device=device, win=[5, 5, 5], name='NCC'),
#     # 'loss_intensity': losses.NMI_Loss(device=device, name='NMI', bin_centers=np.linspace(0, 1, 32)),
#     # 'loss_intensity': losses.L1_Loss(device=device, name='L1'),
#     # 'loss_intensity': losses.GlobalNCC_Loss(device=device, name='GlobalNCC'),
#     'cp_spacing': 20  # Eugenio: control point spacing (in pixels) for nonlinear 2D transforms of slices. We should try different values...
# }  # training params dict, indicating the different models to be trained, number of epochs and loss weights.

###############
# Data Loader #
###############
experimentWriter.write('Loading dataset ...\n')

# Load dataset
data_loader = DataLoaderDownsample(BT_DB)

vol_shape = data_loader.vol_shape
vol_affine = data_loader.MRI_affine
subject_dict = data_loader.subject_dict
slice_id_list = data_loader.slice_id_list

dataset = datasets.BlockRegistrationBTDataset(data_loader)
mosaic = MakeGrayMosaicWithoutNormalization(res=downsampledVoxSize, device=device)

header_orig = {}
for it_bid, bid in enumerate(dataset.data_loader.subject_dict.keys()):
    _, _, header_orig[bid] = dataset[bid]

# Make initial mosaic
if True:#not exists(join(RESULTS_DIR, 'initMosaic.nii.gz')):
    img = mosaic.makeLinMosaic(dataset.images_dict, dataset.headers_dict)
    nib.save(img, join(RESULTS_DIR, 'initMosaic.nii.gz'))

############
# Training #
############

for k, v in training_params.items():
    experimentWriter.write(k)
    experimentWriter.write(k)
    experimentWriter.write(str(v))

for it_opt_level in range(training_params['n_iterations']):

    results_dir_schedule = join(RESULTS_DIR, str(it_opt_level+1))
    if not exists(results_dir_schedule):
        makedirs(results_dir_schedule)

    create_results_dir(results_dir_schedule)

    print('\n')
    print('####### ITERATION NUMBER ' + str(it_opt_level+1) + ' #######' )
    print('   - Model ' + str(training_params['schedule'][it_opt_level]))
    print('   - Number of epochs: ' + str(training_params['nepochs'][it_opt_level]))
    print('\n')

    # Models
    init_model_dict = {}
    init_optimizer_dict = {}
    warper = layers.SpatialTransformerAffine(vol_shape, padding_mode='zeros', torch_dtype=torch_dtype)
    model_dict = {'warper': warper.to(device)}

    parameters = []
    if training_params['schedule'][it_opt_level] == 0: #Global scaling. Parameters per structure: Angle (3), translation (3),
        shared_scaling = torch.nn.Parameter(torch.zeros(2, 1))
        for it_model in range(3):
            model = models.Instance3DGlobalScalingModel(vol_shape, vol_affine, shared_scaling, device=device)
            model = model.to(device)
            # In the future, it may be better to have a single optimizer, ideally with the closure method
            # implemented, so we can try LBFGS
            parameters += list(model.parameters())
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

    elif training_params['schedule'][it_opt_level] == 1: #Global scaling. Parameters per slice: angle (1, in-plane), translation (3),
        shared_scaling = torch.nn.Parameter(torch.zeros(2, 1))
        for it_model in slice_id_list:
            if it_model not in init_model_dict.keys():
                model = models.InstanceGlobalScalingModel(vol_shape, vol_affine, shared_scaling, device=device)
                init_model_dict[it_model] = model.to(device)
                parameters += list(model.parameters())

        for bid, block in subject_dict.items():
            init_model_dict[block.slice_id].normvecs.append(block.get_normvec())
            init_model_dict[block.slice_id].crs.append(block.get_cr())
            model_dict[bid] = init_model_dict[block.slice_id]
            # optimizer_dict[bid] = init_optimizer_dict[block.slice_id]


    elif training_params['schedule'][it_opt_level] == 2: #Global scaling. Parameters per block: angle (1, in-plane), translation (3).
        shared_scaling = torch.nn.Parameter(torch.zeros(2, 1))
        for bid, block in subject_dict.items():
            model = models.InstanceGlobalScalingModel(vol_shape, vol_affine, shared_scaling, device=device)
            model = model.to(device)
            model.normvecs.append(block.get_normvec())
            model.crs.append(block.get_cr())
            parameters += list(model.parameters())
            model_dict[bid] = model


    elif training_params['schedule'][it_opt_level] == 3: #Parameters per block: Angle (1, in-plane), translation (3), scaling (2)
        for bid, block in subject_dict.items():
            model = models.InstanceIndividualScalesModel(vol_shape, vol_affine, device=device)
            model = model.to(device)
            model.normvecs.append(block.get_normvec())
            model.crs.append(block.get_cr())
            parameters += list(model.parameters())
            model_dict[bid] = model

    elif training_params['schedule'][it_opt_level] == 4: #Parameters per block: Angle (3), translation (3), scaling (2)
        for bid, block in subject_dict.items():
            model = models.InstanceIndividualSimilarityModel(vol_shape, vol_affine, device=device)
            model.normvecs.append(block.get_normvec())
            model.crs.append(block.get_cr())
            model = model.to(device)
            parameters += list(model.parameters())
            model_dict[bid] = model

    # Eugenio: added mode 5, which is the same as 4 but with nonlinear transforms for every slice
    elif training_params['schedule'][it_opt_level] == 5: #Parameters per block: Angle (3), translation (3), scaling (2) + nonlinear per slice
        for bid, block in subject_dict.items():
            block_shape = block.vol_shape

            model = models.InstanceIndividualSimilarityModelWithNonlinear(vol_shape, vol_affine, block_shape,
                                                                          cp_spacing=training_params['cp_spacing'],
                                                                          device=device,
                                                                          torch_dtype=torch_dtype)
            model.normvecs.append(block.get_normvec())
            model.crs.append(block.get_cr())
            model = model.to(device)
            parameters += list(model.parameters())
            model_dict[bid] = model

    else:
        raise RuntimeError('Mode must be between 0 and 5')

    # optimizer = torch.optim.SGD(params=parameters, lr=training_params['learning_rate'][it_opt_level])
    optimizer = torch.optim.LBFGS(params=parameters, lr=training_params['learning_rate'][it_opt_level], max_iter=10)


    # Losses
    loss_function_dict = {
        'intensities': training_params['loss_intensity'],
        'overlap': losses.Overlap_Loss(name='overlap', alpha=training_params['alpha_weight'][it_opt_level],
                                      beta=training_params['beta_weight'][it_opt_level]),
        'scale': losses.Scaling_Loss(name='scale', refResolution=reference_resolution)
        }

    loss_weights_dict = {'intensities': training_params['image_weight'][it_opt_level],
                         'overlap': training_params['mask_weight'][it_opt_level],
                         'overlap_cerebellum': training_params['mask_cll_weight'][it_opt_level],
                         'overlap_cerebrum_brainstem': training_params['mask_cr_bs_weight'][it_opt_level],
                         'scale': training_params['scale_weight'][it_opt_level]}

    # Callbacks
    results_filepath = join(results_dir_schedule, 'results', 'training_results.csv')

    log_keys = ['loss_' + lossname for lossname in loss_function_dict.keys()] + ['loss', 'time_duration (s)']
    logger = History(log_keys)
    model_checkpoint = ModelCheckpoint(join(results_dir_schedule, 'checkpoints'), -1)
    training_printer = PrinterCallback()
    training_tocsv = ToCSVCallback(filepath=results_filepath, keys=log_keys)

    callback_list = [logger, model_checkpoint, training_printer, training_tocsv]
    callback_list += [LRDecay(optimizer, n_iter_start=0, n_iter_finish=training_params['nepochs'][it_opt_level])]

    for cb in callback_list:
        cb.on_train_init(model_dict)

    Training = LinearTraining(dataset, loss_function_dict, loss_weights_dict, callback_list, device)
    if it_opt_level + 1 >= training_params['starting_iteration']:
        for epoch in range(0, training_params['nepochs'][it_opt_level]):
            epoch_start_time = time.time()
            for cb in callback_list:
                cb.on_epoch_init(model_dict, epoch)

            t0 = time.time()
            logs_dict = Training.iterate(model_dict, optimizer, epoch, **kwargs_training)

            epoch_end_time = time.time()
            logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

            for cb in callback_list:
                cb.on_epoch_fi(logs_dict, model_dict, epoch, optimizer=optimizer, batch_size=1)

        for cb in callback_list:
            cb.on_train_fi(model_dict)

        if training_params['schedule'][it_opt_level] == 5:
            Training.update_headers_images(model_dict)

        else:
            Training.update_headers(model_dict)

        img = mosaic.makeLinMosaic(dataset.images_dict, dataset.headers_dict)
        if training_params['schedule'][it_opt_level] == 5:# nonlinear
            nib.save(img, join(RESULTS_DIR, 'grayMosaic_' + str(it_opt_level + 1) + '.lin.nii.gz'))
            img = mosaic.makeNonLinMosaic(dataset=dataset, model_dict=model_dict)

        nib.save(img, join(RESULTS_DIR, 'grayMosaic_' + str(it_opt_level+1) + '.nii.gz'))


    else:
        weightsfile = 'model_checkpoint.FI.pth'
        checkpoint = torch.load(join(results_dir_schedule, 'checkpoints', weightsfile), map_location=device)
        for bid, m_bid in model_dict.items():
            m_bid.load_state_dict(checkpoint['state_dict_' + bid])
            m_bid.eval()

        # img = mosaic.makeLinMosaic(dataset.images_dict, dataset.headers_dict)
        # if training_params['schedule'][it_opt_level] == 5:  # nonlinear
        #     nib.save(img, join(RESULTS_DIR, 'grayMosaic_' + str(it_opt_level + 1) + '.lin.nii.gz'))
        #     img = mosaic.makeNonLinMosaic(dataset=dataset, model_dict=model_dict)
        #
        # nib.save(img, join(RESULTS_DIR, 'grayMosaic_' + str(it_opt_level + 1) + '.nii.gz'))

        if training_params['schedule'][it_opt_level] == 5:
            Training.update_headers_images(model_dict)

        else:
            Training.update_headers(model_dict)

if SAVE_BLOCKS:
    print('Saving masks and affine matrices ...')
    # TODO: we should eventually do somethign with nonlinear transforms here...
    # I was wouldn't worry about deformed images; we'll run the ST3 tree later on
    # What we could do is to write the masks; we can smooth them a bit later on, and use them to mask the resampled MRI for SP3
    for it_bid, bid in enumerate(dataset.data_loader.subject_dict.keys()):
        header = dataset.data_loader.subject_dict[bid].affine
        LTA = dataset.data_loader.subject_dict[bid].get_lta()
        T = np.dot(header, np.dot(np.linalg.inv(header_orig[bid]), LTA))
        proxy = nib.load(dataset.data_loader.subject_dict[bid].init_image_gray)
        new_affine = np.dot(T, proxy.affine)
        data = np.asarray(proxy.dataobj)
        img = nib.Nifti1Image(data, new_affine)
        nib.save(img, join(RESULTS_DIR, 'results', BT_DB['SUBJECT'] + '_' + bid + '_volume.LFB.gray.reg.mgz'))

        proxy = nib.load(dataset.data_loader.subject_dict[bid].init_image_mask)
        new_affine = np.dot(T, proxy.affine)
        data = np.asarray(proxy.dataobj)
        img = nib.Nifti1Image(data, new_affine)
        nib.save(img, join(RESULTS_DIR, 'results', BT_DB['SUBJECT'] + '_' + bid + '_volume.LFB.mask.reg.mgz'))

        proxy = nib.load(dataset.data_loader.subject_dict[bid].init_image_rgb)
        new_affine = np.dot(T, proxy.affine)
        data = np.asarray(proxy.dataobj)
        img = nib.Nifti1Image(data, new_affine)
        nib.save(img, join(RESULTS_DIR, 'results', BT_DB['SUBJECT'] + '_' + bid + '_volume.LFB.rgb.reg.mgz'))

print('Done.')
