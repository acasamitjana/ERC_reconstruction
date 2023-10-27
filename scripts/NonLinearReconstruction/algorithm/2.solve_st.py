from os.path import exists
from os import makedirs
import time
from argparse import ArgumentParser
import shutil

# project imports
from dataset.data_loader import DataLoader
from config import config_database, config_dev, config_donors
from src import algorithm_helpers
from utils.deformation_utils import deform2D
from src.algorithm import *

# ******************************************************************************************************************** #
#
# This file runs the ST algorithm to compute the latent transforms form the spanning tree.
# Then, it integrates the computed  SVFs and deforms the HE and LFB sections to match their MRI counterparts.
#
# ******************************************************************************************************************** #


####################
# Input parameters #
####################
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--nc', type=int, default=2, choices=[2, 3], help='Number of contrasts')
arg_parser.add_argument('--c1', type=str, default='LFB', choices=['HE', 'LFB'], help='Contrast 1')
arg_parser.add_argument('--c2', type=str, default='', choices=['HE', 'LFB', ''], help='Contrast 2')
arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
arg_parser.add_argument('--nn', type=int, default=4, help='Number of neighbours')
arg_parser.add_argument('--mdil', type=int, default=7, help='Mask dilation factor')
arg_parser.add_argument('--subject', default='change_orientation', choices=list(config_donors.file_dict.keys()))
arg_parser.add_argument('--bid', default=None, nargs='+')
arg_parser.add_argument('--force', action='store_true')

arguments = arg_parser.parse_args()
ref = 'MRI'
N_CONTRASTS = arguments.nc
c1 = arguments.c1
c2 = arguments.c2
nneighbours = arguments.nn
cost = arguments.cost
mdil = arguments.mdil
SUBJECT = arguments.subject
initial_bid_list = arguments.bid
force_flag = arguments.force

BT_DB = config_database.get_nonlin_dict(SUBJECT)
data_dict = {SUBJECT: BT_DB}
config_file_data = config_donors.file_dict[SUBJECT]
parameter_dict_MRI = config_dev.CONFIG_DICT['MRI']

observations_dir = join(config_file_data.BASE_DIR, 'NonlinearRegistration', 'Registration')
algorithm_dir = join(config_file_data.BASE_DIR, 'NonlinearRegistration')
results_dir = join(algorithm_dir, 'ST' + str(N_CONTRASTS), cost, 'NN' + str(nneighbours))
if not exists(results_dir):
    makedirs(results_dir)

bid_list = initial_bid_list if initial_bid_list is not None else config_file_data.initial_bid_list
bid_list = {SUBJECT: bid_list}

data_loader = DataLoader(data_dict, included_blocks=bid_list)
subject_loader = data_loader.subject_dict[SUBJECT]

for it_block, block in enumerate(subject_loader):

    print('[START] Processing block: ' + str(block.id))
    input_dir = join(observations_dir, block.id)

    if not exists(join(input_dir, 'MRI_' + c1 + '.0N.velocity_field.nii.gz')) or \
        not exists(join(input_dir, 'MRI.' + str(nneighbours) + 'N.velocity_field.nii.gz')) or \
        not exists(join(input_dir, c1 + '.' + str(nneighbours) + 'N.velocity_field.nii.gz')) or \
        not (exists(join(input_dir, c1 + '_' + c2 + '.0N.velocity_field.nii.gz')) or c2 == '') or \
        not (exists(join(input_dir, 'MRI_' + c2 + '.0N.velocity_field.nii.gz')) or c2 == '') or \
        not (exists(join(input_dir, c2 + '.' + str(nneighbours) + 'N.velocity_field.nii.gz')) or c2 == ''):

        print('[WARNING] No observations found for subject ' + block.id + ', contrast ' + c1)
        continue
    else:
        proxy = nib.load(join(input_dir, c1 + '.' + str(nneighbours) + 'N.velocity_field.nii.gz'))
        cp_shape = proxy.shape[1:3]

    nslices = len(block.slice_list)
    results_dir_block = join(results_dir, block.id)
    if not exists(join(results_dir_block)):
        makedirs(results_dir_block)

    elif exists(join(results_dir_block, c1 + '.nii.gz')) and force_flag is False:
        print('[DONE] Subject ' + block.id + ' has already been processed')
        continue

    if force_flag is True or not exists(join(results_dir_block, 'MRI.nii.gz')):
        shutil.copy(join(BT_DB['BASE_DIR'], block.id, 'MRI_images.nii.gz'), join(results_dir_block, 'MRI.nii.gz'))

    block_shape = block.image_shape
    vox2ras0 = block.vox2ras0

    # Update parameter dic with transofrms
    algorithm_helpers.get_dataset(block=block, parameter_dict=parameter_dict_MRI,
                                  registration_type='none', image_shape=data_loader.image_shape)


    ####################################################################################################
    ####################################################################################################
    ####################
    # Run ST algorithm #
    ####################

    t_init = time.time()
    print('[' + str(block.id) + ' - BUILDING GRAPH] Reading SVFs ...')

    if force_flag is False and exists(join(results_dir_block, ref + '.velocity_field.nii.gz')):
        proxy = nib.load(join(results_dir_block, c1 + '.velocity_field.nii.gz'))
        T_C1 = np.asarray(proxy.dataobj)
        if N_CONTRASTS == 3:
            proxy = nib.load(join(results_dir_block, c2 + '.velocity_field.nii.gz'))
            T_C2 = np.asarray(proxy.dataobj)

    elif N_CONTRASTS == 2:
        graph_structure = init_st2(input_dir, cp_shape, nslices,
                                   nneighbours=nneighbours, se=np.ones((mdil, mdil)))

        R, M, W, d_inter, d_Ref, d_C1, NK = graph_structure
        print('[' + str(block.id) + ' - BUILDING GRAPH] Total Elapsed time: ' + str(time.time() - t_init))

        t_init = time.time()
        print('[' + str(block.id) + ' - ALGORITHM] Running the algorithm ...')
        if cost == 'l2':
            Tres = st2_L2(R, M, W, d_inter, d_Ref, d_C1, nslices, niter=5)

        else:
            Tres = st2_L1(R, M, W, nslices)

        T_C1 = Tres[..., :nslices]
        T_Ref = Tres[..., nslices:]

        img = nib.Nifti1Image(T_C1, vox2ras0)
        nib.save(img, join(results_dir_block, c1 + '.velocity_field.nii.gz'))
        img = nib.Nifti1Image(T_Ref, vox2ras0)
        nib.save(img, join(results_dir_block, ref + '.velocity_field.nii.gz'))


    elif N_CONTRASTS == 3:
            graph_structure = init_st3(input_dir, cp_shape, nslices, nneighbours=nneighbours, se=np.ones((mdil, mdil)),
                                       c1=c1, c2=c2)

            R, M, W, d_inter, d_Ref, d_C1, d_C2, NK = graph_structure
            print('[' + str(block.id) + ' - BUILDING GRAPH] Total Elapsed time: ' + str(time.time() - t_init))

            t_init = time.time()
            print('[' + str(block.id) + ' - ALGORITHM] Running the algorithm ...')
            if cost == 'l2':
                Tres = st3_L2(R, M, W, d_inter, d_Ref, d_C1, d_C2, nslices, niter=0)

            else:
                Tres = st3_L1(R, M, W, nslices)

            T_C1 = Tres[..., :nslices]
            T_C2 = Tres[..., nslices:2 * nslices]
            T_Ref = Tres[..., 2 * nslices:]

            img = nib.Nifti1Image(T_C1, vox2ras0)
            nib.save(img, join(results_dir_block, c1 + '.velocity_field.nii.gz'))

            img = nib.Nifti1Image(T_C2, vox2ras0)
            nib.save(img, join(results_dir_block, c2 + '.velocity_field.nii.gz'))

            img = nib.Nifti1Image(T_Ref, vox2ras0)
            nib.save(img, join(results_dir_block, ref + '.velocity_field.nii.gz'))

    print('[' + str(block.id) + ' - ALGORITHM] Total Elapsed time: ' + str(time.time() - t_init))

    ####################################################################################################
    ####################################################################################################
    #################
    # Integrate SVF #
    #################
    if force_flag is True or not exists(join(results_dir_block, c1 + '.flow.nii.gz')):
        t_init = time.time()
        print('[' + str(block.id) + ' - INTEGRATION] Computing deformation field ... ')
        flow_c1 = algorithm_helpers.integrate_RegNet(T_C1, block_shape, parameter_dict_MRI)
        img = nib.Nifti1Image(flow_c1, vox2ras0)
        nib.save(img, join(results_dir_block, c1 + '.flow.nii.gz'))

    if N_CONTRASTS == 3 and (force_flag is True or not exists(join(results_dir_block, c2 + '.flow.nii.gz'))):
        flow_c2 = algorithm_helpers.integrate_RegNet(T_C2, block_shape, parameter_dict_MRI)

        img = nib.Nifti1Image(flow_c2, vox2ras0)
        nib.save(img, join(results_dir_block, c2 + '.flow.nii.gz'))

    print('[' + str(block.id) + ' - INTEGRATION] Total Elapsed time: ' + str(time.time() - t_init))

    ####################################################################################################
    ####################################################################################################
    #################
    # Deform images #
    #################

    t_init = time.time()
    print('[' + str(block.id) + ' - DEFORM] Deforming images ... ')

    # HE
    if c1 == 'HE' or c2 == 'HE':

        proxy = nib.load(join(results_dir_block, 'HE.flow.nii.gz'))
        flow = np.asarray(proxy.dataobj)

        image_deformed = np.zeros(block_shape + (block.nslices,))
        mask_deformed = np.zeros(block_shape + (block.nslices,))

        for it_sl, sl in enumerate(block.slice_list):
            print('         Slice: ' + str(it_sl) + '/' + str(nslices))

            slice_num = int(sl.sid)-1

            image = sl.load_data(modality='HE')
            mask = sl.load_mask(modality='HE')

            image_deformed[..., slice_num] = deform2D(image.astype('float')/255, flow[..., it_sl])
            mask_deformed[..., slice_num] = deform2D(mask, flow[..., it_sl], mode='nearest')

            del image
            del mask

        img = nib.Nifti1Image(image_deformed, vox2ras0)
        nib.save(img, join(results_dir_block, 'HE.nii.gz'))

        img = nib.Nifti1Image(mask_deformed, vox2ras0)
        nib.save(img, join(results_dir_block, 'HE.mask.nii.gz'))


    if c1 == 'LFB' or c2 == 'LFB':
        # LFB
        proxy = nib.load(join(results_dir_block, 'LFB.flow.nii.gz'))
        flow = np.asarray(proxy.dataobj)

        image_deformed = np.zeros(block_shape + (block.nslices,))
        mask_deformed = np.zeros(block_shape + (block.nslices,))

        for it_sl, sl in enumerate(block.slice_list):
            print('         Slice: ' + str(it_sl) + '/' + str(nslices))
            slice_num = int(sl.sid)-1

            image = sl.load_data(modality='LFB')
            mask = sl.load_mask(modality='LFB')

            image_deformed[..., slice_num] = deform2D(image, flow[..., it_sl])
            mask_deformed[..., slice_num] = deform2D(mask, flow[..., it_sl], mode='nearest')

            del image
            del mask

        img = nib.Nifti1Image(image_deformed, vox2ras0)
        nib.save(img, join(results_dir_block, 'LFB.nii.gz'))

        img = nib.Nifti1Image(mask_deformed, vox2ras0)
        nib.save(img, join(results_dir_block, 'LFB.mask.nii.gz'))


    print('[' + str(block.id) + ' - DEFORM] Total Elapsed time: ' + str(time.time() - t_init))
