import pdb
from os.path import join, exists
import subprocess
import functools
import copy

import nibabel as nib
import torch
from torch import nn
import numpy as np
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import interpn
from PIL import Image

from src import models, datasets, layers
from utils.transforms import CropParams, Compose
from utils.deformation_utils import interpolate2D
from utils import image_utils
from setup_repo import *

def initialize_algorithm_masks(dataset, sampler):
    num_elements = len(dataset)
    image_shape = dataset.image_shape
    reference_mask = np.zeros((num_elements,) + image_shape)
    for it_batch, idx in enumerate(sampler):
        data_dict = dataset[idx]
        reference_mask[it_batch] = data_dict['x_ref_mask']
        if it_batch >= num_elements - 1:
            break

    return np.transpose(reference_mask, [1, 2, 0])

def initialize_graph_SbR(model, generator_data, parameter_dict, device):

    num_elements = len(generator_data.dataset)
    num_batches = len(generator_data)
    batch_size = generator_data.batch_size
    image_shape = generator_data.dataset.image_shape

    downsample_factor = [1/parameter_dict['UPSAMPLE_LEVELS']]*2
    vel_shape = tuple([int(i*d) for i,d in zip(image_shape,downsample_factor)])

    with torch.no_grad():

        registered_image = np.zeros((num_elements,) + image_shape)
        registered_mask = np.zeros((num_elements,) + image_shape)
        velocity_field = np.zeros((num_elements, 2) + vel_shape)
        deformation_field = np.zeros((num_elements, 2) + image_shape)

        for it_batch, data_dict in enumerate(generator_data):


            start = it_batch * batch_size
            end = start + batch_size
            if it_batch == num_batches - 1:
                end = num_elements

            if torch.sum(data_dict['x_ref_mask']) == 0 or torch.sum(data_dict['x_flo_mask']) == 0:
                continue

            ref_image = data_dict['x_ref'].to(device)
            flo_image = data_dict['x_flo'].to(device)
            flo_mask = data_dict['x_flo_mask_init'].to(device)

            flo_image_fake = model['G_M'](flo_image)
            flo_image_fake = flo_image_fake * flo_mask
            _, f, v = model['R_M'](flo_image_fake, ref_image)

            r = model['R_M'].predict(flo_image, f, svf=False)
            r_mask = model['R_M'].predict(flo_mask, f, svf=False, mode='nearest')

            registered_image[start:end] = np.squeeze(r.cpu().detach().numpy())
            registered_mask[start:end] = np.squeeze(r_mask.cpu().detach().numpy())
            velocity_field[start:end] = v.cpu().detach().numpy()
            deformation_field[start:end] = f.cpu().detach().numpy()

    velocity_field[np.isnan(velocity_field)] = 0
    deformation_field[np.isnan(deformation_field)] = 0

    return np.transpose(registered_image, [1, 2, 0]), np.transpose(registered_mask, [1, 2, 0]), \
           np.transpose(velocity_field, [1, 2, 3, 0]), np.transpose(deformation_field, [1, 2, 3, 0]),\

def initialize_graph_RegNet(model, generator_data, parameter_dict, device):

    num_elements = len(generator_data.dataset)
    num_batches = len(generator_data)
    batch_size = generator_data.batch_size
    image_shape = generator_data.dataset.image_shape

    downsample_factor = [1/parameter_dict['UPSAMPLE_LEVELS']]*2
    vel_shape = tuple([int(i*d) for i,d in zip(image_shape,downsample_factor)])

    with torch.no_grad():

        registered_image = np.zeros((num_elements,) + image_shape)
        registered_mask = np.zeros((num_elements,) + image_shape)
        velocity_field = np.zeros((num_elements, 2) + vel_shape)
        deformation_field = np.zeros((num_elements, 2) + image_shape)

        for it_batch, data_dict in enumerate(generator_data):

            start = it_batch * batch_size
            end = start + batch_size
            if it_batch == num_batches - 1:
                end = num_elements

            if torch.sum(data_dict['x_ref_mask']) == 0 or torch.sum(data_dict['x_flo_mask']) == 0:
                continue

            ref_image = data_dict['x_ref'].to(device)
            flo_image = data_dict['x_flo'].to(device)
            flo_mask = data_dict['x_flo_mask'].to(device)

            r, f, v = model(flo_image, ref_image)
            r_mask = model.predict(flo_mask, f, svf=False, mode='nearest')

            registered_image[start:end] = np.squeeze(r.cpu().detach().numpy())
            registered_mask[start:end] = np.squeeze(r_mask.cpu().detach().numpy())
            velocity_field[start:end] = v.cpu().detach().numpy()
            deformation_field[start:end] = f.cpu().detach().numpy()

    velocity_field[np.isnan(velocity_field)] = 0
    deformation_field[np.isnan(deformation_field)] = 0

    return np.transpose(registered_image, [1, 2, 0]), np.transpose(registered_mask, [1, 2, 0]), \
           np.transpose(velocity_field, [1, 2, 3, 0]), np.transpose(deformation_field, [1, 2, 3, 0]),\

def initialize_graph_NR(dataset, image_shape, scontrol, tempdir='/tmp'):

    # Filenames
    refFile = join(tempdir, 'refFile.png')
    floFile = join(tempdir, 'floFile.png')
    refMaskFile = join(tempdir, 'refMaskFile.png')
    floMaskFile = join(tempdir, 'floMaskFile.png')

    outputFile = join(tempdir, 'outputFile.png')
    outputMaskFile = join(tempdir, 'outputMaskFile.png')
    nonlinearField = join(tempdir, 'nonlinearField.nii.gz')
    dummyFileNifti = join(tempdir, 'dummyFileNifti.nii.gz')

    # Containers
    num_elements = len(dataset)
    registered_image = np.zeros((num_elements,) + image_shape)
    registered_mask = np.zeros((num_elements,) + image_shape)
    velocity_field = np.zeros((num_elements,2) + image_shape)
    displacement_field = np.zeros((num_elements,2) + image_shape)

    print('      Processing (N=' + str(num_elements) + '): ', end=' ', flush=True)
    nstep = 7
    # NiftyReg for all slices
    for it_batch in range(num_elements):
        print(str(it_batch), end=' ', flush=True)

        data_list = dataset[it_batch]

        x_ref = data_list[0]
        x_flo = data_list[1]
        m_ref = data_list[2]
        m_flo = data_list[3]
        if np.sum(m_ref) > 0 and np.sum(m_flo) > 0:

            #Save images
            img = Image.fromarray((255 * x_ref).astype(np.uint8), mode='L')
            img.save(refFile)
            img = Image.fromarray((255 * x_flo).astype(np.uint8), mode='L')
            img.save(floFile)
            img = Image.fromarray((255 * m_ref).astype(np.uint8), mode='L')
            img.save(refMaskFile)
            img = Image.fromarray((255 * m_flo).astype(np.uint8), mode='L')
            img.save(floMaskFile)

            # System calls
            subprocess.call([F3Dcmd, '-ref' , refFile , '-flo' , floFile , '-res', outputFile , '-cpp' , dummyFileNifti , '-sx', str(scontrol[0]), '-sy' , str(scontrol[1]), '-ln', '4', '-lp', '3', '--lncc', '7', '-pad', '0' , '-vel'], stdout=subprocess.DEVNULL)
            subprocess.call([TRANSFORMcmd  , '-ref' , refFile , '-flow' , dummyFileNifti, nonlinearField], stdout=subprocess.DEVNULL)
            subprocess.call([REScmd , '-ref', refMaskFile , '-flo', floMaskFile, '-trans' , nonlinearField , '-res' , outputMaskFile , '-inter', '0', '-voff'], stdout=subprocess.DEVNULL)

            #Saving images
            data = Image.open(outputFile)
            registered_image[it_batch:it_batch+1] = np.array(data) / 254.0

            data = Image.open(outputMaskFile)
            registered_mask[it_batch:it_batch+1] = np.array(data) / 254.0

            II, JJ = np.meshgrid(np.arange(0, image_shape[0]), np.arange(0, image_shape[1]), indexing='ij')

            proxy = nib.load(nonlinearField)
            proxyarray = np.transpose(np.squeeze(np.asarray(proxy.dataobj)),[2, 0, 1])
            proxyarray[np.isnan(proxyarray)] = 0
            finalarray = np.zeros_like(proxyarray)
            finalarray[0] = proxyarray[1] - II
            finalarray[1] = proxyarray[0] - JJ
            velocity_field[it_batch] = finalarray

            flow_i = finalarray[0]/2**nstep
            flow_j = finalarray[1]/2**nstep
            for it_step in range(nstep):
                di = II + flow_i
                dj = JJ + flow_j
                inci = interpolate2D(flow_i, np.stack([di, dj], -1))
                incj = interpolate2D(flow_j, np.stack([di, dj], -1))
                flow_i = flow_i + inci.reshape(image_shape)
                flow_j = flow_j + incj.reshape(image_shape)

            flow = np.concatenate((flow_i[np.newaxis],flow_j[np.newaxis]))
            displacement_field[it_batch] = flow

        else:
            registered_image[it_batch:it_batch + 1] = np.zeros(image_shape)
            registered_mask[it_batch:it_batch + 1] = np.zeros(image_shape)
            velocity_field[it_batch] = np.zeros((2,) + image_shape)
            displacement_field[it_batch] = np.zeros((2,) + image_shape)


    return np.transpose(registered_image,[1,2,0]), np.transpose(registered_mask,[1,2,0]), \
           np.transpose(velocity_field,[1,2,3,0]),  np.transpose(displacement_field,[1,2,3,0])

def integrate_NR(svf, image_shape):
    nstep = 7
    nslices = svf.shape[-1]

    II, JJ = np.meshgrid(np.arange(0, image_shape[0]), np.arange(0, image_shape[1]), indexing='ij')

    flow_i = svf[0] / 2 ** nstep
    flow_j = svf[1] / 2 ** nstep

    flow = np.zeros((2,) + image_shape + (nslices,))
    for it_slice in range(nslices):
        fi = resize(flow_i[..., it_slice], image_shape)
        fj = resize(flow_j[..., it_slice], image_shape)
        if np.sum(fi) + np.sum(fj) == 0:
            continue

        for it_step in range(nstep):
            di = II + fi
            dj = JJ + fi
            inci = interpolate2D(fi, np.stack([di, dj], -1))
            incj = interpolate2D(fj, np.stack([di, dj], -1))
            fi = fi + inci.reshape(image_shape)
            fj = fj + incj.reshape(image_shape)

        flow[0, ..., it_slice] = fi
        flow[1, ..., it_slice] = fj

    return flow

def integrate_RegNet(svf, image_shape, factor=2, int_end=1):
    integrator = layers.VecInt(svf.shape[1:3], int_steps=7, int_end=int_end)
    upscaler = layers.RescaleTransform(svf.shape[1:3], factor=factor)

    input_shape = tuple([i_s * factor for i_s in svf.shape[1:3]])
    nslices = svf.shape[-1]
    new_svf = torch.tensor(np.transpose(svf, [3, 0, 1, 2]))

    flow = integrator(new_svf)
    flow = upscaler(flow)
    flow = np.transpose(flow.detach().numpy(), [1, 2, 3, 0])

    flow_image = np.zeros((2,) + image_shape + (nslices,))

    parameter_dict = {'TRANSFORM': [CropParams(input_shape)]}

    transform = Compose(parameter_dict['TRANSFORM'])
    for it_slice in range(nslices):
        f = flow[..., it_slice]
        f_i, f_j = transform.inverse([f[0], f[1]], img_shape=[image_shape] * 2)
        flow_image[0, ..., it_slice] = f_i
        flow_image[1, ..., it_slice] = f_j

    return flow_image

def integrate_RegNet_old(svf, image_shape, parameter_dict):

    nslices = svf.shape[-1]
    input_shape = tuple([i_s * parameter_dict['UPSAMPLE_LEVELS'] for i_s in svf.shape[1:3]])
    model = models.RegNet(
        nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
        inshape=input_shape,
        int_steps=7,
        int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
    )
    new_svf = torch.tensor(np.transpose(svf, [3, 0, 1, 2]))
    flow = model.get_flow_field(new_svf)
    flow = np.transpose(flow.detach().numpy(), [1, 2, 3, 0])

    flow_image = np.zeros((2,) + image_shape + (nslices,))

    parameter_dict = {'TRANSFORM': [CropParams(input_shape)]}

    transform = Compose(parameter_dict['TRANSFORM'])
    for it_slice in range(nslices):
        f = flow[..., it_slice]
        f_i, f_j = transform.inverse([f[0], f[1]], img_shape=[image_shape] * 2)
        flow_image[0, ..., it_slice] = f_i
        flow_image[1, ..., it_slice] = f_j

    return flow_image

def get_dataset(block, parameter_dict, registration_type, image_shape=None, num_neighbours=None, fix_neighbors=True,
                mdil=15,**kwargs):

    if mdil:
        mask_dilation = np.ones((mdil, mdil))
    else:
        mask_dilation = False

    if image_shape is not None:
        factor = 2**len(parameter_dict['ENC_NF'])
        parameter_dict['IMAGE_SHAPE'] = tuple([int(factor * np.round(i_s/factor)) for i_s in image_shape])
        parameter_dict['TRANSFORM'] = [CropParams(crop_shape=parameter_dict['IMAGE_SHAPE'])]

    if registration_type == 'intermodal':
        dataset = datasets.BlockInterModalRegistrationDataset(
            data_loader=block,
            ref_modality=parameter_dict['REF_MODALITY'],
            flo_modality=parameter_dict['FLO_MODALITY'],
            affine_params=parameter_dict['AFFINE'],
            nonlinear_params=parameter_dict['NONLINEAR'],
            tf_params=parameter_dict['TRANSFORM'],
            norm_params=parameter_dict['NORMALIZATION'],
            mask_dilation=mask_dilation,
            **kwargs
        )
        sampler = datasets.BlockInterModalSampler(block=block)

    elif registration_type == 'intramodal':
        sampler = datasets.BlockIntraModalFixedNeighSampler(block=block, neighbor_distance=num_neighbours)
        dataset = datasets.BlockIntraModalRegistrationDataset(
            data_loader=block,
            modality=parameter_dict['REF_MODALITY'],
            affine_params=parameter_dict['AFFINE'],
            nonlinear_params=parameter_dict['NONLINEAR'],
            tf_params=parameter_dict['TRANSFORM'],
            norm_params=parameter_dict['NORMALIZATION'],
            mask_dilation=mask_dilation,
            neighbor_distance=-num_neighbours,
            fix_neighbors=fix_neighbors,
            **kwargs
        )


    else:
        print("Only updating the parameter dict")
        return

    return dataset, sampler


def get_model(model_type, image_shape, parameter_dict, device, weightsfile=None):

    if model_type == 'regnet':
        model = models.RegNet(
            nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
            inshape=image_shape,
            int_steps=7,
            int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
        )
        model.init_net(device, weightsfile=weightsfile)
        # model = model.to(device)

        # if weightsfile is not None:
        #     if not exists(weightsfile):
        #         raise ValueError("No trained weights are found for " + weightsfile)
        #
        #     checkpoint = torch.load(weightsfile, map_location=device)
        #     model.load_unet(checkpoint['state_dict'])
        #     model.load_state_dict(checkpoint['state_dict'], strict=True)
        #     model.eval()


    elif model_type == 'sbr':
        generator = models.ResnetGenerator(
            input_nc=1,
            output_nc=1,
            ngf=32,
            n_blocks=6,
            norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
            n_downsampling=3,
            tanh=False
        )
        # generator = generator.to(device)

        registration = models.RegNet(
            nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
            inshape=image_shape,
            int_steps=7,
            int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
        )
        # registration = registration.to(device)

        model = {'G_M': generator, 'R_M': registration}
        if weightsfile is not None:
            if not exists(weightsfile):
                raise ValueError("No trained weights are found for " + weightsfile)

            # checkpoint = torch.load(weightsfile, map_location=device)
            for model_key, m in model.items():
                m.init_net(device, weightsfile, model_key=model_key)
                # if model_key == 'R_M':
                #     m.load_unet(checkpoint['state_dict_' + model_key])
                # else:
                # m.load_state_dict(checkpoint['state_dict_' + model_key])
                # m.eval()
    else:
        raise ValueError("Please, specify a valid model_type [regnet, sbr]")

    return model

def inverse_tf(data, transform, image_shape, nslices):

    output_vol = []
    for d in data:
        vol = np.zeros(image_shape + (nslices,))
        for it_s in range(nslices):
            vol[..., it_s] = transform.inverse([d[..., it_s]],  img_shape=[image_shape] * 1)[0]
        output_vol.append(vol)

    return output_vol

def get_weightsfile(base_dir, sid, bid, epoch='FI'):
    dedicated_file = join(base_dir,  sid, bid, 'checkpoints', 'model_checkpoint.' + str(epoch) + '.pth')
    global_file = join(base_dir, 'checkpoints', 'model_checkpoint.' + str(epoch) + '.pth')
    if exists(dedicated_file):
        return dedicated_file
    else:
        return global_file


def generate_MRI_masks(data_loader_full, BLOCK_FILES, MASKS_DIR):

    blurKernelSigma = [2,2,8]
    MRI_MASK = data_loader_full.load_MRI_mask() > 0.5
    MRI_MASK_CEREBELLUM = np.squeeze(data_loader_full.load_MRI_mask_cerebellum()) > 0.5
    MRI_MASK_CEREBRUM = np.squeeze(data_loader_full.load_MRI_mask_cerebrum()) > 0.5
    MRI_MASK_BS = np.squeeze(data_loader_full.load_MRI_mask_brainstem()) > 0.5
    MRI_MASK_CLL_BS = MRI_MASK_CEREBELLUM | MRI_MASK_BS

    MRI_AFFINE = data_loader_full.MRI_affine

    I, J, K = np.meshgrid(np.arange(0, MRI_MASK.shape[0]),
                          np.arange(0, MRI_MASK.shape[1]),
                          np.arange(0, MRI_MASK.shape[2]), indexing='ij')

    I = I[MRI_MASK]
    J = J[MRI_MASK]
    K = K[MRI_MASK]

    mask_cerebrum_vec = MRI_MASK_CEREBRUM[MRI_MASK]
    mask_cer_bs_vec = MRI_MASK_CLL_BS[MRI_MASK]
    num_voxels = len(I)
    voxMosaic = np.concatenate((I.reshape(-1, 1), J.reshape(-1, 1), K.reshape(-1, 1), np.ones((num_voxels, 1))), axis=1)
    rasMosaic = np.dot(MRI_AFFINE, voxMosaic.T)
    blockID = np.zeros((num_voxels, ))
    blocks_mask = np.zeros((len(BLOCK_FILES), num_voxels, ), dtype=np.float)
    winningDist = 1e10 * np.ones((num_voxels, ))

    for it_block_filepath, block_filepath in enumerate(BLOCK_FILES):

        bid = block_filepath.split('_')[1]
        print('Working on block ' + bid + ' ('+ str(it_block_filepath) + '/' + str(len(BLOCK_FILES)) + ')')
        proxy_block = nib.load(join(MASKS_DIR, block_filepath))
        proxy_block = image_utils.padMRI(proxy_block, margin=[10, 10, 10])# margin=[40, 40, 10])
        vox2ras0 = copy.copy(proxy_block.affine)
        data_block = (np.asarray(proxy_block.dataobj) > 0).astype(np.float)

        # mask_block = data_block > 0
        mask_block = gaussian_filter(data_block, sigma=blurKernelSigma) > 0.5

        Din = -(distance_transform_edt(mask_block))
        Dout = distance_transform_edt(~mask_block)
        D = np.zeros_like(Din)
        D[mask_block] = Din[mask_block]
        D[~mask_block] = Dout[~mask_block]

        IJK = np.dot(np.linalg.inv(vox2ras0), rasMosaic)
        Ib = IJK[0]
        Jb = IJK[1]
        Kb = IJK[2]
        del IJK

        ok1 = Ib >= 0
        ok2 = Jb >= 0
        ok3 = Kb >= 0
        ok4 = Ib <= mask_block.shape[0] - 1
        ok5 = Jb <= mask_block.shape[1] - 1
        ok6 = Kb <= mask_block.shape[2] - 1
        ok = ok1 & ok2 & ok3 & ok4 & ok5 & ok6
        del ok1, ok2, ok3, ok4, ok5, ok6

        if 'B' in bid or 'C' in bid:
            ok[~mask_cer_bs_vec] = False
        else:
            ok[~mask_cerebrum_vec] = False

        points = (np.arange(0, D.shape[0]), np.arange(0, D.shape[1]), np.arange(0, D.shape[2]))
        xi = np.concatenate((Ib[ok].reshape(-1,1), Jb[ok].reshape(-1,1), Kb[ok].reshape(-1,1)), axis=1)
        dist_block = interpn(points, D, xi=xi, method='linear')
        dist = 1e10*np.ones((num_voxels,))
        dist[ok] = dist_block

        idx = dist < winningDist
        winningDist[idx] = dist[idx]
        blockID[idx] = it_block_filepath + 1
        blocks_mask[it_block_filepath] = dist

    return blockID, blocks_mask


