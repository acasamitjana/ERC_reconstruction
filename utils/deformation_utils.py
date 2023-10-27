import os
import pdb

import numpy as np
import nibabel as nib
from scipy.interpolate import interpn, RegularGridInterpolator as rgi

import torch

torch_dtype = torch.float


##########
# Linear #
##########
def get_affine_from_rotation(angle_list):

    affine_matrix = np.zeros((len(angle_list), 2,3))
    for it_a, angle in enumerate(angle_list):
        angle_rad = angle * np.pi / 180
        affine_matrix[it_a] = np.array([
            [np.cos(angle_rad).item(), -np.sin(angle_rad).item(), 0],
            [np.sin(angle_rad).item(), np.cos(angle_rad).item(), 0],
        ])
    return affine_matrix

def affine_to_dense(affine_matrix, volshape):

    ndims = len(volshape)

    vectors = [np.arange(0, s) for s in volshape]
    YY, XX = np.meshgrid(*vectors, indexing=('ij')) #grid of vectors
    mesh = [XX, YY]
    mesh = [f.astype('float32') for f in mesh]
    mesh = [mesh[f] - (volshape[ndims - f - 1] - 1) / 2 for f in range(ndims)] #shift center

    # mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    # mesh = [tf.cast(f, 'float32') for f in mesh]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [np.reshape(f, (-1,)) for f in mesh]
    flat_mesh.append(np.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = np.transpose(np.stack(flat_mesh, axis=1))  # ndims+1 x nb_voxels

    # compute locations
    loc_matrix = np.matmul(affine_matrix, mesh_matrix)  # ndims+1 x nb_voxels
    loc = np.reshape(loc_matrix[:ndims, :], [ndims] + list(volshape))  # ndims x *volshape

    # get shifts and return

    shift = loc - np.stack(mesh, axis=0)
    return shift.astype('float32')

class MakeGrayMosaicWithoutNormalization(object):

    '''
    Class to make mosaic puzzle images
    '''

    def __init__(self, res, device='cpu'):
        self.res = res
        self.device = device

    def makeMosaic(self, image_block, headers,):
        # Get the maximum size
        maxSize = 0
        for image in image_block.values():
            maxSize = max(maxSize, max(image.shape))

        # Get the corners of cuboid in RAS space
        minR, minA, minS = np.inf, np.inf, np.inf
        maxR, maxA, maxS = -np.inf, -np.inf, -np.inf

        for iid, image in image_block.items():
            for i in [0, image.shape[0]]:
                for j in [0, image.shape[1]]:
                    for k in [0, image.shape[2]]:
                        aux = np.dot(headers[iid], np.asarray([i, j, k, 1]).T)

                        minR, maxR = min(minR, aux[0]), max(maxR, aux[0])
                        minA, maxA = min(minA, aux[1]), max(maxA, aux[1])
                        minS, maxS = min(minS, aux[2]), max(maxS, aux[2])

        # Define header and size
        vox2ras0 = np.asarray([[self.res, 0, 0, minR],
                               [0, self.res, 0, minA],
                               [0, 0, self.res, minS],
                               [0, 0, 0, 1]])

        size = np.asarray([np.int(np.ceil((maxR - minR) / self.res)),
                           np.int(np.ceil((maxA - minA) / self.res)),
                           np.int(np.ceil((maxS - minS) / self.res))])
        vol = np.zeros(size)

        # Resample
        RR, AA, SS = np.meshgrid(np.arange(0, size[0]), np.arange(0, size[1]), np.arange(0, size[2]), indexing='ij')
        voxMosaic = np.concatenate(
            (RR.reshape(-1, 1), AA.reshape(-1, 1), SS.reshape(-1, 1), np.ones((np.prod(size), 1))), axis=1)
        rasMosaic = np.dot(vox2ras0, voxMosaic.T)

        for iid, image in image_block.items():
            voxB = np.dot(np.linalg.inv(headers[iid]), rasMosaic)
            vR = voxB[0]
            vA = voxB[1]
            vS = voxB[2]
            ok1 = vR >= 0
            ok2 = vA >= 0
            ok3 = vS >= 0
            ok4 = vR <= image.shape[0] - 1
            ok5 = vA <= image.shape[1] - 1
            ok6 = vS <= image.shape[2] - 1
            ok = ok1 & ok2 & ok3 & ok4 & ok5 & ok6

            xiR = np.reshape(vR[ok], (-1, 1))
            xiA = np.reshape(vA[ok], (-1, 1))
            xiS = np.reshape(vS[ok], (-1, 1))
            xi = np.concatenate((xiR, xiA, xiS), axis=1)

            vals = np.zeros(vR.shape[0])
            vals[ok] = interpn((np.arange(0, image.shape[0]),
                                np.arange(0, image.shape[1]),
                                np.arange(0, image.shape[2])), image, xi=xi)

            intensities = vals.reshape(size)
            vol += intensities

        img = nib.Nifti1Image(vol, vox2ras0)
        return img

    def makeLinMosaic(self, image_block, headers):
        return self.makeMosaic(image_block, headers)

    def makeNonLinMosaic(self, dataset, model_dict):

        # nonlinear case
        image_block = {}
        for it_bid, bid in enumerate(dataset.data_loader.subject_dict.keys()):
            image, mask, header = dataset[bid]
            image, mask, header = image.to(self.device), mask.to(self.device), header.to(self.device)

            model = model_dict[bid]
            affine, new_header, fields = model(header.to(self.device), image.shape)
            fields = fields.type(torch.float)
            nonlin_image = model.warp(image, fields)
            image_block[bid] = np.squeeze(nonlin_image.to('cpu').detach().numpy())

        return self.makeMosaic(image_block, dataset.headers_dict)


#################
# Interpolation #
#################


def interpolate3D(image, deformation, vox2ras0=None, mode='bilinear'):
    '''

    :param image: 2D np.array (nrow, ncol)
    :param deformation: 3D np.array (2, nrow, ncol)
    :param mode: 'bilinear' or 'nearest'
    :return:
    '''

    if mode == 'bilinear': mode='linear'
    II = deformation[..., 0]
    JJ = deformation[..., 1]
    KK = deformation[..., 2]
    ref_shape = deformation.shape[:-1]
    output_shape = ref_shape + image.shape[3:]

    del deformation

    if vox2ras0 is not None:
        num_voxels = np.prod(ref_shape)
        voxMosaic = np.dot(np.linalg.inv(vox2ras0),
                           np.concatenate((II.reshape(-1, 1),
                                           JJ.reshape(-1, 1),
                                           KK.reshape(-1, 1),
                                           np.ones((num_voxels, 1))), axis=1).T)
        IId = voxMosaic[0].reshape(ref_shape)
        JJd = voxMosaic[1].reshape(ref_shape)
        KKd = voxMosaic[2].reshape(ref_shape)
        del voxMosaic
    else:
        IId = II
        JJd = JJ
        KKd = KK

    del II, JJ, KK

    minR, maxR = 0, image.shape[0] - 1
    minA, maxA = 0, image.shape[1] - 1
    minS, maxS = 0, image.shape[2] - 1

    points = (np.linspace(minR, maxR, image.shape[0]),
              np.linspace(minA, maxA, image.shape[1]),
              np.linspace(minS, maxS, image.shape[2]))

    my_interpolation_function = rgi(points, image, method=mode, bounds_error=False, fill_value=0)

    ok1 = IId >= minR
    ok2 = JJd >= minA
    ok3 = KKd >= minS
    ok4 = IId <= maxR
    ok5 = JJd <= maxA
    ok6 = KKd <= maxS
    ok = ok1 & ok2 & ok3 & ok4 & ok5 & ok6

    del ok1, ok2, ok3, ok4, ok5, ok6

    xi = np.concatenate((IId[ok].reshape(-1, 1), JJd[ok].reshape(-1, 1), KKd[ok].reshape(-1, 1)), axis=1)

    del IId, JJd, KKd


    if mode == 'onehot':

        unique_labels = np.unique(image)
        output_flat = np.zeros((xi.shape[0], len(unique_labels)))
        for it_ul, ul in enumerate(unique_labels):
            image_onehot = np.zeros_like(image).astype('float32')
            image_onehot[image == ul] = 1

            my_interpolation_function = rgi(points, image_onehot, method='linear', bounds_error=False, fill_value=0)
            output_flat[..., it_ul] = my_interpolation_function(xi)

        del points, xi, image

        output_unordered = np.argmax(output_flat, axis=-1).astype('uint16')
        del output_flat

        output_ordered = np.zeros_like(output_unordered)
        for it_ul, ul in enumerate(unique_labels): output_ordered[output_unordered == it_ul] = ul

        output = np.zeros(output_shape)
        output[ok] = output_ordered

    else:
        output_flat = my_interpolation_function(xi)
        output = np.zeros(output_shape)
        output[ok] = output_flat

    return output


def interpolate2D(image, mosaic, resized_shape=None, mode='linear'):
    '''

    :param image: nib.nifti1.Nifti1Image, np.array or list of np.arrays.
    :param mosaic: Nx2
    :param vox2ras0: optional if the mosaic is specified at ras space.
    :param mode: 'nearest' or 'linear'
    :return:
    '''

    image_shape = image.shape
    if len(image.shape) > 2:
        nchannels = image.shape[2]
    else:
        nchannels = 1
        image = image[..., np.newaxis]

    x = np.arange(0, image_shape[0])
    y = np.arange(0, image_shape[1])

    ok1 = mosaic[..., 0] >= 0
    ok2 = mosaic[..., 1] >= 0
    ok3 = mosaic[..., 0] <= image_shape[0]
    ok4 = mosaic[..., 1] <= image_shape[1]
    ok = ok1 & ok2 & ok3 & ok4

    del ok1, ok2, ok3, ok4

    my_interpolation_function = rgi((x, y), image, method=mode, bounds_error=False, fill_value=0)

    del x, y, image

    im_resampled_ok = my_interpolation_function(mosaic[ok])

    output_flat = np.zeros(ok.shape + (nchannels,))
    output_flat[ok] = im_resampled_ok

    del im_resampled_ok

    if resized_shape is not None:
        output = output_flat.reshape(resized_shape + (nchannels,))
    else:
        output = output_flat

    del output_flat

    if nchannels == 1:
        output = output[..., 0]

    return output



###############
# Deformation #
###############

def deform2D(image, field, mode='linear'):
    '''
    :param image: 2D np.array (nrow, ncol)
    :param field: 3D np.array (2, nrow, ncol)
    :param mode: 'linear' or 'nearest'
    :return:
    '''

    if mode == 'bilinear': mode = 'linear'

    output_shape = field.shape[1:]

    dI = field[0]
    dJ = field[1]

    del field

    II, JJ = np.meshgrid(np.arange(0, output_shape[0]), np.arange(0, output_shape[1]), indexing='ij')

    IId = II + dI
    JJd = JJ + dJ

    del JJ, II, dI, dJ

    mosaic = np.concatenate((IId.reshape(-1, 1), JJd.reshape(-1, 1)), axis=1)

    del JJd, IId

    output = interpolate2D(image, mosaic, resized_shape=output_shape, mode=mode)

    return output

