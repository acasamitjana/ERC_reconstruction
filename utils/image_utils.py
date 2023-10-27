import pdb

import copy

import numpy as np
import nibabel as nib
from skimage.morphology import binary_dilation
from scipy.special import softmax
from scipy.ndimage import distance_transform_edt, convolve, gaussian_filter
from munkres import Munkres
import torch
from torch.nn import functional as F
from scipy.interpolate import RegularGridInterpolator as rgi

torch_dtype = torch.float


#############
# Functions #
#############
def normalize_target_tensor(labels, class_labels=None, num_classes=None):

    if class_labels is None:
        if num_classes is None:
            raise ValueError('Need to specify class_labels or num_classes')
        else:
            class_labels = list(range(num_classes))

    for it_cl, cl in enumerate(class_labels):
        labels[labels == cl] = it_cl

    return labels

def one_hot_encoding(target, num_classes, categories=None):
    '''

    Parameters
    ----------
    target (np.array): target vector of dimension (d1, d2, ..., dN).
    num_classes (int): number of classes
    categories (None or list): existing categories. If set to None, we will consider only categories 0,...,num_classes

    Returns
    -------
    labels (np.array): one-hot target vector of dimension (num_classes, d1, d2, ..., dN)

    '''

    if categories is None:
        categories = list(range(num_classes))

    labels = np.zeros((num_classes,) + target.shape)
    for it_class in categories:
        idx_class = np.where(target == it_class)
        idx = (it_class,)+ idx_class
        labels[idx] = 1

    return labels.astype(int)

def filter_3d(image, kernel_sigma, device):
    kernel_size = kernel_sigma[0] * 3
    kernel_size += np.mod(kernel_size + 1, 2)
    padding = int((kernel_size - 1)//2)

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    coord = torch.arange(kernel_size)

    grids = torch.meshgrid([coord,coord,coord])
    grid = torch.stack(grids)  # y, x, z
    grid = torch.unsqueeze(grid, 0)  # add batch
    xyz_grid = grid.type(torch_dtype)

    mean = (kernel_size - 1) / 2.
    variance = kernel_sigma[0] ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    # 2.506628274631 = sqrt(2 * pi)

    norm_kernel = (1. / (np.sqrt(2*np.pi) * kernel_sigma[0])**3)
    kernel = norm_kernel * torch.exp(-torch.sum((xyz_grid - mean) ** 2., dim=1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to 2d depthwise convolutional weight
    kernel = kernel.view(1, 1, kernel_size, kernel_size,kernel_size)
    kernel = kernel.to(device)

    output = F.conv3d(image, kernel, stride=(1,1,1), padding=padding)

    return output

def grad3d(x):

    filter = np.asarray([-1,0,1])
    gx = convolve(x, np.reshape(filter, (3,1,1)), mode='constant')
    gy = convolve(x, np.reshape(filter, (1,3,1)), mode='constant')
    gz = convolve(x, np.reshape(filter, (1,1,3)), mode='constant')

    gx[0], gx[-1] = x[1] - x[0], x[-1] - x[-2]
    gy[:, 0], gy[:, -1] = x[:,1] - x[:,0], x[:, -1] - x[:, -2]
    gz[..., 0], gz[..., -1] = x[..., 1] - x[..., 0], x[..., -1] - x[..., -2]

    gmodule = np.sqrt(gx**2 + gy**2 + gz**2)
    return gmodule, gx, gy, gz

def crop_label(mask, margin=10, threshold=0):

    ndim = len(mask.shape)
    if isinstance(margin, int):
        margin=[margin]*ndim

    crop_coord = []
    idx = np.where(mask>threshold)
    for it_index, index in enumerate(idx):
        clow = max(0, np.min(idx[it_index]) - margin[it_index])
        chigh = min(mask.shape[it_index], np.max(idx[it_index]) + margin[it_index])
        crop_coord.append([clow, chigh])

    crop_coord_slice = tuple([slice(i[0], i[1]) for i in crop_coord])
    mask_cropped = mask[crop_coord_slice]

    return mask_cropped, crop_coord

def apply_crop(image, crop_coord):
    crop_coord_slice = tuple([slice(i[0], i[1]) for i in crop_coord])
    return image[crop_coord_slice]


##############
# Algorithms #
##############
def fast_marching_cubes(image, label_list, mask=None):

    # Compute geodesic distance with probability maps
    # see: https://arxiv.org/pdf/cs/0703082.pdf
    # image = np.argmax(image_onehot, axis=-1)
    if mask is None:
        mask = np.ones_like(image)
    res_shape = image.shape

    strel = np.ones((3,3))
    dists = np.zeros(res_shape + (len(label_list),))

    # One label at the time
    print(' (label:', end=' ', flush=True)
    for it_label, label in enumerate(label_list):
        if it_label == len(label_list)-1:
            print(str(int(label)), end=') \n', flush=True)
        else:
            print(str(int(label)), end=', ', flush=True)

        # initialize
        acc_d = np.zeros(res_shape)
        known = image == label

        trial = (binary_dilation(known, strel)) & (~known)
        trialvals = np.inf * np.ones(res_shape)
        idx = np.where(trial == True)

        its, jts = idx[0], idx[1]
        for it_its in range(its.shape[0]):
            it = its[it_its]
            jt = jts[it_its]
            d = computeNeighDist2D(image, known, acc_d, it, jt, mask)
            trialvals[it, jt] = d

        # fast marching
        ready = 0
        npix = 0
        while ready == 0:
            npix += 1
            # if np.mod(npix, 1000) == 0:
            #     print(str(npix) + '/' + str(np.sum(mask)) + ' ' + str(np.prod(res_shape)) + '/' + str(np.sum(known)))
            mini = np.min(trialvals)
            idx = np.where(trialvals == mini)
            idx = (idx[0][0], idx[1][0])

            known[idx[0], idx[1]] = 1
            trial[idx[0], idx[1]] = 0
            trialvals[idx[0], idx[1]] = np.inf
            acc_d[idx[0], idx[1]] = mini

            if (idx[0] > 0):
                if (known[idx[0] - 1, idx[1]] == 0):
                    trial[idx[0] - 1, idx[1]] = 1
                    trialvals[idx[0] - 1, idx[1]] = computeNeighDist2D(image, known, acc_d, idx[0] - 1, idx[1], mask)


            if (idx[0] < res_shape[0] - 1):
                if (known[idx[0] + 1, idx[1]] == 0):
                    trial[idx[0] + 1, idx[1]] = 1
                    trialvals[idx[0] + 1, idx[1]] = computeNeighDist2D(image, known, acc_d, idx[0] + 1, idx[1], mask)

            if (idx[1] > 0):
                if (known[idx[0], idx[1] - 1] == 0):
                    trial[idx[0], idx[1] - 1] = 1
                    trialvals[idx[0], idx[1] - 1] = computeNeighDist2D(image, known, acc_d, idx[0], idx[1] - 1, mask)


            if (idx[1] < res_shape[1] - 1):
                if (known[idx[0], idx[1] + 1] == 0):
                    trial[idx[0], idx[1] + 1] = 1
                    trialvals[idx[0], idx[1] + 1] = computeNeighDist2D(image, known, acc_d, idx[0], idx[1] + 1, mask)


            # if (idx[0] > 0) & (idx[1] > 0):
            #     if (known[idx[0] - 1, idx[1] - 1] == 0):
            #         trial[idx[0] - 1, idx[1] - 1] = 1
            #         trialvals[idx[0] - 1, idx[1] - 1] = computeNeighDist2D(image, known, acc_d, idx[0] - 1, idx[1] - 1, mask)
            #
            #
            # if (idx[0] < res_shape[0] - 1) & (idx[1] < res_shape[1] - 1):
            #     if (known[idx[0] + 1, idx[1] + 1] == 0):
            #         trial[idx[0] + 1, idx[1] + 1] = 1
            #         trialvals[idx[0] + 1, idx[1] + 1] = computeNeighDist2D(image, known, acc_d, idx[0] + 1, idx[1] + 1, mask)

            if np.sum(known) == np.prod(res_shape):
                ready = 1

            # if mask is not None:
            #     if np.sum(known[mask]) == np.sum(mask):
            #         ready = 1

        dists[..., it_label] = acc_d

    return dists

def computeNeighDist2D(image, known, acc_d, i, j, mask):

    label1 = image[i, j]
    d = np.inf
    if (i > 0):
        if known[i - 1, j] > 0:
            label2 = image[i - 1, j]
            m = mask[i - 1, j]
            d = min(d, acc_d[i - 1, j] + 1 + 1000000 * (1-m))

    if (i < image.shape[0] - 1):
        if known[i + 1, j] > 0:
            label2 = image[i + 1, j]
            m = mask[i + 1, j]
            d = min(d, acc_d[i + 1, j] + 1 + 1000000 * (1-m))

    if (j > 0):
        if known[i, j-1] > 0:
            label2 = image[i, j - 1]
            m = mask[i, j - 1]
            d = min(d, acc_d[i, j - 1] + 1 + 1000000 * (1-m))

    if (j < image.shape[1] - 1):
        if known[i, j + 1] > 0:
            label2 = image[i, j + 1]
            m = mask[i, j + 1]
            d = min(d, acc_d[i, j + 1] + 1 + 1000000 * (1-m))

    # if (i > 0) & (j > 0):
    #     if known[i - 1, j - 1] > 0:
    #         label2 = image[i - 1, j - 1]
    #         m = mask[i - 1, j - 1]
    #         d = min(d, acc_d[i - 1 , j - 1] + 1 + 1000 * ((label1 != label2) & (label2>0)) + 1000 * (m == 0))
    #
    # if (i < image.shape[0] - 1) & (j < image.shape[1] - 1):
    #     if known[i + 1, j + 1] > 0:
    #         label2 = image[i, j + 1]
    #         m = mask[i, j + 1]
    #         d = min(d, acc_d[i + 1, j + 1] + 1 + 1000 * ((label1 != label2) & (label2>0)) + 1000 * (m == 0))

    return d

def compute_distance_map(labelmap, soft_seg=True):
    unique_labels = np.unique(labelmap)
    distancemap = -200 * np.ones(labelmap.shape + (len(unique_labels),), dtype='float32')
    # print('Working in label: ', end='', flush=True)
    for it_ul, ul in enumerate(unique_labels):
        # print(str(ul), end=', ', flush=True)

        mask_label = labelmap == ul
        bbox_label, crop_coord = crop_label(mask_label, margin=5)

        d_in = (distance_transform_edt(bbox_label))
        d_out = -distance_transform_edt(~bbox_label)
        d = np.zeros_like(d_in)
        d[bbox_label] = d_in[bbox_label]
        d[~bbox_label] = d_out[~bbox_label]

        distancemap[crop_coord[0][0]: crop_coord[0][1],
                    crop_coord[1][0]: crop_coord[1][1],
                    crop_coord[2][0]: crop_coord[2][1], it_ul] = d


    if soft_seg:
        prior_labels = softmax(distancemap, axis=-1)
        # soft_labelmap = np.argmax(prior_labels, axis=-1).astype('uint16')
        return prior_labels
    else:
        return distancemap

#######
# MRI #
#######

def padBlock(proxy_mri, margin):
    if isinstance(margin, int):
        margin = [[margin, margin], [margin, margin], [0, 0]]

    vox2ras0 = copy.copy(proxy_mri.affine)
    input_vol = np.asarray(proxy_mri.dataobj)
    output_vol = np.zeros(tuple([i + m[0] + m[1] for i, m in zip(input_vol.shape, margin)]))
    output_vol[margin[0][0]: -margin[0][1], margin[1][0]: -margin[1][1]] = input_vol
    vox2ras0[:3, 3] = vox2ras0[:3, 3] - np.dot(vox2ras0[:3, :3], np.array([margin[0][0], margin[1][0], 0]))

    img = nib.Nifti1Image(output_vol, vox2ras0)
    return img


def padMRI(proxy_mri, margin):
    if isinstance(margin, int):
        margin = [margin]*3

    vox2ras0 = copy.copy(proxy_mri.affine)
    input_vol = np.asarray(proxy_mri.dataobj)
    output_vol = np.zeros(tuple([i + 2*m for i,m in zip(input_vol.shape, margin)]))
    output_vol[margin[0]: -margin[0], margin[1]: -margin[1], margin[2]: -margin[2]] = input_vol
    vox2ras0[:3, 3] = vox2ras0[:3, 3] - np.dot(vox2ras0[:3, :3], margin)

    img = nib.Nifti1Image(output_vol, vox2ras0)
    return img

def align_with_identity_vox2ras0(V, vox2ras0):

    COST = np.zeros((3,3))
    for i in range(3):
        for j in range(3):

            # worker is the vector
            b = vox2ras0[:3,i]

            # task is j:th axis
            a = np.zeros((3,1))
            a[j] = 1

            COST[i, j] = - np.abs(np.dot(a.T, b))/np.linalg.norm(a, 2)/np.linalg.norm(b, 2)

    m = Munkres()
    indexes = m.compute(COST.T)
    v2r = np.zeros_like(vox2ras0)
    for idx in indexes:
        v2r[:, idx[0]] = vox2ras0[:, idx[1]]
    v2r[:, 3] = vox2ras0[:, 3]
    V = np.transpose(V, axes=[idx[1] for idx in indexes])
    for d in range(3):
        if v2r[d,d] < 0:
            v2r[:3, d] = -v2r[:3, d]
            v2r[:3, 3] = v2r[:3, 3] - v2r[:3, d] * (V.shape[d] -1)
            V = np.flip(V, axis=d)

    return V, v2r

def rescale_voxel_size(volume, aff, new_vox_size):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    volume_filt = gaussian_filter(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = rgi((x, y, z), volume_filt)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2
