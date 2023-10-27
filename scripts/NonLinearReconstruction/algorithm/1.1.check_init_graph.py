from os.path import join, exists
import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi

from config import config_donors, config_database


# ******************************************************************************************************************** #
#
# This file groups the following reconstructions from all blocks to generate a 3D brain histology reconstruction at
# a given resolution (--res, 0.5 by default:
#   - LFB, HE grayscale
#   - LFB, HE color images
#   - Histological labels
#
# It uses the SVF computed using the ST algorithm with --nc contrasts, --cost objective function and --nn neigbours
#
# ******************************************************************************************************************** #

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--res', type=float, default=0.5)
arg_parser.add_argument('--subject', default='change_orientation', choices=list(config_donors.file_dict.keys()))

arguments = arg_parser.parse_args()
TARGET_RES = arguments.res
SUBJECT = arguments.subject

BT_DB = config_database.get_nonlin_dict(SUBJECT)
BT_DB_MRI = config_database.get_lin_dict(SUBJECT)['MRI']
config_file_data = config_donors.file_dict[SUBJECT]

blockspath = join(config_file_data.BASE_DIR, 'NonlinearRegistration', 'Registration')
outputdir = join(blockspath, 'mosaic_' + str(TARGET_RES))
if not exists(outputdir): os.makedirs(outputdir)

MRaseg = BT_DB_MRI['MRI_ASEG_FILE']#'/home/acasamitjana/Data/change_orientation/P57_16_scanned_20170607/mask_mri.rasoriented.nii.gz'
MRmask = BT_DB_MRI['MRI_MASK_FILE']#'/home/acasamitjana/Data/change_orientation/P57_16_scanned_20170607/mask_mri.rasoriented.nii.gz'
MRmask_cr = BT_DB_MRI['MRI_MASK_CEREBRUM_FILE']#'/home/acasamitjana/Data/change_orientation/P57_16_scanned_20170607/mask_cerebrum.rasoriented.nii.gz'
MRmask_cl = BT_DB_MRI['MRI_MASK_CEREBELLUM_FILE']#'/home/acasamitjana/Data/change_orientation/P57_16_scanned_20170607/mask_cerebellum.rasoriented.nii.gz'
MRmask_bs = BT_DB_MRI['MRI_MASK_BS_FILE']#'/home/acasamitjana/Data/change_orientation/P57_16_scanned_20170607/mask_brainstem.rasoriented.nii.gz'


MRfile = BT_DB_MRI['MRI_FILE']
proxy = nib.load(MRfile)
mri = np.asarray(proxy.dataobj)
volshape = mri.shape
vox2ras0 = proxy.affine.astype('float32')
corners = np.asarray([[0, 0, 0],
                      [0, 0, volshape[2]-1],
                      [0, volshape[1]-1, 0],
                      [0, volshape[1]-1, volshape[2]-1],
                      [volshape[0]-1, 0, 0],
                      [volshape[0]-1, 0, volshape[2]-1],
                      [volshape[0]-1, volshape[1]-1, 0],
                      [volshape[0]-1, volshape[1]-1, volshape[2]-1]]
                     )

cornersRAS = np.dot(vox2ras0, np.concatenate((corners, np.ones((8,1))),axis=1).T)
origin = np.min(cornersRAS[:3,:], axis=1)
spread = np.max(cornersRAS[:3,:], axis=1) - origin
new_vox2ras0 = np.diag([TARGET_RES, TARGET_RES, TARGET_RES, 1])
new_vox2ras0[:3, 3] = origin
new_vox2ras0 = new_vox2ras0.astype('float32')
siz = tuple(np.round(spread.T/TARGET_RES).astype('int'))
num_voxels = int(np.prod(siz))

x = np.arange(0, siz[0], dtype='float32')
y = np.arange(0, siz[1], dtype='float32')
z = np.arange(0, siz[2], dtype='float32')

XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')

voxMosaic = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1), ZZ.reshape(-1, 1), np.ones((num_voxels, 1), dtype='float32')), axis=1)

del XX, YY, ZZ

rasMosaic = np.dot(new_vox2ras0, voxMosaic.T)

del voxMosaic

print('Upsampling MRI... ')
if not exists(join(outputdir, 'upsampledMRI.nii.gz')) or not exists(join(outputdir, 'upsampledMRIMask.nii.gz')) \
        or not exists(join(outputdir, 'upsampledMRIMaskCR.nii.gz')) \
        or not exists(join(outputdir, 'upsampledMRIMaskCLL.nii.gz')) \
        or not exists(join(outputdir, 'upsampledMRIMaskBS.nii.gz')):

    IJK = np.dot(np.linalg.inv(vox2ras0), rasMosaic)
    Ib = IJK[0]
    Jb = IJK[1]
    Kb = IJK[2]

    del IJK

    ok1 = Ib >= 0
    ok2 = Jb >= 0
    ok3 = Kb >= 0
    ok4 = Ib <= mri.shape[0] - 1
    ok5 = Jb <= mri.shape[1] - 1
    ok6 = Kb <= mri.shape[2] - 1
    ok = ok1 & ok2 & ok3 & ok4 & ok5 & ok6

    del ok1, ok2, ok3, ok4, ok5, ok6
    x = np.arange(0, mri.shape[0], dtype='float32')
    y = np.arange(0, mri.shape[1], dtype='float32')
    z = np.arange(0, mri.shape[2], dtype='float32')
    points = (x,y,z)
    xi = np.concatenate((Ib[ok].reshape(-1, 1), Jb[ok].reshape(-1, 1), Kb[ok].reshape(-1, 1)), axis=1)

    del Ib, Jb, Kb
    if not exists(join(outputdir, 'upsampledMRI.nii.gz')):
        my_interpolating_function = rgi(points, mri)
        data = my_interpolating_function(xi)

        new_mri = np.zeros((num_voxels,))
        new_mri[ok] = data
        new_mri[np.isnan(new_mri)] = 0
        img = nib.Nifti1Image(new_mri.reshape(siz), new_vox2ras0)
        nib.save(img, join(outputdir, 'upsampledMRI.nii.gz'))

        del new_mri
        del data

    if not exists(join(outputdir, 'upsampledMRIMask.nii.gz')):
        proxy = nib.load(MRmask)
        mri_mask = np.asarray(proxy.dataobj)

        my_interpolating_function = rgi(points, mri_mask)
        data = my_interpolating_function(xi)

        new_mri = np.zeros((num_voxels,))
        new_mri[ok] = data
        new_mri[np.isnan(new_mri)] = 0
        img = nib.Nifti1Image(new_mri.reshape(siz).astype(np.float32), new_vox2ras0)
        nib.save(img, join(outputdir, 'upsampledMRIMask.nii.gz'))

        del new_mri
        del data

    if not exists(join(outputdir, 'upsampledMRIMaskCR.nii.gz')):
        proxy = nib.load(MRmask_cr)
        mri_mask = np.asarray(proxy.dataobj)

        my_interpolating_function = rgi(points, mri_mask)
        data = my_interpolating_function(xi)

        new_mri = np.zeros((num_voxels,))
        new_mri[ok] = data
        new_mri[np.isnan(new_mri)] = 0
        img = nib.Nifti1Image(new_mri.reshape(siz).astype(np.float32), new_vox2ras0)
        nib.save(img, join(outputdir, 'upsampledMRIMaskCR.nii.gz'))

        del new_mri
        del data

    if not exists(join(outputdir, 'upsampledMRIMaskCLL.nii.gz')):
        proxy = nib.load(MRmask_cl)
        mri_mask = np.asarray(proxy.dataobj)

        my_interpolating_function = rgi(points, mri_mask)
        data = my_interpolating_function(xi)

        new_mri = np.zeros((num_voxels,))
        new_mri[ok] = data
        new_mri[np.isnan(new_mri)] = 0
        img = nib.Nifti1Image(new_mri.reshape(siz).astype(np.float32), new_vox2ras0)
        nib.save(img, join(outputdir, 'upsampledMRIMaskCLL.nii.gz'))

        del new_mri
        del data

    if not exists(join(outputdir, 'upsampledMRIMaskBS.nii.gz')):
        proxy = nib.load(MRmask_bs)
        mri_mask = np.asarray(proxy.dataobj)

        my_interpolating_function = rgi(points, mri_mask)
        data = my_interpolating_function(xi)

        new_mri = np.zeros((num_voxels,))
        new_mri[ok] = data
        new_mri[np.isnan(new_mri)] = 0
        img = nib.Nifti1Image(new_mri.reshape(siz).astype(np.float32), new_vox2ras0)
        nib.save(img, join(outputdir, 'upsampledMRIMaskBS.nii.gz'))

        del new_mri
        del data

    del points
    del xi
    del ok

num_labels = 0
label_dict = {}
lfb_num = np.zeros((num_voxels,), dtype=np.float32)
lfb_num3 = np.zeros((num_voxels,3), dtype=np.float32)
lfb_den = np.zeros((num_voxels,), dtype=np.float32)

he_num = np.zeros((num_voxels,), dtype=np.float32)
he_num3 = np.zeros((num_voxels,3), dtype=np.float32)
he_den = np.zeros((num_voxels,), dtype=np.float32)

labels_volume = np.zeros((num_voxels,), dtype=np.uint16)
labels_cortical_volume = np.zeros((num_voxels,), dtype=np.uint16)

block_list = os.listdir(blockspath)
block_list = filter(lambda x: 'P' in x or 'A' in x or 'C' in x or 'B' in x, block_list)
files = ['MRI_LFB', 'MRI_HE']
print('Grouping blocks ... ')
for d in block_list:
    print(d, end='\n', flush=True)

    if d in ['B5.1', 'B6.1']:
        continue

    proxy = nib.load(join(blockspath, d, 'MRI_LFB.0N.nii.gz'))
    vox2ras0 = proxy.affine.astype('float32')

    IJK = np.dot(np.linalg.inv(vox2ras0), rasMosaic)
    Ib = IJK[0]
    Jb = IJK[1]
    Kb = IJK[2]

    del IJK

    ok1 = Ib >= 0
    ok2 = Jb >= 0
    ok3 = Kb >= 0
    ok4 = Ib <= proxy.shape[0] - 1
    ok5 = Jb <= proxy.shape[1] - 1
    ok6 = Kb <= proxy.shape[2] - 1
    ok = ok1 & ok2 & ok3 & ok4 & ok5 & ok6

    del ok1, ok2, ok3, ok4, ok5, ok6

    points = (np.arange(0, proxy.shape[0]), np.arange(0, proxy.shape[1]), np.arange(0, proxy.shape[2]))
    xi = np.concatenate((Ib[ok].reshape(-1, 1), Jb[ok].reshape(-1, 1), Kb[ok].reshape(-1, 1)), axis=1)

    del Ib, Jb, Kb

    #LFB
    proxy = nib.load(join(blockspath, d, 'MRI_LFB.0N.nii.gz'))
    mri = np.asarray(proxy.dataobj)

    my_interpolating_function = rgi(points, mri)
    data = my_interpolating_function(xi)

    my_interpolating_function = rgi(points, mri>0)
    data_prob = my_interpolating_function(xi)

    data = data.astype(np.float32)
    data_prob = data_prob.astype(np.float32)
    lfb_num[ok] += data * data_prob
    lfb_den[ok] += data_prob

    del data

    #HE
    proxy = nib.load(join(blockspath, d, 'MRI_HE.0N.nii.gz'))
    mri = np.asarray(proxy.dataobj)

    my_interpolating_function = rgi(points, mri)
    data = my_interpolating_function(xi)

    my_interpolating_function = rgi(points, mri>0)
    data_prob = my_interpolating_function(xi)

    data = data.astype(np.float32)
    data_prob = data_prob.astype(np.float32)

    he_num[ok] += data * data_prob
    he_den[ok] += data_prob

    del data


lfb_volume = lfb_num / (lfb_den + 1e-5)
lfb_volume = lfb_volume.reshape(siz)
img = nib.Nifti1Image(lfb_volume.astype('float32'), new_vox2ras0)
nib.save(img, join(outputdir, 'MRI_LFB.nii.gz'))

del lfb_volume

he_volume = he_num / (he_den + 1e-5)
he_volume = he_volume.reshape(siz)
img = nib.Nifti1Image(he_volume.astype('float32'), new_vox2ras0)
nib.save(img, join(outputdir, 'MRI_HE.nii.gz'))

del he_volume