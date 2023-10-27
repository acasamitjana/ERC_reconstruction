import pdb
from os import listdir, remove
import shutil
import json
from argparse import ArgumentParser
import csv

import nibabel as nib
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.morphology import ball
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from webptools import cwebp

from utils.io_utils import create_results_dir, load_lut
from utils.image_utils import compute_distance_map
from utils.webpage_utils import DICT_CLASSES, get_num_slices_dict
from config import config_donors
from setup_repo import *


arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='P57-16', choices=['P57-16', 'P58-16', 'P41-16', 'P85-18', 'EX9-19'])
arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
arg_parser.add_argument('--nn', type=int, default=2, help='Number of neighbours')
arg_parser.add_argument('--downsample_factor', type=int, default=2, help='Downsample factor for Histology HR')
arg_parser.add_argument('--do_mri', action='store_true', help='Process mri directory')
arg_parser.add_argument('--do_histo', action='store_true', help='Process Histology directory')
arg_parser.add_argument('--do_histo_hr', action='store_true', help='Process histology_hr directory')

arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
NEIGHBORS = arguments.nn
cost = arguments.cost
downsample_factor = arguments.downsample_factor
DO_MRI = arguments.do_mri
DO_HISTOLOGY = arguments.do_histo
DO_HISTOLOGY_HR = arguments.do_histo_hr

Image.MAX_IMAGE_PIXELS = 1024*1024*1024//4
ORIENTATION_CLASS = DICT_CLASSES[SUBJECT]()

# Directories used
DATA_SUBJECT_DIR = join(DATA_DIR, SUBJECT)
PROCESSED_DIR = join(RESULTS_DIR, SUBJECT, 'NonlinearRegistration', 'ST3', cost, 'NN' + str(NEIGHBORS))
RIGID_RESULTS = join(RESULTS_DIR, SUBJECT, 'RigidRegistration', 'results')

PARENT_DIR = join(WEBPAGE_DIR, 'BrainAtlas-' + SUBJECT, SUBJECT)
MRI_DIR = join(PARENT_DIR, 'mri_rotated')
HISTOLOGY_DIR = join(PARENT_DIR, 'histology')
HISTOLOGY_HR_DIR = join(PARENT_DIR, 'histology_hr')
RGB_LUT = load_lut(join(WEBPAGE_DIR, 'AllenAtlasLUT'))

dir_mri = ['slices_sagittal', 'slices_coronal', 'slices_axial',
           'indices_sagittal', 'indices_coronal', 'indices_axial',
           'matrices', 'matrices_hr']

dir_hist = ['slices_LFB', 'slices_HE', 'slices_MRI', 'slices_labels', 'slices_labels_npz']
dir_hist_hr = ['slices_LFB', 'slices_HE', 'slices_MRI', 'slices_labels', 'slices_labels_npz']
create_results_dir(MRI_DIR, subdirs=dir_mri)
num_slice_dict = get_num_slices_dict(join(RIGID_RESULTS, 'slices', 'slice_id.txt'))

# Parameters
config_file_data = config_donors.file_dict[SUBJECT]
BLOCK_res = config_file_data.BLOCK_res
HISTO_res = config_file_data.HISTO_res
flip_blocks = config_file_data.flip_mri_blocks

if not exists(HISTOLOGY_DIR): makedirs(HISTOLOGY_DIR)

print('# -------------- #')
print('Processing subject ' + str(SUBJECT) + ', with NNEIGH=' + str(NEIGHBORS) + ' and downsampling factor ' + str(downsample_factor))
print('MRI: ' + str(DO_MRI) + '\nHISTO: ' + str(DO_HISTOLOGY) + '\nHISTO_HR: ' + str(DO_HISTOLOGY_HR))
print('# -------------- #')

if not exists(join(MRI_DIR, 'indices.nii.gz')):
    if exists(join(RIGID_RESULTS, 'BLOCK_MASK_MOSAIC.rasoriented.B5incl.nii.gz')):
        shutil.copy(join(RIGID_RESULTS, 'BLOCK_MASK_MOSAIC.rasoriented.B5incl.nii.gz'), join(MRI_DIR, 'indices.nii.gz'))
    else:
        shutil.copy(join(RIGID_RESULTS, 'BLOCK_MASK_MOSAIC.rasoriented.nii.gz'), join(MRI_DIR, 'indices.nii.gz'))

    proxy = nib.load(join(MRI_DIR, 'indices.nii.gz'))
    indices_vol = np.squeeze(np.asarray(proxy.dataobj))
    indices_vol = indices_vol.astype('uint8')
    mask_indices = (indices_vol > 0).astype('uint8')

    struct = ball(3)
    mask_mri = binary_dilation(mask_indices, struct)
    mask_mri = binary_fill_holes(mask_mri)
    img = nib.Nifti1Image(mask_mri.astype('uint8'), proxy.affine)
    nib.save(img, join(MRI_DIR, 'prova.nii.gz'))

    soft_indices = compute_distance_map(indices_vol, soft_seg=True)

    hard_indices = np.argmax(soft_indices[..., 1:], axis=-1) + 1
    indices_vol[mask_mri+mask_indices==1] = hard_indices[mask_mri+mask_indices==1]
    img = nib.Nifti1Image(indices_vol, proxy.affine)
    nib.save(img, join(MRI_DIR, 'indices.nii.gz'))

else:
    proxy = nib.load(join(MRI_DIR, 'indices.nii.gz'))
    indices_vol = np.asarray(proxy.dataobj)
    indices_vol = indices_vol.astype('uint8')

if not exists(join(MRI_DIR, 'mri.nii.gz')):
    if exists(join(DATA_SUBJECT_DIR, 'MRI', 'averageWithReg.stripped.bfcorr.rasoriented.B5incl.nii.gz')):
        shutil.copy(join(DATA_SUBJECT_DIR, 'MRI', 'averageWithReg.stripped.bfcorr.rasoriented.B5incl.nii.gz'), join(MRI_DIR, 'mri.nii.gz'))
    else:
        shutil.copy(join(DATA_SUBJECT_DIR, 'MRI', 'averageWithReg.stripped.bfcorr.rasoriented.nii.gz'), join(MRI_DIR, 'mri.nii.gz'))

    proxy = nib.load(join(MRI_DIR, 'mri.nii.gz'))
    vox2ras_MRI = proxy.affine
    data = np.asarray(proxy.dataobj, order='C')
    # data = (255 * data/np.max(data)).astype('uint8')
    data = (255 * data/np.percentile(data, 99))
    data = np.clip(data, 0, 255).astype('uint8')
    data = np.squeeze(data)
    data[indices_vol==0] = 0
    img = nib.Nifti1Image(data, proxy.affine)
    nib.save(img, join(MRI_DIR, 'mri.nii.gz'))

else:
    proxy = nib.load(join(MRI_DIR, 'mri.nii.gz'))
    vox2ras_MRI = proxy.affine
    data = np.asarray(proxy.dataobj, order='C')
    # data = (255 * data/np.percentile(data, 99))
    # data = np.clip(data, 0, 255).astype('uint8')
    # data = np.squeeze(data)

# ------- #
# MRI DIR #
# ------- #
print('\n\n\n')
print(' --- MRI --- '+ str(DO_MRI))
print('\n')
if DO_MRI:

    d_MRI = {
        "sagittal": {
            "width": data.shape[1],
            "height": data.shape[2],
            "slices": data.shape[0],
            "coordinateMappings": {
                "axisX": "slice",
                "axisY": "mouseX",
                "axisZ": "mouseY"
            }
        },
        "coronal": {
            "width": data.shape[0],
            "height": data.shape[2],
            "slices": data.shape[1],
            "coordinateMappings": {
                "axisX": "mouseX",
                "axisY": "slice",
                "axisZ": "mouseY"
            }
        },
        "axial": {
            "width": data.shape[0],
            "height": data.shape[1],
            "slices": data.shape[2],
            "coordinateMappings": {
                "axisX": "mouseX",
                "axisY": "mouseY",
                "axisZ": "slice"
            }
        }
    }

    json_object = json.dumps(d_MRI, indent=4)
    with open(join(PARENT_DIR, 'mriDimensionsKey.json'), "w") as outfile:
        outfile.write(json_object)

    for it_dim in range(3):
        axes = [it_dim] + list(range(it_dim)) + list(range(it_dim+1,3))
        data_T = np.transpose(data, axes=axes)
        indices_vol_T = np.transpose(indices_vol, axes=axes)
        nslices = data.shape[it_dim]
        for it_s in range(nslices):
            slice_i = np.squeeze(data_T[it_s])
            index_i = np.squeeze(indices_vol_T[it_s])
            if it_dim == 0:
                slice_i = np.rot90(slice_i)
                index_i = np.rot90(index_i)
            else:
                slice_i = np.fliplr(np.rot90(slice_i))
                index_i = np.fliplr(np.rot90(index_i))

            img = Image.fromarray(slice_i, mode='L')
            if np.sum(slice_i) == 0:
                img.save(join(MRI_DIR, dir_mri[it_dim], 'slice_' + "{:03d}".format(it_s) + '.png'), compress_level=9)
            else:
                img.save(join(MRI_DIR, dir_mri[it_dim], 'slice_' + "{:03d}".format(it_s) + '.png'), compress_level=0)
            np.save(join(MRI_DIR, dir_mri[it_dim+3], 'slice_' + "{:03d}".format(it_s) + '.npy'), index_i)



# ------------- #
# HISTOLOGY_DIR #
# ------------- #

# Create block-number list
MASKS_DIR = join(RIGID_RESULTS, 'masks')
BLOCK_FILES = listdir(MASKS_DIR)
BLOCK_FILES = list(filter(lambda x: '_volume.LFB.mask.reg.shift.mgz' in x, BLOCK_FILES))

if exists(join(RIGID_RESULTS, 'block_idx-name_relationship.txt')):
    dict_block = {}
    with open(join(RIGID_RESULTS, 'block_idx-name_relationship.txt'), 'r') as readFile:
        csvreader = csv.reader(readFile)
        for row in csvreader:
            print(row)
            if row[0] == 'BLOCK_NUM':
                continue
            if row[1] == '':
                continue

            if SUBJECT == 'P41-16' and row[1] == 'B5.1':
                continue
            dict_block[int(row[0])] = {'name': row[1]}
else:
    blocks = np.unique(indices_vol)
    dict_block = {n + 1: {'name': '', 'vox2ras': np.zeros((4, 4))} for n in range(len(blocks))}
    with open(join(RIGID_RESULTS, 'block_idx-name_relationship.txt'), 'w') as writeFile:
        writeFile.write('BLOCK_NUM,BLOCK_NAME\n')
        for it_b, b in enumerate(BLOCK_FILES):
            bnum = it_b + 1
            bname = b.split('_')[1]
            writeFile.write(str(bnum) + ',' + bname + '\n')
            dict_block[bnum]['name'] = bname

print('\n\n\n')
print(' --- Histology LR --- ' + str(DO_HISTOLOGY))
print('\n')
if DO_HISTOLOGY:
    d_HISTO = {}
    slicer_array = np.zeros(len(dict_block.keys()))
    for block in dict_block.keys():
        if block == 0: continue #background, no block available
        # if dict_block[block]['name'] not in flip_blocks: continue
        # if SUBJECT == 'P57-16' and dict_block[block]['name'] not in ['B1.1', 'B2.1', 'B3.1', 'B4.1', 'B5.1']: continue#['C1.1', 'C2.1', 'C3.1', 'C4.1', 'A3.1', 'P1.4', 'A1.4', 'C5.1']: continue
        # if SUBJECT == 'P41-16' and dict_block[block]['name'] not in []: continue
        # if SUBJECT == 'P58-16' and dict_block[block]['name'] not in []: continue
        # if SUBJECT == 'P85-18' and dict_block[block]['name'] not in []: continue
        # if SUBJECT == 'P85-18' and dict_block[block]['name'] not in ['B4.1']: continue
        if SUBJECT == 'EX9-19' and dict_block[block]['name'] not in ['P5.2']: continue

        if 'B' in dict_block[block]['name']:
            orientation = 'axial'
        elif 'C' in dict_block[block]['name']:
            orientation = 'sagittal'
        else:
            orientation = 'coronal'

        print(str(block) + ': ' + dict_block[block]['name'])
        BLOCK_DIR = join(HISTOLOGY_DIR, "{:02d}".format(block))
        create_results_dir(BLOCK_DIR, subdirs=dir_hist)

        if dict_block[block]['name'] == 'B5.1':
            path = join(BLOCK_DIR, 'slices_LFB')
            proxy = nib.load(join(PROCESSED_DIR, dict_block[block]['name'], 'LFB_IMAGE.nii.gz'))
            init_shape = proxy.shape

            volume = np.asarray(proxy.dataobj)
            volume = volume.astype('uint8')
            slice_i = np.squeeze(volume[..., 4, :])
            slice_i = ORIENTATION_CLASS.change_orientation(slice_i, dict_block[block]['name'], proxy.affine, vox2ras_MRI)

            blockshape = slice_i.shape[:2]
            nslices = np.max([int(a) for a in num_slice_dict[dict_block[block]['name']]])
            volume = np.zeros(blockshape + (nslices,), dtype='int')

        else:
            proxy = nib.load(join(PROCESSED_DIR, dict_block[block]['name'], 'LFB_IMAGE.nii.gz'))
            # proxy = nib.load(join(RIGID_RESULTS, 'slices', dict_block[block]['name'], 'MRI_images.nii.gz'))
            # print(proxy.affine)
            init_shape = proxy.shape
            volume = np.asarray(proxy.dataobj)
            volume = volume.astype('uint8')
            nslices = np.max([int(a) for a in num_slice_dict[dict_block[block]['name']]])
            slice_i = np.squeeze(volume[..., 4, :])
            slice_i = ORIENTATION_CLASS.change_orientation(slice_i, dict_block[block]['name'], proxy.affine, vox2ras_MRI)
            blockshape = slice_i.shape[:2]

        d_HISTO[str(block)] = {"width": int(blockshape[1]), "height": int(blockshape[0]),
                               "slices": int(nslices), "orientation": orientation}
        d_HISTO[str(block)] = {"width": int(blockshape[1]),  "height": int(blockshape[0]),
                               "slices": int(proxy.shape[2]), "orientation": orientation}

        v2r_init = np.load(join(RIGID_RESULTS, 'slices',  dict_block[block]['name'], 'vox2ras0.npy'))#nib.load(join(PROCESSED_DIR, dict_block[block]['name'], 'LFB.nii.gz'))
        vox2ras0 = ORIENTATION_CLASS.get_vox2ras0(v2r_init, dict_block[block]['name'], init_shape, v2r_ref=vox2ras_MRI, nslices=nslices)

        # MATRICES
        matrix = np.dot(np.linalg.inv(vox2ras0), vox2ras_MRI)
        matrix_inv = np.linalg.inv(matrix)


        file = join(MRI_DIR, 'matrices', 'block_' + "{:02d}".format(block) + '.txt')
        file_inv = join(BLOCK_DIR, 'matrix.txt')

        with open(file, 'w') as writeFile:
            for i in range(4):
                writeFile.write(' '.join([str(matrix[i, j]) for j in range(4)]))
                writeFile.write('\n')

        with open(file_inv, 'w') as writeFile:
            for i in range(4):
                writeFile.write(' '.join([str(matrix_inv[i, j]) for j in range(4)]))
                writeFile.write('\n')
        # # LFB
        # path = join(BLOCK_DIR, 'slices_LFB')
        # proxy = nib.load(join(PROCESSED_DIR, dict_block[block]['name'], 'LFB_IMAGE.nii.gz'))
        #
        # volume = np.asarray(proxy.dataobj)
        # volume = volume.astype('uint8')
        # nslices = volume.shape[2]
        # for it_s in range(nslices):
        #     slice_i = np.squeeze(volume[..., it_s, :])
        #     slice_i = ORIENTATION_CLASS.change_orientation(slice_i, dict_block[block]['name'], v2r_init, vox2ras_MRI)
        #     img = Image.fromarray(slice_i, mode='RGB')
        #     if dict_block[block]['name'] in flip_blocks:
        #         img.save(join(path, 'slice_' + "{:02d}".format(nslices-it_s-1) + '.jpg'), compress_level=0)
        #     else:
        #         img.save(join(path, 'slice_' + "{:02d}".format(it_s) + '.jpg'), compress_level=0)
        #
        # # HE
        # path = join(BLOCK_DIR, 'slices_HE')
        # proxy = nib.load(join(PROCESSED_DIR, dict_block[block]['name'], 'HE_IMAGE.nii.gz'))
        # volume = np.asarray(proxy.dataobj)
        # volume = volume.astype('uint8')
        # for it_s in range(nslices):
        #     slice_i = np.squeeze(volume[..., it_s, :])
        #     slice_i = ORIENTATION_CLASS.change_orientation(slice_i, dict_block[block]['name'], v2r_init, vox2ras_MRI)
        #     img = Image.fromarray(slice_i, mode='RGB')
        #     if dict_block[block]['name'] in flip_blocks:
        #         img.save(join(path, 'slice_' + "{:02d}".format(nslices - it_s - 1) + '.jpg'), compress_level=0)
        #     else:
        #         img.save(join(path, 'slice_' + "{:02d}".format(it_s) + '.jpg'), quality=90)
        #
        #
        # # MRI
        # path = join(BLOCK_DIR, 'slices_MRI')
        # if not exists(join(RIGID_RESULTS, 'slices', dict_block[block]['name'], 'MRI_images.nii.gz')):
        #     for it_s in range(nslices):
        #         slice_i = np.zeros((d_HISTO[str(block)]['height'], d_HISTO[str(block)]['width']))
        #         img = Image.fromarray(slice_i, mode='L')
        #         img.save(join(path, 'slice_' + "{:02d}".format(it_s) + '.jpg'), quality=90)
        # else:
        #     mri_list = []
        #     proxy = nib.load(join(RIGID_RESULTS, 'slices', dict_block[block]['name'], 'MRI_images.nii.gz'))
        #     volume = np.asarray(proxy.dataobj)
        #     volume = volume.astype('uint8')
        #     for it_s in range(nslices):
        #         slice_i = np.squeeze(volume[..., it_s])
        #         slice_i = ORIENTATION_CLASS.change_orientation(slice_i, dict_block[block]['name'], v2r_init, vox2ras_MRI)
        #         mri_list.append(slice_i)
        #         img = Image.fromarray(slice_i, mode='L')
        #         if dict_block[block]['name'] in flip_blocks:
        #             img.save(join(path, 'slice_' + "{:02d}".format(nslices - it_s - 1) + '.jpg'), compress_level=0)
        #         else:
        #             img.save(join(path, 'slice_' + "{:02d}".format(it_s) + '.jpg'), quality=90)
        #
        #     mr = np.stack(mri_list, axis=-1)
        #     img = nib.Nifti1Image(mr, vox2ras0)
        #     nib.save(img, join(PROCESSED_DIR, dict_block[block]['name'], 'MRI_IMAGE.nii.gz'))

        # LABELS
        path = join(BLOCK_DIR, 'slices_labels')
        path_npz = join(BLOCK_DIR, 'slices_labels_npz')
        proxy = nib.load(join(PROCESSED_DIR, dict_block[block]['name'], 'LABELS.nii.gz'))
        volume = np.asarray(proxy.dataobj)
        filedict = {}

        for it_s in num_slice_dict[dict_block[block]['name']]:
            it_s = int(it_s) - 1
            slice_i = np.ascontiguousarray(np.squeeze(volume[..., it_s]).astype('uint16'))
            slice_i = ORIENTATION_CLASS.change_orientation(slice_i, dict_block[block]['name'], v2r_init, vox2ras_MRI, order=0)
            if dict_block[block]['name'] in flip_blocks:
                np.savez_compressed(join(path_npz, 'slice_' + "{:02d}".format(nslices - it_s - 1) + '.npz'), data=slice_i)
            else:
                np.savez_compressed(join(path_npz, 'slice_' + "{:02d}".format(it_s) + '.npz'), data=slice_i)

            slice_i_out = np.zeros(slice_i.shape + (3,))
            idx = np.where((slice_i > 15000))#& (slice_i < 1900))
            if len(idx[0])>0:
                print('   --> I believe this is block has some issues with the background. Values>15000 are set to 0.')
                slice_i[idx] = 0
            slice_i_out[..., 0] = RGB_LUT[slice_i, 0]
            slice_i_out[..., 1] = RGB_LUT[slice_i, 1]
            slice_i_out[..., 2] = RGB_LUT[slice_i, 2]

            img = Image.fromarray(slice_i_out.astype('uint8'), mode='RGB')
            if dict_block[block]['name'] in flip_blocks:
                img.save(join(path, 'slice_' + "{:02d}".format(nslices - it_s - 1) + '.png'), compress_level=0)
            else:
                img.save(join(path, 'slice_' + "{:02d}".format(it_s) + '.png'), quality=90)

    # json_object = json.dumps(d_HISTO, indent=4)
    # with open(join(PARENT_DIR, 'histologyDimensionsKey.json'), "w") as outfile:
    #     outfile.write(json_object)


# ---------------- #
# HISTOLOGY_HR_DIR #
# ---------------- #
print('\n\n\n')
print(' --- Histology HR --- ' + str(DO_HISTOLOGY_HR))
print('\n')
if DO_HISTOLOGY_HR:
    d_HISTO_HR = {}
    for block in dict_block.keys():
        if block == 0: continue  # background, no block available
        # if dict_block[block]['name'] not in flip_blocks: continue
        if SUBJECT == 'P57-16' and dict_block[block]['name'] not in ['B1.1', 'B2.1', 'B3.1', 'B4.1', 'B5.1']: continue#['C1.1', 'C2.1', 'C3.1', 'C4.1', 'A3.1', 'P1.4', 'A1.4', 'C5.1']: continue
        # if SUBJECT == 'P41-16' and dict_block[block]['name'] not in []: continue
        # if SUBJECT == 'P58-16' and dict_block[block]['name'] not in []: continue
        # if SUBJECT == 'P85-18' and dict_block[block]['name'] not in ['B4.1']: continue
        if SUBJECT == 'EX9-19' and dict_block[block]['name'] not in ['P1.3']: continue #['A6.1', 'P5.2', 'A3.3', 'P6.1', 'P6.2']

        if 'B' in dict_block[block]['name']:
            orientation = 'axial'
        elif 'C' in dict_block[block]['name']:
            orientation = 'sagittal'
        else:
            orientation = 'coronal'

        print(str(block) + ': ' + dict_block[block]['name'])

        BLOCK_DIR = join(HISTOLOGY_HR_DIR, "{:02d}".format(block))
        create_results_dir(BLOCK_DIR, subdirs=dir_hist_hr)

        if dict_block[block]['name'] == 'B5.1':
            v2r_tmp = np.load(join(PROCESSED_DIR, dict_block[block]['name'], 'LFB', 'vox2ras.npy'))
            slice_tmp = np.asarray(Image.open(join(PROCESSED_DIR, dict_block[block]['name'], 'LFB', 'slice_05_' + str(downsample_factor) + 'D' +'.jpg')))
            init_shape = slice_tmp.shape[:2]
            slice_tmp = ORIENTATION_CLASS.change_orientation(slice_tmp, dict_block[block]['name'], v2r_tmp, vox2ras_MRI)

            blockshape = slice_tmp.shape[:2]
            nslices = np.max([int(a) for a in num_slice_dict[dict_block[block]['name']]])
            volume = np.zeros(blockshape + (nslices,), dtype='int')

        else:
            proxy = nib.load(join(RIGID_RESULTS, 'slices', dict_block[block]['name'], 'MRI_images.nii.gz'))
            init_shape = proxy.shape[:2]
            volume = np.asarray(proxy.dataobj)
            volume = volume.astype('uint8')

            nslices = np.max([int(a) for a in num_slice_dict[dict_block[block]['name']]])
            slice_i = np.squeeze(volume[..., 4, :])
            slice_i = ORIENTATION_CLASS.change_orientation(slice_i, dict_block[block]['name'], proxy.affine, vox2ras_MRI)
            blockshape = slice_i.shape[:2]

            init_shape = tuple([int(i * BLOCK_res / (HISTO_res * downsample_factor)) for i in init_shape])
            blockshape = tuple([int(i * BLOCK_res / (HISTO_res * downsample_factor)) for i in blockshape])

        d_HISTO_HR[str(block)] = {"width": int(blockshape[1]), "height": int(blockshape[0]),
                                  "slices": int(nslices),
                                  "orientation": orientation}

        v2r_init = np.load(join(PROCESSED_DIR, dict_block[block]['name'], 'LFB', 'vox2ras.npy'))
        vox2ras_BLOCK_HR = ORIENTATION_CLASS.get_vox2ras0(v2r_init, dict_block[block]['name'], init_shape, v2r_ref=vox2ras_MRI, nslices=nslices)

        # MATRICES
        matrix = np.dot(np.linalg.inv(vox2ras_BLOCK_HR), vox2ras_MRI)
        matrix_inv = np.linalg.inv(matrix)

        file = join(MRI_DIR, 'matrices_hr', 'block_' + "{:02d}".format(block) + '.txt')
        file_inv = join(BLOCK_DIR, 'matrix.txt')

        with open(file, 'w') as writeFile:
            for i in range(4):
                writeFile.write(' '.join([str(matrix[i, j]) for j in range(4)]))
                writeFile.write('\n')

        with open(file_inv, 'w') as writeFile:
            for i in range(4):
                writeFile.write(' '.join([str(matrix_inv[i, j]) for j in range(4)]))
                writeFile.write('\n')

        # LFB
        path = join(BLOCK_DIR, 'slices_LFB')
        init_path = join(PROCESSED_DIR, dict_block[block]['name'], 'LFB')
        files = listdir(init_path)
        files = list(filter(lambda x: 'jpg' in x, files))
        files = list(filter(lambda x: str(downsample_factor) + 'D' in x, files))
        for f in files:
            slice_num = int(f.split('slice_')[1].split('_' + str(downsample_factor) + 'D')[0])-1
            if np.mod(slice_num-1, 2)==1: continue
            im = Image.open(join(init_path, f))
            slice_i = np.array(im)
            slice_i = ORIENTATION_CLASS.change_orientation(slice_i, dict_block[block]['name'], v2r_init, vox2ras_MRI)
            img = Image.fromarray(slice_i, mode='RGB')
            img.save(join(init_path, f[:-3] + '.oriented.png'), compress_level=0)
            if dict_block[block]['name'] in flip_blocks:
                cwebp(input_image=join(init_path, f[:-3] + '.oriented.png'),
                      output_image=join(path, 'slice_' + "{:02d}".format(nslices - slice_num - 1) + '.webp'),
                      option="-q 50", logging="-v")
            else:
                cwebp(input_image=join(init_path, f[:-3] + '.oriented.png'),
                      output_image=join(path, 'slice_' + "{:02d}".format(slice_num) + '.webp'),
                      option="-q 50", logging="-v")
            remove(join(init_path, f[:-3] + '.oriented.png'))

        # HE
        path = join(BLOCK_DIR, 'slices_HE')
        init_path = join(PROCESSED_DIR, dict_block[block]['name'], 'HE')
        files = listdir(init_path)
        files = list(filter(lambda x: 'jpg' in x, files))
        files = list(filter(lambda x: str(downsample_factor) + 'D' in x, files))
        for f in files:
            slice_num = int(f.split('slice_')[1].split('_' + str(downsample_factor) + 'D')[0])-1
            im = Image.open(join(init_path, f))
            slice_i = np.array(im)
            slice_i = ORIENTATION_CLASS.change_orientation(slice_i, dict_block[block]['name'], v2r_init, vox2ras_MRI)
            img = Image.fromarray(slice_i, mode='RGB')
            img.save(join(init_path, f[:-3] + '.oriented.png'), compress_level=0)
            if dict_block[block]['name'] in flip_blocks:
                cwebp(input_image=join(init_path, f[:-3] + '.oriented.png'),
                      output_image=join(path, 'slice_' + "{:02d}".format(nslices - slice_num - 1) + '.webp'),
                      option="-q 50", logging="-v")
            else:
                cwebp(input_image=join(init_path, f[:-3] + '.oriented.png'),
                      output_image=join(path, 'slice_' + "{:02d}".format(slice_num) + '.webp'),
                      option="-q 50", logging="-v")
            remove(join(init_path, f[:-3] + '.oriented.png'))


        # MRI
        path = join(BLOCK_DIR, 'slices_MRI')
        for it_slice in num_slice_dict[dict_block[block]['name']]:
            it_s = int(it_slice) - 1
            slice_i = np.squeeze(volume[..., it_s])
            slice_i = resize(slice_i, init_shape, anti_aliasing=False)
            slice_i = ORIENTATION_CLASS.change_orientation(slice_i, dict_block[block]['name'], v2r_init, vox2ras_MRI)
            img = Image.fromarray((255*slice_i).astype('uint8'), mode='L')
            img.save(join(path, 'slice_' + "{:02d}".format(it_s) + '.jpg'), quality=90)
            if dict_block[block]['name'] in flip_blocks:
                cwebp(input_image=join(path, 'slice_' +  "{:02d}".format(it_s) + '.jpg'),
                      output_image=join(path, 'slice_' + "{:02d}".format(nslices - it_s - 1) + '.webp'),
                      option="-q 50", logging="-v")
            else:
                cwebp(input_image=join(path, 'slice_' +  "{:02d}".format(it_s) + '.jpg'),
                      output_image=join(path, 'slice_' + "{:02d}".format(it_s) + '.webp'),
                      option="-q 50", logging="-v")
            remove(join(path, 'slice_' + "{:02d}".format(it_s) + '.jpg'))

        # LABELS
        path = join(BLOCK_DIR, 'slices_labels')
        path_npz = join(BLOCK_DIR, 'slices_labels_npz')
        init_path = join(PROCESSED_DIR, dict_block[block]['name'], 'LABELS')
        files = listdir(init_path)
        files = list(filter(lambda x: 'npz' in x, files))
        files = list(filter(lambda x: str(downsample_factor) + 'D' in x, files))
        if not files: #if no labels available
            for it_slice in num_slice_dict[dict_block[block]['name']]:
                slice_i = np.zeros(blockshape, dtype = 'uint16', order='C')
                np.savez_compressed(join(path_npz, 'slice_' + "{:02d}".format(it_slice) + '.npz'), data=slice_i)

                slice_i_out = np.zeros(slice_i.shape + (3,), dtype='uint16')
                slice_i_out[..., 0] = slice_i
                slice_i_out[..., 1] = slice_i
                slice_i_out[..., 2] = slice_i

                img = Image.fromarray(slice_i_out.astype('uint8'), mode='RGB')
                img.save(join(path, 'slice_' + "{:02d}".format(it_slice) + '.png'))
                if dict_block[block]['name'] in flip_blocks:
                    cwebp(input_image=join(init_path, f[:-3] + '.oriented.png'),
                          output_image=join(path, 'slice_' + "{:02d}".format(nslices - it_slice - 1) + '.webp'),
                          option="-q 50", logging="-v")
                else:
                    cwebp(input_image=join(path, 'slice_' + "{:02d}".format(it_slice) + '.png'),
                          output_image=join(path, 'slice_' + "{:02d}".format(it_slice) + '.webp'),
                          option="-q 50", logging="-v")
                remove(join(path, 'slice_' + "{:02d}".format(it_slice) + '.png'))

        for f in files:
            slice_num = int(f.split('slice_')[1].split(str(downsample_factor) + 'D')[0])-1

            # if exists(join(path, 'slice_' + "{:02d}".format(slice_num) + '.webp')):
            #     continue
            slice_i = np.ascontiguousarray(np.load(join(init_path, f))['data'].astype('uint16'))
            slice_i = ORIENTATION_CLASS.change_orientation(slice_i, dict_block[block]['name'], v2r_init, vox2ras_MRI, order=0)
            if dict_block[block]['name'] in flip_blocks:
                np.savez_compressed(join(path_npz, 'slice_' + "{:02d}".format(nslices - slice_num - 1) + '.npz'), data=slice_i)

            else:
                np.savez_compressed(join(path_npz, 'slice_' + "{:02d}".format(slice_num) + '.npz'), data=slice_i)

            slice_i_out = np.zeros(slice_i.shape + (3,), dtype='uint8')
            idx = np.where((slice_i > 15000))  # & (slice_i < 1900)) # idx = np.where((slice_i > 843) & (slice_i < 1900))
            if len(idx[0]) > 0:
                print('   --> I believe this is block has some issues with the background. Values>15000 are set to 0.')
                slice_i[idx] = 0
            slice_i_out[..., 0] = RGB_LUT[slice_i, 0]
            slice_i_out[..., 1] = RGB_LUT[slice_i, 1]
            slice_i_out[..., 2] = RGB_LUT[slice_i, 2]

            img = Image.fromarray(slice_i_out.astype('uint8'), mode='RGB')
            img.save(join(path, 'slice_' + "{:02d}".format(slice_num) + '.png'))
            if dict_block[block]['name'] in flip_blocks:
                cwebp(input_image=join(path, 'slice_' + "{:02d}".format(slice_num) + '.png'),
                      output_image=join(path, 'slice_' + "{:02d}".format(nslices - slice_num - 1) + '.webp'),
                      option="-q 50", logging="-v")
            else:
                cwebp(input_image=join(path, 'slice_' + "{:02d}".format(slice_num) + '.png'),
                      output_image=join(path, 'slice_' + "{:02d}".format(slice_num) + '.webp'),
                      option="-q 50", logging="-v")
            remove(join(path, 'slice_' + "{:02d}".format(slice_num) + '.png'))

    # json_object = json.dumps(d_HISTO_HR, indent=4)
    # with open(join(PARENT_DIR, 'histologyHRDimensionsKey.json'), "w") as outfile:
    #     outfile.write(json_object)
