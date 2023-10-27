# py
import pdb
import os
from os.path import join, exists, basename
import subprocess
from argparse import ArgumentParser
import shutil

# libraries imports
import nibabel as nib
import numpy as np
from PIL import Image
from scipy import io
from skimage.morphology import binary_erosion
from skimage.transform import rotate as imrotate, resize
from skimage.exposure import match_histograms
import cv2
from joblib import delayed, Parallel

# project imports
from utils.io_utils import create_results_dir, write_affine_matrix
from utils.image_utils import padBlock
from dataset import read_slice_info
from config import config_dev, config_database, config_donors

#  *********************************  #
#
# This script generates the slices to be registered (original and linearly registered using reg_aladin) at the desired
# resolution. It needs:
#     - slice_id.txt files with BLOCK_ID, SLICE_ID, SLICE_ID_MM, STRUCTURES, FLIP_MRI header
#     - virtual MRI blocks.
#     - rotation and flip files
#     - ProcessScannedSlides.m (with masks computed either from Neural Networks or manual labels!)
#
#  *********************************  #
print('')
print('-----------------------')
print('--- Generate Slices ---')
print('-----------------------')
print('')

def generate_orig_slices(bid):

    # Directories
    BLOCK_DIR = join(SLICES_DIR, bid)
    HISTO_DIR = join(BT_DB['SLIDES_DIR'], BT_DB['SUBJECT'] + '_' + bid)
    # if exists(join(BLOCK_DIR, 'MRI_images.nii.gz')):
    #     print('BLOCK ' + bid + ' DONE.')
    #     return

    # MRI processing
    proxy = nib.load(join(REG_BLOCKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.mri.reg.mgz'))
    # proxy = padBlock(proxy, [[100, 1], [100, 1], [0, 0]])
    mri_vol = np.asarray(proxy.dataobj)
    vox2ras0_block = proxy.affine

    proxy = nib.load(join(REG_BLOCKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.mri.mask.reg.closing.mgz'))
    # proxy = padBlock(proxy, [[100, 1], [100, 1], [0, 0]])
    mri_mask = np.asarray(proxy.dataobj)

    proxy = nib.load(join(REG_BLOCKS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_volume.aparc.reg.mgz'))
    # proxy = padBlock(proxy, [[100, 1], [100, 1], [0, 0]])
    mri_labels = np.asarray(proxy.dataobj)

    mri_mask_bool = mri_mask > 0.5

    ## If we run out of mask for whatever reason, we copy from previous slices
    ## Not the prettiest, but it works. TODO: improve?
    aux = np.squeeze(np.sum(np.sum(mri_mask, axis=0), axis=0))
    ok = np.where(aux > 5000)[0]
    nok = np.where(aux < 5000)[0]
    for j in nok:
        dist = np.abs(j - ok)

        idx = np.argmin(dist)
        ref = ok[idx]
        mri_mask[..., j] = mri_mask[..., ref]

    ## Stretch the histogram
    aux = mri_vol[mri_mask_bool]
    mini = np.quantile(aux, 0.001)
    maxi = np.quantile(aux, 0.999)

    mri_vol = np.clip(mri_mask * (mri_vol - mini) / (maxi - mini), 0, 1)
    mri_vol = np.double(mri_vol)

    mri_labels = mri_labels * mri_mask
    mri_labels = np.double(mri_labels)

    # Update vox2ras0
    angle = -mapping_dict[bid][0]     # MRI->to histo, while the angle was computed the other way around.
    if bid == 'B2.1' and 'P57-16' in SUBJECT:
        angle += 35

    m_trans = np.eye(4)
    m_trans_inv = np.eye(4)
    m_trans[0, -1] = -mri_vol.shape[0] // 2
    m_trans[1, -1] = -mri_vol.shape[1] // 2
    m_trans_inv[0, -1] = mri_vol.shape[0] // 2
    m_trans_inv[1, -1] = mri_vol.shape[1] // 2
    m_compensate = np.eye(4)
    if angle in [90, -270]:
        m_compensate[0, -1] = mri_vol.shape[1] // 2 - mri_vol.shape[0] // 2
        m_compensate[1, -1] = mri_vol.shape[0] // 2 - mri_vol.shape[1] // 2

    if angle in [-90, 270]:
        m_compensate[0, -1] = mri_vol.shape[0] // 2 - mri_vol.shape[1] // 2
        m_compensate[1, -1] = mri_vol.shape[1] // 2 - mri_vol.shape[0] // 2

    if bid == 'B2.1' and 'P57-16' in SUBJECT:
        m_compensate[0, -1] = 487 // 2 - mri_vol.shape[0] // 2
        m_compensate[1, -1] = 485 // 2 - mri_vol.shape[1] // 2

    m_rot = np.asarray([[np.cos(np.pi * angle / 180), np.sin(np.pi * angle / 180), 0, 0],
                        [-np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    m_flip = np.eye(4)
    if mapping_dict[bid][1]:
        m_flip[1, 1] = -1

    if bid == 'B4.1' and 'P57-16' in SUBJECT:
        m_flip_ud = np.eye(4)
        if mapping_dict[bid][1]:
            m_flip_ud[0, 0] = -1

        m_flip = m_flip_ud @ m_flip


    m = m_compensate @ m_trans_inv @ m_flip @ m_rot @ m_trans

    if mapping_dict[bid][2]:
        m_zflip = np.eye(4)
        m_zshift = np.eye(4)
        # if 'C' in bid:
        #     m_zflip[0, 0] = -1
        #     m_zshift[2, 3] = mri_vol.shape[0]
        # elif 'B' in bid:
        #     m_zflip[0, 0] = -1
        #     m_zshift[1, 3] = mri_vol.shape[1]
        # else:
        m_zflip[2, 2] = -1
        m_zshift[2, 3] = mri_vol.shape[2] -1

        m = m_zshift @ m_zflip @ m

    vox2ras0_block = np.dot(vox2ras0_block, m)
    if bid == 'C4.1' and SUBJECT == 'P41-16':
        vox2ras0_block[0, -1] = 31.7310
        vox2ras0_block[1, -1] = -14.3608
        vox2ras0_block[2, -1] = -30.9241

    if mapping_dict[bid][1]:
        mri_vol = np.fliplr(mri_vol)  # flip
        mri_mask = np.fliplr(mri_mask)  # flip
        mri_labels = np.fliplr(mri_labels)  # flip

    if bid == 'B4.1' and 'P57-16' in SUBJECT:
        mri_vol = np.flipud(mri_vol)  # flip
        mri_mask = np.flipud(mri_mask)  # flip
        mri_labels = np.flipud(mri_labels)  # flip

    mri_vol_list = []
    mri_mask_list = []
    mri_labels_list = []
    mri_mask = mri_mask.astype('float')
    for it_z in range(mri_vol.shape[2]):
        mri_vol_list.append(imrotate(mri_vol[..., it_z], angle, resize=True))  # rotate
        mri_mask_list.append(imrotate(mri_mask[..., it_z], angle, resize=True))  # rotate
        mri_labels_list.append(imrotate(mri_labels[..., it_z], angle, resize=True))  # rotate

    mri_vol = np.transpose(np.asarray(mri_vol_list), [1, 2, 0])
    mri_mask = np.transpose(np.asarray(mri_mask_list).astype('uint8'), [1, 2, 0])
    mri_labels = np.transpose(np.asarray(mri_labels_list), [1, 2, 0])

    if mapping_dict[bid][2]:
        mri_vol = mri_vol[..., ::-1]  # change order of the most posterior and most lateral parts of the cerebrum/cerebellum
        mri_mask = mri_mask[..., ::-1]  # idem
        mri_labels = mri_labels[..., ::-1]  # idem

    if 'B' in bid and 'P57-16' in SUBJECT:
        minC = 1000000
        maxC = 0
        minR = 1000000
        maxR = 0
        for it_sl in range(mri_mask.shape[-1]):
            idx = np.where(mri_mask[..., it_sl] > 0)
            minC = min(minC, np.min(idx[0]))
            minR = min(minR, np.min(idx[1]))
            maxC = max(maxC, np.max(idx[0]))
            maxR = max(maxR, np.max(idx[1]))

        minC = max(0, minC - 30)
        minR = max(0, minR - 30)
        maxC = min(mri_mask.shape[0], maxC + 30)
        maxR = min(mri_mask.shape[1], maxR + 30)
        mri_vol = mri_vol[minC:maxC, minR:maxR]
        mri_mask = mri_mask[minC:maxC, minR:maxR]
        mri_labels = mri_labels[minC:maxC, minR:maxR]

        tx = np.eye(4)
        tx[0, -1] = minC
        tx[1, -1] = minR

        vox2ras0_block = np.dot(vox2ras0_block, tx)

    if bid=='B1.1' in bid and SUBJECT=='P57-16':
        mri_vol = np.flipud(mri_vol)  # flip
        mri_mask = np.flipud(mri_mask)  # flip
        mri_labels = np.flipud(mri_labels)  # flip

        m_flip_ud = np.eye(4)
        if mapping_dict[bid][1]:
            m_flip_ud[0, 0] = -1

        vox2ras0_block = np.dot(vox2ras0_block, m_flip_ud)


    if bid == 'C4.1' and SUBJECT == 'P58-16':
        vox2ras0_block = config_data_sbj.C4_1_vox2ras
    elif bid == 'C5.1' and SUBJECT == 'P85-16':
        vox2ras0_block = config_data_sbj.C5_1_vox2ras
    elif bid == 'C5.1' and SUBJECT == 'EX9-19':
        vox2ras0_block = config_data_sbj.C5_1_vox2ras
    elif bid == 'B5.1' and SUBJECT == 'EX9-19':
        vox2ras0_block = np.asarray(config_data_sbj.B5_1_vox2ras)
    elif bid == 'B4.1' and SUBJECT == 'EX9-19':
        vox2ras0_block = np.asarray(config_data_sbj.B4_1_vox2ras)
    elif (bid == 'P9.1' and SUBJECT in ['P58-16']) or (bid == 'P8.1' and SUBJECT in ['P41-16', 'P85-18']):
        mfliplr = np.eye(4)
        mfliplr[0, 0] = -1
        m[0, -1] = mri_vol.shape[0] -1
        vox2ras0_block = vox2ras0_block @ mfliplr

    if  not exists(join(BLOCK_DIR, 'vox2ras0.npy')):
        write_affine_matrix(join(BLOCK_DIR, 'vox2ras0.txt'), vox2ras0_block)
        np.save(join(BLOCK_DIR, 'vox2ras0.npy'), vox2ras0_block)

    for stain in ['HE']:#, 'LFB']:

    #     if exists(join(BLOCK_DIR, 'MRI_masks.nii.gz')):
    #         print('Block ' + bid + ' (' + stain + ') done.')
    #         continue

        slice_list = slice_dict[bid]
        slice_list.sort(key=lambda x: int(x))
        process_slice_list = slice_list[len(slice_list) // 2:] + slice_list[:len(slice_list) // 2][::-1]#slice_list[-1:]#
        for it_sid, sid in enumerate(process_slice_list):
            print('#' + sid, end=' ', flush=True)

            slice_num = int(sid)

            sid_4str = "{:04d}".format(slice_num)
            sid_2str = "{:02d}".format(slice_num)
            it_z = int(sid) - 1

            # if exists(join(BLOCK_DIR, stain, 'images_orig', 'slice_' + sid + '.png')):
            #    continue

            mri_slice = mri_vol[..., it_z]
            mri_mask_slice = mri_mask[..., it_z]
            mri_mask_slice[mri_slice == 0] = 0
            mri_labels_slice = mri_labels[..., it_z]

            if stain == 'LFB':

                img = Image.fromarray((255 * mri_slice).astype(np.uint8), mode='L')
                img.save(join(BLOCK_DIR, 'MRI', 'images', 'slice_' + sid_4str + '.png'))

                img = Image.fromarray((255 * mri_mask_slice).astype(np.uint8), mode='L')
                img.save(join(BLOCK_DIR, 'MRI', 'masks', 'slice_' + sid_4str + '.png'))

                img = Image.fromarray((255 * mri_labels_slice).astype(np.uint8), mode='L')
                img.save(join(BLOCK_DIR, 'MRI', 'labels', 'slice_' + sid_4str + '.png'))


            filename = BT_DB['SUBJECT'] + '_' + bid + '_' + stain + '_' + sid_2str
            histo_filepath = join(HISTO_DIR, stain, filename + '.jpg')
            histo_mask_filepath = join(HISTO_DIR, stain, filename + '.mask.png')

            H = cv2.imread(histo_filepath)
            M = cv2.imread(histo_mask_filepath, flags=0)
            if H is None: H = np.zeros_like(mri_slice)
            if M is None: M = np.zeros(H.shape[:2])
            if np.max(M) == 0:
                img = Image.fromarray(M.astype('uint8'), mode='L')
                img.save(join(BLOCK_DIR, stain, 'masks_orig', 'slice_' + sid_4str + '.png'))

                img = Image.fromarray(M.astype('uint8'), mode='L')
                img.save(join(BLOCK_DIR, stain, 'images_orig', 'slice_' + sid_4str + '.png'))

                continue

            M = (M / np.max(M)) > 0.5
            H_gray = cv2.cvtColor(H, cv2.COLOR_BGR2GRAY)
            H_gray[~M] = 0
            M = (255 * M).astype('uint8')
            del H

            resized_shape = tuple([int(i * config_data_sbj.HISTO_res / BLOCK_res) for i in M.shape])
            H_gray = resize(H_gray, resized_shape, order=1, anti_aliasing=True)  # bilinear
            M = resize(M, resized_shape, order=1, anti_aliasing=True)  # bilinear

            if stain == 'LFB' and np.sum(M) > 0 and np.sum(mri_mask_slice) > 0 and not (bid in ['C2.1', 'C4.1'] and SUBJECT == 'P41-16') :
                mri_mask_slice = binary_erosion(mri_mask_slice > 0.5, np.ones((5,5)))
                H_gray[M > 0.5] = match_histograms(H_gray[M > 0.5], mri_slice[mri_mask_slice > 0.5])  # Histogram matching

            img = Image.fromarray((255 * M).astype('uint8'), mode='L')
            img.save(join(BLOCK_DIR, stain, 'masks_orig', 'slice_' + sid_4str + '.png'))

            img = Image.fromarray((255 * H_gray).astype('uint8'), mode='L')
            img.save(join(BLOCK_DIR, stain, 'images_orig', 'slice_' + sid_4str + '.png'))

        print('Block ' + bid + ' (' + stain + ') done.')

    directories = [join(BLOCK_DIR, 'MRI', 'images'), join(BLOCK_DIR, 'MRI', 'masks')]
    vox2ras0_block = np.load(join(BLOCK_DIR, 'vox2ras0.npy'))
    for d in directories:
        # if exists(join(BLOCK_DIR, 'MRI_' + basename(d) + '.nii.gz')):
        #     continue
        print(d)
        files = os.listdir(d)
        processed_files = [int(f.split('_')[1].split('.')[0]) for f in files]
        vol = []
        for it_slice_num, slice_num in enumerate(processed_files):
            im = cv2.imread(join(d, 'slice_' + "{:04d}".format(slice_num) + '.png'))[..., 0]
            if it_slice_num == 0:
                vol = np.zeros(im.shape + (np.max([int(ps) for ps in processed_files]),))

            vol[..., slice_num - 1] = im

        img = nib.Nifti1Image(vol, vox2ras0_block)
        nib.save(img, join(BLOCK_DIR, 'MRI_' + basename(d) + '.nii.gz'))


def linear_alignment(bid):

    # Directories
    BLOCK_DIR = join(SLICES_DIR, bid)
    LABELS_DIR = join(BT_DB['HISTO_LABELS_DIR'], bid)

    for stain in ['LFB']:#, 'HE']:

        # if exists(join(BLOCK_DIR, stain + '_masks.nii.gz')):
        #     print('Block ' + bid + ' (' + stain + ') done.')
        #     continue
        slice_list = slice_dict[bid]
        slice_list.sort(key=lambda x: int(x))
        process_slice_list = slice_list[len(slice_list) // 2:] + slice_list[:len(slice_list) // 2][::-1]#slice_list[-1:]#
        for it_sid, sid in enumerate(process_slice_list):

            slice_num = int(sid)
            print('#' + sid, end=' ', flush=True)

            sid_4str = "{:04d}".format(slice_num)
            sid_2str = "{:02d}".format(slice_num)
            # if exists(join(BLOCK_DIR, stain, 'affine', 'slice_' + sid + '.aff')):
            #     print('Block ' + bid + ' (' + stain + ',' + sid + ') done.')
            #     continue

            # refFile = join(BLOCK_DIR, stain, 'images', 'slice_' + "{:04d}".format(slice_num - 1) + '.png')
            # refMaskFile = join(BLOCK_DIR, stain, 'masks', 'slice_' + "{:04d}".format(slice_num - 1) + '.png')
            refFile = join(BLOCK_DIR, 'MRI', 'images', 'slice_' + sid_4str + '.png')
            refMaskFile = join(BLOCK_DIR, 'MRI', 'masks', 'slice_' + sid_4str + '.png')

            floGrayFile = join(BLOCK_DIR, stain, 'images_orig', 'slice_' + sid_4str + '.png')
            floMaskFile = join(BLOCK_DIR, stain, 'masks_orig', 'slice_' + sid_4str + '.png')

            outputGrayFile = join(BLOCK_DIR, stain, 'images', 'slice_' + sid_4str + '.png')
            outputMaskFile = join(BLOCK_DIR, stain, 'masks', 'slice_' + sid_4str + '.png')
            affineFile = join(BLOCK_DIR, stain, 'affine', 'slice_' + sid_4str + '.aff')

            # If not enough matching blocks (i.e. at the end of the cerebellum/brain stem), register to the neighbour
            init_reg_aladin = ['-ref', refFile, '-flo', floGrayFile, '-res', outputGrayFile, '-aff', affineFile]
            reg_aladin_params = ['-ln', '3', '-lp', '3', '-pad', '0', '-maxit', '20', '-voff', '-omp', '2']

            try:
                extra_reg_aladin_params = []

                if 'B' in bid:
                    pass#extra_reg_aladin_params.extend(['-rigOnly'])  # , '-fmask', floMaskFile, '-cog'])
                elif exists(join(LABELS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_' + sid_2str + '.nii.gz')):
                    extra_reg_aladin_params.extend(['-rmask', refMaskFile])#, '-fmask', floMaskFile, '-cog'])
                elif sid in [slice_list[0], slice_list[1]]:
                    refFile = join(BLOCK_DIR, stain, 'images', 'slice_' + "{:04d}".format(slice_num - 1) + '.png')
                    init_reg_aladin = ['-ref', refFile, '-flo', floGrayFile, '-res', outputGrayFile, '-aff', affineFile]

                subprocess.check_output([config_dev.ALADINcmd] + init_reg_aladin + reg_aladin_params + extra_reg_aladin_params)

                # extra_reg_aladin_params.extend(['-rmask', refMaskFile])#, '-fmask', floMaskFile, '-cog'])
                # subprocess.check_output( [config_dev.ALADINcmd] + init_reg_aladin + reg_aladin_params + extra_reg_aladin_params)

            except subprocess.CalledProcessError as e:

                floarray = cv2.imread(floGrayFile)
                if np.sum(floarray) == 0:
                    refarray = cv2.imread(refFile)
                    floarray = np.zeros(refarray.shape[:-1])

                    img = Image.fromarray(floarray.astype('uint8'), mode='L')
                    img.save(outputGrayFile)

                    img = Image.fromarray(floarray.astype('uint8'), mode='L')
                    img.save(outputMaskFile)

                    continue

                refFile = join(BLOCK_DIR, stain, 'images', 'slice_' + "{:04d}".format(slice_num-1) + '.png')
                init_reg_aladin = ['-ref', refFile, '-flo', floGrayFile, '-res', outputGrayFile, '-aff', affineFile]
                extra_reg_aladin_params = []

                if 'B' in bid:
                    pass
                elif exists(join(LABELS_DIR, BT_DB['SUBJECT'] + '_' + bid + '_' + sid_2str + '.nii.gz')):
                    pass#extra_reg_aladin_params.extend(['-rmask', refMaskFile, '-fmask', floMaskFile, '-cog'])

                subprocess.call([config_dev.ALADINcmd] + init_reg_aladin + reg_aladin_params + extra_reg_aladin_params)

            subprocess.call([config_dev.REScmd, '-ref', refFile, '-flo', floGrayFile, '-trans', affineFile, '-res', outputGrayFile,   '-inter', '3', '-voff' ])
            subprocess.call([config_dev.REScmd, '-ref', refFile, '-flo', floMaskFile, '-trans', affineFile, '-res', outputMaskFile, '-inter', '0', '-voff' ])

        directories = [join(BLOCK_DIR, stain, 'images'), join(BLOCK_DIR, stain, 'masks')]
        for d in directories:
            vol = []
            for it_sid, sid in enumerate(process_slice_list):
                slice_num = int(sid)

                im = cv2.imread(join(d, 'slice_' + "{:04d}".format(slice_num) + '.png'))
                if im is None:
                    print(slice_num)
                    continue
                if len(im.shape) > 2:
                    im = im[..., 0]

                if it_sid == 0:
                    vol = np.zeros(im.shape + (np.max([int(ps) for ps in process_slice_list]),))

                vol[..., slice_num - 1] = im

            vox2ras0_block = np.load(join(BLOCK_DIR, 'vox2ras0.npy'))
            img = nib.Nifti1Image(vol, vox2ras0_block)
            nib.save(img, join(BLOCK_DIR, stain + '_' + basename(d) + '.nii.gz'))

        print('Block ' + bid + ' (' + stain + ') done.')


arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='P57-16', choices=['P57-16',  'P41-16', 'P58-16', 'P85-18', 'EX9-19'])
arg_parser.add_argument('--block_res', default=0.1, type=float)
arg_parser.add_argument('--bid', default=None, nargs='+')

arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
BLOCK_res = arguments.block_res
initial_bid_list = arguments.bid

# Config
BT_DB = config_database.get_lin_dict(SUBJECT)
config_data_sbj = config_donors.file_dict[SUBJECT]

# Directories
BASE_DIR = join(config_data_sbj.BASE_DIR, 'RigidRegistration')
RESULTS_DIR = join(BASE_DIR, 'results')
REG_BLOCKS_DIR = join(RESULTS_DIR, 'reg_blocks')
SLICES_DIR = join(RESULTS_DIR, 'slices')
OLD_SLICES_DIR = join(config_data_sbj.BASE_DIR, 'RigidRegistration_bo', 'results', 'slices')

bid_list = initial_bid_list if initial_bid_list is not None else config_data_sbj.initial_bid_list

subdirs = ['images', 'masks', 'labels', 'affine', 'masks_orig', 'images_orig', 'labels_orig']
for it_bid, bid in enumerate(bid_list):
    BLOCK_DIR = join(SLICES_DIR, bid)
    create_results_dir(join(BLOCK_DIR, 'MRI'), subdirs=subdirs[:3])
    create_results_dir(join(BLOCK_DIR, 'HE'), subdirs=subdirs)
    create_results_dir(join(BLOCK_DIR, 'LFB'), subdirs=subdirs)

# Slice info
slice_dict, mapping_dict = read_slice_info(SLICES_DIR)

# num_cores = 4
# results = Parallel(n_jobs=num_cores)(delayed(generate_orig_slices)(bid) for bid in bid_list)

for bid in bid_list:
    print(bid)
    # generate_orig_slices(bid)
    linear_alignment(bid)
