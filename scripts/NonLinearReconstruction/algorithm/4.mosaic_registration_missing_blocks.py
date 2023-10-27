import subprocess
from os.path import join, exists, basename
import os
import copy
from argparse import ArgumentParser
import pdb
import shutil

import cv2
import numpy as np
from PIL import Image
import nibabel as nib
from scipy.interpolate import interpn

from utils.io_utils import create_results_dir, write_affine_matrix
from utils.deformation_utils import interpolate2D
from utils.image_utils import padMRI
from config import config_database, config_donors
from setup_repo import *

#  *********************************  #
#
# This script is used to compute the registration of blocks with missing MR contrast (e.g., P57-16_B5.1)
# semi-automatically. We perform the following steps:
#    - Get the first NISSL image and define it as the reference image
#    - Register the remaining NISSL images to the previous one
#    - Register the first HE section to the NISSL and register the remaining to the previous one.
#    - (!) Need to manually register the block to the MRI and update the AFFINE matrix in the code.
#    - Write the following numbers in the code: (i) the block number, (ii) the intensity in MRI volume
#    - Then, the code will pad the MRI and BLOCK_NUMBER files and fill them accordingly (thanks to the manually
#      registered image from freeview
#
#  *********************************  #


arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='P57-16', choices=list(config_donors.file_dict.keys()))
arg_parser.add_argument('--bid', help="Block identifier (most probable from the brainstem)")
arg_parser.add_argument('--nslices', type=int)
arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
arg_parser.add_argument('--nn', type=int, default=2, help='Number of neighbours')
arg_parser.add_argument('--update_header', action='store_true')

arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
BLOCK = arguments.bid
NSLICES = arguments.nslices
nneighbours = arguments.nn
cost = arguments.cost
INITIAL_COMPUTE_TRUE_UPDATE_HEADER_FALSE = arguments.update_header
SUBJECT_BLOCK = SUBJECT + '_' + BLOCK

BT_DB = config_database.get_lin_dict(SUBJECT)
DATA_PATH = join(DATA_DIR, SUBJECT, 'ScannedSlides', SUBJECT_BLOCK)
OUTPUT_PATH = join(RESULTS_DIR, SUBJECT, 'RigidRegistration', 'results', 'slices', BLOCK)
REGISTRATION_PATH = join(RESULTS_DIR, SUBJECT, 'NonlinearRegistration', 'ST3', cost, 'NN' + str(nneighbours), BLOCK)

if not exists(REGISTRATION_PATH): os.makedirs(REGISTRATION_PATH)

config_file_data = config_donors.file_dict[SUBJECT]
BLOCK_res = config_file_data.BLOCK_res
HISTO_res = config_file_data.HISTO_res

subdirs = ['images', 'masks', 'labels', 'affine', 'masks_orig', 'images_orig', 'labels_orig']
create_results_dir(join(OUTPUT_PATH, 'MRI'), subdirs=subdirs[:3])
create_results_dir(join(OUTPUT_PATH, 'HE'), subdirs=subdirs)
create_results_dir(join(OUTPUT_PATH, 'LFB'), subdirs=subdirs)

scontrol = [-8, -8]
tempdir = '/tmp'

INIT_SLICES_DICT = {
    'P58-16': 2,
    'EX9-19': 0,
}
init_slice = INIT_SLICES_DICT[SUBJECT]

if not INITIAL_COMPUTE_TRUE_UPDATE_HEADER_FALSE:


    # init_affine_P58_B5 = np.array([
    #     [0.1328, -1.2429, 0, 56.3505],
    #     [1.2429, 0.1328, 0, 41.6600],
    #     [0, 0, 0.3125, 41.1759],
    #     [0, 0, 0, 1.0000]])
    #
    # init_affine_EX9_B5 = np.array([
    #     [-0.5946, 1.0995, 0, 35.8486],
    #     [-1.0995, -0.5946, 0, 113.9731],
    #     [0, 0, 0.3125, 23.7770],
    #     [0, 0, 0, 1.0000]
    #
    # ])
    # # upsample LFB block to 0.1 mm in plane
    # block_res = np.sqrt(np.sum(init_affine_P58_B5 * init_affine_P58_B5, axis=0))[:-1]
    #
    # inplane_factor = [b / BLOCK_res for b in block_res[:2]]
    # aux = np.asarray([(ipf - 1) / (2 * ipf) for ipf in inplane_factor] + [0])
    # affine = init_affine_P58_B5
    # affine[:3, 0] = affine[:3, 0] / inplane_factor[0]
    # affine[:3, 1] = affine[:3, 1] / inplane_factor[1]
    # affine[:3, 2] = affine[:3, 2]
    # affine[:3, 3] = affine[:3, 3] - np.dot(init_affine_P58_B5[:3, :3], aux.T)
    #
    # np.save(join(OUTPUT_PATH, 'vox2ras0.npy'), affine)
    # exit()
    affine = np.load(join(OUTPUT_PATH, 'vox2ras0.npy'))

    for stain in ['LFB','HE']:
        slice_list = list(np.arange(init_slice, NSLICES))
        process_slice_list = slice_list[5:] + slice_list[:5][::-1]#slice_list[-1:]#
        for slice_num in process_slice_list:
            print(slice_num)
            sid = "{:04d}".format(slice_num+1)
            sid_2str = "{:02d}".format(slice_num+1)

            filename = SUBJECT + '_' + BLOCK + '_' + stain + '_' + sid_2str
            histo_filepath = join(DATA_PATH, stain, filename + '.jpg')
            histo_mask_filepath = join(DATA_PATH, stain, filename + '.mask.png')
            H = cv2.imread(histo_filepath)
            M = cv2.imread(histo_mask_filepath, flags=0)
            if np.max(M) == 0 or M is None:
                print('slice_not_found')
                continue

            M = (M / np.max(M)) > 0.5
            H_gray = cv2.cvtColor(H, cv2.COLOR_BGR2GRAY)
            H_gray[~M] = 0
            M = (255 * M).astype('uint8')

            resized_shape = tuple([int(i * HISTO_res / BLOCK_res) for i in M.shape])
            H_gray_small = cv2.resize(H_gray, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
            M_small = cv2.resize(M, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)

            margin = [20,20]
            H_gray = np.zeros(tuple([i + 2 * m for i, m in zip(H_gray_small.shape, margin)]))
            H_gray[margin[0]: -margin[0], margin[1]: -margin[1]] = H_gray_small

            M = np.zeros(tuple([i + 2 * m for i, m in zip(M_small.shape, margin)]))
            M[margin[0]: -margin[0], margin[1]: -margin[1]] = M_small

            img = Image.fromarray(H_gray.astype('uint8'), mode='L')
            img.save(join(OUTPUT_PATH, stain, 'images_orig', 'slice_' + sid + '.png'))

            img = Image.fromarray(M.astype('uint8'), mode='L')
            img.save(join(OUTPUT_PATH, stain, 'masks_orig', 'slice_' + sid + '.png'))

            if slice_num == process_slice_list[0]:
                if stain == 'LFB':

                    img = Image.fromarray(M.astype('uint8'), mode='L')
                    img.save(join(OUTPUT_PATH, stain, 'masks', 'slice_' + sid + '.png'))

                    img = Image.fromarray(H_gray.astype('uint8'), mode='L')
                    img.save(join(OUTPUT_PATH, stain, 'images', 'slice_' + sid + '.png'))

                    matrix = np.eye(4)
                    with open(join(OUTPUT_PATH, stain, 'affine', 'slice_' + sid + '.aff'), 'w') as writeFile:
                        for i in range(4):
                            writeFile.write(' '.join([str(matrix[i, j]) for j in range(4)]))
                            writeFile.write('\n')

                else:
                    refFile = join(OUTPUT_PATH, 'LFB', 'images', 'slice_' + sid + '.png')
                    refMaskFile = join(OUTPUT_PATH, 'LFB', 'masks', 'slice_' + sid + '.png')

                    floGrayFile = join(OUTPUT_PATH, stain, 'images_orig', 'slice_' + sid + '.png')
                    floMaskFile = join(OUTPUT_PATH, stain, 'masks_orig', 'slice_' + sid + '.png')

                    outputGrayFile = join(OUTPUT_PATH, stain, 'images', 'slice_' + sid + '.png')
                    outputMaskFile = join(OUTPUT_PATH, stain, 'masks', 'slice_' + sid + '.png')

                    affineFile = join(OUTPUT_PATH, stain, 'affine', 'slice_' + sid + '.aff')

                    subprocess.call(
                        [ALADINcmd, '-ref', refFile, '-flo', floGrayFile, '-aff', affineFile, '-res', outputGrayFile,
                         '-rigOnly', '-noSym', '-ln', '4', '-lp', '3', '-pad', '0'], stdout=subprocess.DEVNULL)

                    subprocess.call(
                        [REScmd, '-ref', refFile, '-flo', floGrayFile, '-trans', affineFile, '-res', outputGrayFile,
                         '-inter', '3', '-voff'])

                    subprocess.call(
                        [REScmd, '-ref', refMaskFile, '-flo', floMaskFile, '-trans', affineFile, '-res', outputMaskFile,
                         '-inter', '0', '-voff'])

            else :

                if slice_num > process_slice_list[0]:
                    sid_pre = "{:04d}".format(slice_num)
                else:
                    sid_pre = "{:04d}".format(slice_num+2)


                refFile = join(OUTPUT_PATH, stain, 'images', 'slice_' + sid_pre + '.png')
                refMaskFile = join(OUTPUT_PATH, stain, 'masks', 'slice_' + sid_pre + '.png')

                floGrayFile = join(OUTPUT_PATH, stain, 'images_orig', 'slice_' + sid + '.png')
                floMaskFile = join(OUTPUT_PATH, stain, 'masks_orig', 'slice_' + sid + '.png')

                outputGrayFile = join(OUTPUT_PATH, stain, 'images', 'slice_' + sid + '.png')
                outputMaskFile = join(OUTPUT_PATH, stain, 'masks', 'slice_' + sid + '.png')

                affineFile = join(OUTPUT_PATH, stain, 'affine', 'slice_' + sid + '.aff')

                subprocess.call(
                    [ALADINcmd, '-ref', refFile, '-flo', floGrayFile, '-aff', affineFile, '-res', outputGrayFile,
                     '-rigOnly', '-noSym', '-ln', '4', '-lp', '3', '-pad', '0'], stdout=subprocess.DEVNULL)
                subprocess.call(
                    [REScmd, '-ref', refFile, '-flo', floGrayFile, '-trans', affineFile, '-res', outputGrayFile,
                     '-inter', '3', '-voff'])
                subprocess.call(
                    [REScmd, '-ref', refMaskFile, '-flo', floMaskFile, '-trans', affineFile, '-res', outputMaskFile,
                     '-inter', '0', '-voff'])

        directories = [join(OUTPUT_PATH, stain, 'images'), join(OUTPUT_PATH, stain, 'masks')]
        for d in directories:
            vol = []
            for slice_num in range(init_slice, NSLICES):
                sid = slice_num + 1
                im = cv2.imread(join(d, 'slice_' + "{:04d}".format(sid) + '.png'))
                if im is None:
                    print(sid)
                    continue
                if len(im.shape) > 2:
                    im = im[..., 0]

                if slice_num == init_slice:
                    vol = np.zeros(im.shape + (NSLICES,))

                vol[..., slice_num] = im

            vox2ras0_block = np.load(join(OUTPUT_PATH, 'vox2ras0.npy'))
            img = nib.Nifti1Image(vol, vox2ras0_block)
            nib.save(img, join(OUTPUT_PATH, stain + '_' + basename(d) + '.nii.gz'))

        M = cv2.imread(join(OUTPUT_PATH, 'LFB', 'masks_orig', 'slice_' + "{:04d}".format(process_slice_list[0] + 1) + '.png'))
        volume_shape = M.shape[:2] + (NSLICES,)
        output_volume = np.zeros(volume_shape)
        output_mask = np.zeros(volume_shape)
        velocity_field = np.zeros((2,) + volume_shape)
        displacement_field = np.zeros((2,) + volume_shape)
        totalfield = np.zeros((2,) + volume_shape)

        for slice_num in process_slice_list:
        # for slice_num in range(init_slice, NSLICES - 1):
            print(slice_num)
            if slice_num > process_slice_list[0]:
                sid_ref = "{:04d}".format(slice_num)
            else:
                sid_ref = "{:04d}".format(slice_num + 2)

            sid_flo = "{:04d}".format(slice_num + 1)

            if slice_num == process_slice_list[0]:
                refFile = join(OUTPUT_PATH, stain, 'images', 'slice_' + sid_flo + '.png')
                refMaskFile = join(OUTPUT_PATH, stain, 'masks', 'slice_' + sid_flo + '.png')
                floFile = join(tempdir, stain + '_images_' + sid_flo + '.png')
                floMaskFile = join(tempdir, stain + '_masks_' + sid_flo + '.png')
                output_volume[..., slice_num] = cv2.imread(refFile)[..., 0] / 254.0
                output_mask[..., slice_num] = cv2.imread(refMaskFile)[..., 0] / 254.0
                shutil.copy(refFile, floFile)
                shutil.copy(refMaskFile, floMaskFile)
                continue

            # Nonlinear registration
            refFile = join(tempdir, stain + '_images_' + sid_ref + '.png')
            refMaskFile = join(tempdir, stain + '_masks_' + sid_ref + '.png')
            floFile = join(OUTPUT_PATH, stain, 'images', 'slice_' + sid_flo + '.png')
            floMaskFile = join(OUTPUT_PATH, stain, 'masks', 'slice_' + sid_flo + '.png')
            if not exists(refFile) or not exists(floFile): continue

            outputFile = join(tempdir, stain + '_images_' + sid_flo + '.png')
            outputMaskFile = join(tempdir, stain + '_masks_' + sid_flo + '.png')
            nonlinearField = join(tempdir, 'nonlinearField.nii.gz')
            dummyFileNifti = join(tempdir, 'dummyFileNifti.nii.gz')

            subprocess.call([
                F3Dcmd, '-ref', refFile, '-flo', floFile, '-res', outputFile, '-cpp', dummyFileNifti,
                '-sx', str(scontrol[0]), '-sy', str(scontrol[1]), '-ln', '3', '-lp', '2',
                '--lncc', '5', '-pad', '0', '-vel', '-voff'], stdout=subprocess.DEVNULL)

            subprocess.call([
                TRANSFORMcmd, '-ref', refFile, '-flow', dummyFileNifti, nonlinearField], stdout=subprocess.DEVNULL)

            subprocess.call([
                REScmd, '-ref', refMaskFile, '-flo', floMaskFile, '-trans', nonlinearField, '-res', outputMaskFile,
                '-inter', '0', '-voff'], stdout=subprocess.DEVNULL)

            data = Image.open(outputFile)
            output_volume[..., slice_num] = np.array(data) / 254.0

            data = Image.open(outputMaskFile)
            output_mask[..., slice_num] = np.array(data) / 254.0

            II, JJ = np.meshgrid(np.arange(0, volume_shape[0]), np.arange(0, volume_shape[1]), indexing='ij')

            proxy = nib.load(nonlinearField)
            proxyarray = np.transpose(np.squeeze(np.asarray(proxy.dataobj)), [2, 1, 0])
            proxyarray[np.isnan(proxyarray)] = 0
            finalarray = np.zeros_like(proxyarray)
            finalarray[1] = proxyarray[0] - JJ
            finalarray[0] = proxyarray[1] - II
            velocity_field[..., slice_num] = finalarray

            nstep = 7
            fi = finalarray[0] / 2 ** nstep
            fj = finalarray[1] / 2 ** nstep
            for it_step in range(nstep):
                di = II + fi
                dj = JJ + fj
                inci = interpolate2D(fi, np.stack([di, dj], -1))
                incj = interpolate2D(fj, np.stack([di, dj], -1))
                fi = fi + inci.reshape(volume_shape[:2])
                fj = fj + incj.reshape(volume_shape[:2])

            flow = np.concatenate((fi[np.newaxis], fj[np.newaxis]))
            displacement_field[..., slice_num] = flow

            refFile = join(OUTPUT_PATH, stain, 'images', 'slice_' + "{:04d}".format(slice_num+1) + '.png')
            affineFile = join(OUTPUT_PATH, stain, 'affine', 'slice_' + "{:04d}".format(slice_num+1) + '.aff')
            dummyFileNifti = 'images_tmp2w.nii.gz'

            subprocess.call([TRANSFORMcmd, '-ref', refFile, '-disp', affineFile, dummyFileNifti], stdout=subprocess.DEVNULL)

            proxy_aff = nib.load(dummyFileNifti)
            affine_field = np.asarray(proxy_aff.dataobj)[:, :, 0, 0, :]
            affine_field = np.transpose(affine_field, axes=[2, 1, 0])

            #
            II, JJ = np.meshgrid(np.arange(0, displacement_field.shape[1]), np.arange(0, displacement_field.shape[2]), indexing='ij')
            II2 = II + displacement_field[0, ..., slice_num]
            JJ2 = JJ + displacement_field[1, ..., slice_num]
            inci = interpolate2D(affine_field[1], np.stack((II2, JJ2), axis=-1))
            incj = interpolate2D(affine_field[0], np.stack((II2, JJ2), axis=-1))
            II3 = II2 + inci
            JJ3 = JJ2 + incj
            field_i = II3 - II
            field_j = JJ3 - JJ

            totalfield[0, ..., slice_num] = field_i
            totalfield[1, ..., slice_num] = field_j


        img = nib.Nifti1Image(output_volume, affine)
        nib.save(img, join(REGISTRATION_PATH, stain + '.nii.gz'))

        img = nib.Nifti1Image(output_mask, affine)
        nib.save(img, join(REGISTRATION_PATH, stain + '.mask.nii.gz'))

        img = nib.Nifti1Image(displacement_field, affine)
        nib.save(img, join(REGISTRATION_PATH, stain + '.flow.nii.gz'))

        img = nib.Nifti1Image(totalfield, affine)
        nib.save(img, join(REGISTRATION_PATH, stain + '.totalflow.nii.gz'))

    print('REMEMBER TO PROPAGATE THE AFFINE MATRIX')

else:

# New affine computed using freeview
    new_affine_P57_B5 = np.array([
        [-0.0033, 0.0999, 0.0045, -26.1457],
        [-0.0923, -0.0028, -0.1410, -3.5116],
        [0.0383, 0.0018, -0.3094, -84.7734],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ]
    )

    new_affine_P41_B5 = np.array([
        [0.0026, 0.0999,0.0152,  -2.9537],
        [0.0590, -0.0004, -0.2592, -55.5936],
        [- 0.0807,  -0.0028, -0.2472, -21.7062],
        [0.0000, 0.0000, 0.0000,   1.0000]

    ])

    new_affine_P58_B5 = np.array([
        [0.0167, -0.0985, 0.0119, 17.0806],
        [0.0866, 0.0165, 0.1472, -55.8667],
        [-0.0470, -0.0046, 0.2754, - 50.8106],
        [0.0000, 0.0000, 0.0000, 1.0000]

    ])

    new_affine_EX9_B5 = np.array(config_donors.config_EX9_19.B5_1_vox2ras)
    # new_affine_EX9_B5 = np.array([
    #     [-0.0476, 0.0880, -0.0000,30.2300],
    #     [-0.0664, -0.0359, 0.2050, -39.0245],
    #     [0.0577, 0.0312, 0.2358, -93.5427],
    #     [0.0000, 0.0000, 0.0000, 1.0000]
    # ])

    new_affine = new_affine_EX9_B5

    write_affine_matrix(join(OUTPUT_PATH, 'vox2ras0.txt'), new_affine)
    np.save(join(OUTPUT_PATH, 'vox2ras0.npy'), new_affine)

    files = os.listdir(REGISTRATION_PATH)
    files = filter(lambda x: 'nii.gz' in x and 'lta' not in x and '~' not in x, files)
    for f in files:
        proxy = nib.load(join(REGISTRATION_PATH,f))
        data = np.asarray(proxy.dataobj)

        img = nib.Nifti1Image(data, new_affine)
        nib.save(img, join(REGISTRATION_PATH, f))


    # Project to reso space image
    indices_vol = join('/home/acasamitjana/Results/Registration/BUNGEE_Tools', SUBJECT, 'RigidRegistration/results/',
                       'BLOCK_MASK_MOSAIC.rasoriented.nii.gz')

    out_indices_vol = join('/home/acasamitjana/Results/Registration/BUNGEE_Tools', SUBJECT, 'RigidRegistration/results/',
                           'BLOCK_MASK_MOSAIC.rasoriented.B5incl.nii.gz')

    mri_vol = join('/home/acasamitjana/Data/BUNGEE_TOOLS', SUBJECT, 'MRI',
                   'averageWithReg.stripped.bfcorr.rasoriented.nii.gz')

    out_mri_vol = join('/home/acasamitjana/Data/BUNGEE_TOOLS', SUBJECT, 'MRI',
                       'averageWithReg.stripped.bfcorr.rasoriented.B5incl.nii.gz')

    mask_proxy = nib.load(join(REGISTRATION_PATH, 'LFB.mask.nii.gz'))
    images = {50: [indices_vol, out_indices_vol], 300: [mri_vol, out_mri_vol]}#{300: [mri_vol, out_mri_vol]}#
    for image_value, image_file in images.items():

        img = padMRI(nib.load(image_file[0]), margin=[1,1,40])
        nib.save(img, image_file[1])

        mri_proxy = nib.load(image_file[1])

        # Read mask
        mask = (np.asarray(mask_proxy.dataobj) > 0).astype('int')
        mri = np.asarray(mri_proxy.dataobj)

        # Define header and size
        vox2ras0 = mri_proxy.affine
        size = mri_proxy.shape

        # Resample
        RR, AA, SS = np.meshgrid(np.arange(0, size[0]), np.arange(0, size[1]), np.arange(0, size[2]), indexing='ij')
        voxMosaic = np.concatenate(
            (RR.reshape(-1, 1), AA.reshape(-1, 1), SS.reshape(-1, 1), np.ones((np.prod(size), 1))), axis=1)
        rasMosaic = np.dot(vox2ras0, voxMosaic.T)

        voxB = np.dot(np.linalg.inv(new_affine), rasMosaic)
        vR = voxB[0]
        vA = voxB[1]
        vS = voxB[2]
        ok1 = vR >= 0
        ok2 = vA >= 0
        ok3 = vS >= 0
        ok4 = vR <= mask.shape[0] - 1
        ok5 = vA <= mask.shape[1] - 1
        ok6 = vS <= mask.shape[2] - 1
        ok = ok1 & ok2 & ok3 & ok4 & ok5 & ok6

        xiR = np.reshape(vR[ok], (-1, 1))
        xiA = np.reshape(vA[ok], (-1, 1))
        xiS = np.reshape(vS[ok], (-1, 1))
        xi = np.concatenate((xiR, xiA, xiS), axis=1)

        vals = np.zeros(vR.shape[0])
        vals[ok] = interpn((np.arange(0, mask.shape[0]),
                            np.arange(0, mask.shape[1]),
                            np.arange(0, mask.shape[2])), mask, xi=xi)

        intensities = image_value * (vals.reshape(size) > 0.5).astype('uint16')
        img = nib.Nifti1Image(intensities, vox2ras0)
        nib.save(img, 'prova.nii.gz')

        mri_fi = mri
        mri_fi[mri == 0] = intensities[mri_fi == 0]
        img = nib.Nifti1Image(mri_fi, vox2ras0)
        nib.save(img, image_file[1])
