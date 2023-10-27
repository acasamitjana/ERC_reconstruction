import os
import pdb
from os.path import join, exists, isdir
import csv
import openpyxl
import copy

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_erosion, binary_dilation, disk, ball
from skimage.transform import rotate as imrotate
import cv2

def get_rotation_angle(subject, block_id):
    # if subject == 'P57-16' and block_id == 'P1.3':
    #     rotation_angle = 0

    if subject == 'P41-16' and  block_id not in ['P3.3', 'P3.2', 'A2.2', 'A1.2', 'A3.2', 'P1.2','P1.3', 'P2.2',
                                                   'P2.4', 'P4.1', 'P4.2', 'P5.1', 'P5.2', 'P6.1', 'P6.2', 'P7.1',
                                                   'P8.1', 'P9.1', 'A5.1', 'C1.1', 'C2.1', 'C3.1', 'C4.1',
                                                   'B1.1', 'B2.1', 'B3.1', 'B4.1', 'A6.1']: #,
        rotation_angle = 0

    elif subject == 'P41-16' and block_id in ['P5.1', 'P6.1', 'P2.2', 'P1.2']:
        rotation_angle = -90

    elif subject == 'P58-16' and block_id == 'P3.2':
        rotation_angle = 0

    elif subject == 'P85-18' and block_id == 'A2.2':
        rotation_angle = -90

    else:
        rotation_angle = 90

    return rotation_angle

def get_label_image(sid, sbj_block, LABELS_DIR):
    sid_2str = "{:02d}".format(sid)
    file1 = join(LABELS_DIR, sbj_block + '_' + sid_2str + '.nii.gz')

    if exists(file1):
        proxy = nib.load(file1)
    else:
        print('Non existent files: ' + file1 + ' continue.', end= ' ', flush=True)
        return None

    label_image = np.asarray(proxy.dataobj).astype(float)
    label_image = np.squeeze(label_image)
    if len(label_image.shape) > 2:
        label_image = label_image[..., 0]

    mask = (label_image > 0).astype('uint8')
    strel = disk(5)
    mask_e = binary_erosion(mask, strel)
    strel = disk(7)
    mask_d = binary_dilation(mask_e, strel)
    label_image[~mask_d] = 0

    return label_image

def preprocess_label_image(image, subject, bid, rotation_angle, orig_shape=None, resized_shape=None, order=1,
                           inverse=False, **kwargs):

    interpmethod = cv2.INTER_LINEAR if order is 1 else cv2.INTER_NEAREST

    if subject == 'P41-16' and bid in ['A3.3', 'A1.3', 'A2.3', 'P2.1', 'P2.3']:
        image = np.fliplr(image)

    # elif  subject == 'P85-18' and bid in ['A2.2']:
    #     image = imrotate(image, rotation_angle, resize=True, order=order, center=None)  # rotate

    else:
        if inverse:
            if rotation_angle != 0:
                image = imrotate(image, rotation_angle, resize=True, order=order, center=None)  # rotate
            image = np.flipud(image)
        else:
            image = np.flipud(image)
            if rotation_angle != 0:
                image = imrotate(image, rotation_angle, resize=True, order=order, center=None)  # rotate


    if orig_shape is not None and not inverse:

        # from utils.visualization import slices
        # filename = subject + '_' + bid + '_LFB_' + kwargs['sid_2str']
        # HISTO_DIR = join(kwargs['BT_DB_blocks']['SLIDES_DIR'], subject + '_' + bid)
        # histo_mask_filepath = join(HISTO_DIR, 'LFB', filename + '.mask.png')
        # orig_image = cv2.imread(histo_mask_filepath)
        # orig_image = cv2.resize(orig_image, (orig_image.shape[1]//4, orig_image.shape[0]//4), interpolation=cv2.INTER_NEAREST)
        # # orig_image = cv2.resize(orig_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # slices([orig_image>0, image>0])
        # pdb.set_trace()

        image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4), interpolation=interpmethod)

        x1 = int(np.floor((orig_shape[0] - image.shape[0]) / 2))
        x2 = int(np.ceil((orig_shape[0] - image.shape[0]) / 2))
        y1 = int(np.floor((orig_shape[1] - image.shape[1]) / 2))
        y2 = int(np.ceil((orig_shape[1] - image.shape[1]) / 2))

        label_orig = np.zeros(orig_shape)
        if subject == 'P85-18' and bid in ['P8.1', 'P3.2'] or subject == 'EX9-19' and bid in ['P1.3']:
            label_orig = image[-x2-x1:, -y1-y2:]

        elif x1 >= 0 and y1 >= 0:

            if subject == 'P57-16' and bid in ['A4.2'] and y1==0 and y2==0:
                label_orig[x1 + x2:] = image
            elif subject == 'P57-16' and bid in ['A4.2'] and x1>0 and x2>0 and y1==2 and y2==2:
                label_orig[x1+x2:, y1:-y2] = image
            elif x2 == 0 and y2 == 0:
                label_orig = image
            elif x2 == 0:
                label_orig[:, y1:-y2] = image
            elif y2 == 0:
                label_orig[x1:-x2] = image

            else:
                label_orig[x1:-x2, y1:-y2] = image

        elif x1 >= 0 and y1 < 0:
            if x2 == 0 and y2 == 0:
                label_orig[x1:,:] = image[:, -y1:]

            elif x2 == 0:
                label_orig[x1:,:] = image[:, -y1-y2:]

            elif y2 == 0:
                label_orig[x1:-x2, :] = image[:, -y1:]

            else:
                label_orig[x1:-x2, :] = image[:, -y1 - y2:]
                # if subject == 'P85-18' and bid in ['A2.2']:
                #     label_orig[x1:-x2, :] = image[:, -y1:y2]
                # else:
                #     label_orig[x1:-x2, :] = image[:, -y1-y2:]


        elif x1 < 0 and y1 >= 0:
            if x2 == 0 and y2 ==0:
                label_orig[:, y1:] = image[-x1:, :]

            elif x2 == 0:
                label_orig[:, y1:-y2] = image[-x1:, :]

            elif y2 == 0:
                label_orig[:, y1:] = image[-x1:x2, :]

            else:
                label_orig[:, y1:-y2] = image[-x1:x2, :]

        else:
            if x2 == 0 and y2 == 0:
                label_orig = image[-x1:, -y1:]

            elif x2 == 0:
                label_orig = image[-x1:, -y1:y2]

            elif y2 == 0:
                label_orig = image[-x1:x2, -y1:]

            else:
                label_orig = image[-x1:x2, -y1:y2]
                # if subject == 'P85-18' and bid in ['P3.2']:
                #     print('AS')
                #     label_orig = image[-x1-x2:, -y1:y2]
                # else:
                #     label_orig = image[-x1:x2, -y1:y2]

            # label_orig = image[-x1:x2, -y1-y2:]

        res_orig_shape = (int(label_orig.shape[0] // 4), int(label_orig.shape[1] // 4))
        image = cv2.resize(label_orig, (res_orig_shape[1], res_orig_shape[0]), interpolation=interpmethod)

        # slices([orig_image>0, image>0, (orig_image[..., 0]>0).astype('int')*(image>0).astype('int')])
        # pdb.set_trace()

    if resized_shape is None:
        return image
    else:
        return cv2.resize(image, (resized_shape[1], resized_shape[0]), interpolation=interpmethod)

def postprocess_label_image(labels_volume, subject, bid, missing_slices):

    if subject == 'P57-16' and bid == 'A2.4':
        for it_slice_2 in missing_slices:
            labels_volume[..., it_slice_2] = labels_volume[..., it_slice_2+1]

    elif subject == 'P57-16'  and bid == 'A3.2':
        for it_slice_2 in range(1, labels_volume.shape[-1], 2):
            labels_volume[..., it_slice_2] = labels_volume[..., it_slice_2-1]

    # elif subject == 'P57-16'  and bid == 'P2.2':
    #     for it_slice_2 in [19, 21, 23, 25, 29, 31]:
    #         labels_volume[..., it_slice_2] = labels_volume[..., it_slice_2-1]

    return labels_volume.astype(np.uint16)


def read_slice_info(SLICES_DIR):
    mapping_file = join(SLICES_DIR, 'mapping.csv')
    mapping_dict = {}
    with open(mapping_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for it_row, row in enumerate(csvreader):
            mapping_dict[row['block_id']] = [float(row['rotation']), bool(int(row['lrflip'])), bool(int(row['zflip']))]

    slice_file = join(SLICES_DIR, 'slice_id.txt')
    slice_dict = {}
    with open(slice_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for it_row, row in enumerate(csvreader):
            if row['BLOCK_ID'] not in slice_dict.keys():
                slice_dict[row['BLOCK_ID']] = []

            slice_dict[row['BLOCK_ID']].append(row['SLICE_ID'])

    return slice_dict, mapping_dict

def load_ontology(label_file, unique_labels=None):
    labels_list = []
    h2_labels_dict = {}
    h1_labels_dict = {}

    # Read label mapping from Nellie and Juri: nested dictionaries with different ontologies form the top to the bottom.
    wb = openpyxl.load_workbook(label_file)
    ws = wb.active
    is_title = True
    max_label = 0
    for row in ws.iter_rows(values_only=True):
        if is_title:
            is_title = False
            continue

        if row[0] is not None:
            fs_label = row[0]
            fs_name = row[1]
            h1_labels_dict[fs_label] = {'name': fs_name, 'allen_labels': {}}

        if row[2] is not None:
            allen_label = row[2]
            if unique_labels is not None:
                if allen_label not in unique_labels:
                    continue

            allen_name = row[3]
            if allen_label > max_label:
                max_label = allen_label
            labels_list.append({'h2_name': allen_name, 'h1_name': fs_name, 'h2_num': allen_label, 'h1_num': fs_label})
            h2_labels_dict[allen_label] = {'name': allen_name}
            h1_labels_dict[fs_label]['allen_labels'][allen_label] = allen_name

    h1_labels_dict[24] = {'name': 'CSF', 'allen_labels': {20001: 'CSF'}}
    h1_labels_dict[165] = {'name': 'Skull', 'allen_labels': {20002: 'Skull'}}
    h1_labels_dict[258] = {'name': 'Head-ExtraCerebral', 'allen_labels': {20003: 'Head-ExtraCerebral'}}
    h1_labels_dict[259] = {'name': 'SkullApprox', 'allen_labels': {20004: 'SkullApprox'}}

    h2_labels_dict[20001] = {'name': 'CSF'}
    h2_labels_dict[20002] = {'name': 'Skull'}
    h2_labels_dict[20003] = {'name': 'Head-ExtraCerebral'}
    h2_labels_dict[20004] = {'name': 'SkullApprox'}

    return labels_list, [h1_labels_dict, h2_labels_dict]

def one_hot_encoding_gaussian(target, mri_mask_bool, missing_mask, label, sigma):
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

    print(label, end=' ', flush=True)
    idx = np.where(target == label)
    onehot_target = np.zeros(target.shape)
    onehot_target[idx] = 1

    featuremap = np.zeros(mri_mask_bool.shape)
    featuremap[mri_mask_bool] = onehot_target
    featuremap = gaussian_filter(featuremap.astype(np.double), sigma=sigma)
    featuremap = featuremap.astype(float)
    featuremap[~mri_mask_bool] = 0

    return featuremap[mri_mask_bool][missing_mask]

def get_memory_used():
    import sys
    local_vars = list(locals().items())
    for var, obj in local_vars: print(var, sys.getsizeof(obj) / 1000000000)

def mkdir(directory):
    if not os.path.exists(directory): os.makedirs(directory)

def read_lta_matrix(lta_file):
    affine_matrix = np.zeros((4,4))
    with open(lta_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        it_row = 0
        for row in csvreader:
            if it_row > 8:
                break

            elif it_row >= 5:
                affine_matrix[it_row-5] = [float(i) for i in row[0].split(' ')[:4]]

            it_row += 1

    return affine_matrix


def change_block_v2r():
    DATA_DIR = '/home/acasamitjana/Results/Registration/BUNGEE_Tools/P57-16/SP3_1/Algorithm/ST3_RegNet/l1/NN4/C5.1'
    files = os.listdir(DATA_DIR)
    vox2ras0 = nib.load(  '/home/acasamitjana/Results/Registration/BUNGEE_Tools/P57-16/RigidRegistration/results/slices/C5.1/MRI_images.nii.gz').affine
    for f in files:
        if isdir(join(DATA_DIR, f)):
            HISTO_res = 3.9688e-3  # = 25.4/6400
            BLOCK_res = 0.1
            downsample_factor = 2
            init_shape = (535, 773)

            resize_factor = BLOCK_res / (HISTO_res * downsample_factor)
            resized_shape = tuple([int(np.round(a * resize_factor)) for a in init_shape])
            resize_factor = [a / b for a, b in zip(resized_shape, init_shape)]

            aux = np.asarray([(ipf - 1) / (2 * ipf) for ipf in resize_factor] + [0])
            v2r_hr = copy.copy(vox2ras0)
            v2r_hr[:3, 0] = v2r_hr[:3, 0] / resize_factor[0]
            v2r_hr[:3, 1] = v2r_hr[:3, 1] / resize_factor[1]
            v2r_hr[:3, 3] = v2r_hr[:3, 3] - np.dot(vox2ras0[:3, :3], aux.T)
            np.save(join(DATA_DIR, f, 'vox2ras.npy'), v2r_hr)

        else:
            proxy = nib.load(join(DATA_DIR, f))
            data = np.asarray(proxy.dataobj)
            img = nib.Nifti1Image(data, vox2ras0)
            nib.save(img, join(DATA_DIR, f))

