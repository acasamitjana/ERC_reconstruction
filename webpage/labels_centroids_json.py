import os
import openpyxl
import json
from argparse import ArgumentParser
import pdb

import nibabel as nib
import numpy as np
import csv
import openpyxl

from setup_repo import *
from utils.webpage_utils import DICT_CLASSES

def centeroidnp(arr_x, arr_y):
    length = arr_x.shape[0]
    sum_x = np.sum(arr_x)
    sum_y = np.sum(arr_y)
    return sum_x/length, sum_y/length

# ----------- #
# Read labels
# Read all block labels
# Iterate over all labels
#    Iterate ovel all blocks
#       Find the block with the highest number of pixels belonging to that label
#       Find the slice with highest number of pixels belonging to that label
#       Compute the centroid of the label or the closest pixel to the centroid.
# ----------- #

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='P57-16', choices=['P57-16', 'P58-16', 'P41-16', 'P85-18', 'EX9-19'])
arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
arg_parser.add_argument('--nn', type=int, default=2, help='Number of neighbours')

arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
nneighbours = arguments.nn
cost = arguments.cost

RIGID_RESULTS = os.path.join(RESULTS_DIR, SUBJECT, 'RigidRegistration', 'results')
REGISTER_DIR = os.path.join(RESULTS_DIR,  SUBJECT, 'NonlinearRegistration', 'ST3', cost, 'NN' + str(nneighbours))
PROCESSED_DIR = join(RESULTS_DIR, SUBJECT, 'NonlinearRegistration', 'ST3', cost, 'NN' + str(nneighbours))
PARENT_DIR = join(WEBPAGE_DIR, 'BrainAtlas-' + SUBJECT, SUBJECT)
MRI_DIR = join(PARENT_DIR, 'mri_rotated')
ORIENTATION_CLASS = DICT_CLASSES[SUBJECT]()

# MRI for re-orient slices
proxy = nib.load(join(MRI_DIR, 'mri.nii.gz'))
vox2ras_MRI = proxy.affine

proxy = nib.load(join(MRI_DIR, 'indices.nii.gz'))
index_vol = np.asarray(proxy.dataobj)

# file with the ontology
label_file = join(DATA_DIR, 'Documents', 'FS_label_mapping.xlsx')

# dir to load labels from
data_dir = os.path.join(WEBPAGE_DIR, 'BrainAtlas-' + SUBJECT, SUBJECT, 'histology')

# output file
table_file = os.path.join(WEBPAGE_DIR, 'BrainAtlas-' + SUBJECT, SUBJECT, 'image_ontology_hierarchical.json')

# dictionaries used
labels_dict = {}
fs_labels_dict = {}
block_dict = {}

# Read the XLSX file from Nellie and Juri
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
        fs_labels_dict[fs_label] = {'name': fs_name, 'allen_labels': {}}

    if row[2] is not None:
        allen_label = row[2]
        allen_name = row[3]
        if allen_label > max_label:
            max_label = allen_label
        labels_dict[allen_label] = {'Hierarchy #1': fs_name, 'Hierarchy #2': allen_name, 'H1_label': fs_label, 'H2_label': allen_label}
        fs_labels_dict[fs_label]['allen_labels'][allen_label] = allen_name


blocks = os.listdir(data_dir)
labels_block_dict = {label: np.zeros((len(blocks),)) for label in labels_dict.keys()}
to_write_keys = ['Block number', 'xh', 'yh', 'zh', 'xm', 'ym', 'zm']
missing_labels = []
dict_block = {}
with open(os.path.join(RIGID_RESULTS, 'block_idx-name_relationship.txt'), 'r') as readFile:
    csvreader = csv.DictReader(readFile)
    for row in csvreader:
        dict_block["{:02d}".format(int(row['BLOCK_NUM']))] = row['BLOCK_NAME']

print('Missing labels in: ')
for it_b, b in enumerate(blocks):
    if not os.path.exists(os.path.join(REGISTER_DIR, dict_block[b], 'LABELS.nii.gz')):
        continue
    proxy = nib.load(os.path.join(REGISTER_DIR, dict_block[b], 'LABELS.nii.gz'))

    data = np.asarray(proxy.dataobj).astype('int')
    block_dict[b] = data

    unique_labels = np.unique(data)
    bincount_labels = np.bincount(data.reshape(-1))
    for ul in unique_labels:
        if ul not in labels_block_dict.keys():
            missing_labels.append(ul)
            idx = np.where(data == ul)
            print(' - (' + str(ul) + ',' + str(dict_block[b]) + ',' + str('-'.join([str(f) for f in np.unique(idx[2])])) + ')')
        else:
            labels_block_dict[ul][it_b] = bincount_labels[ul]

print('Fi.')
to_write = []
for label, fsd in fs_labels_dict.items():
    # if label != 3: continue
    print('#' + str(label) + ',', end=' ', flush=True)
    allen_labels = list(fsd['allen_labels'].keys())
    fs_bincount = np.zeros((len(blocks),))
    for al in allen_labels:
        if al == 611: continue
        fs_bincount += labels_block_dict[al]

    if np.sum(fs_bincount) == 0:
        continue

    block_num = np.argmax(fs_bincount)
    data = block_dict[blocks[block_num]]
    v2r_init = np.load(join(RIGID_RESULTS, 'slices', dict_block[blocks[block_num]], 'vox2ras0.npy'))
    # v2r_init = np.load(join(PROCESSED_DIR, dict_block[blocks[block_num]], 'LFB', 'vox2ras.npy'))

    slice_total = []
    for it_sl in range(data.shape[-1]):
        sum_tmp = 0
        for al in allen_labels:
            sum_tmp += np.sum(data[..., it_sl] == al)
        slice_total.append(sum_tmp)

    slice_num = np.argmax(slice_total)
    slice_max = data[..., slice_num]
    slice_max = ORIENTATION_CLASS.change_orientation(slice_max, dict_block[blocks[block_num]], v2r_init, vox2ras_MRI, order=0)
    indices = list(np.where(slice_max == allen_labels[0]))
    for al in allen_labels[1:]:
        idx_tmp = np.where(slice_max == al)
        indices[0] = np.concatenate((indices[0], idx_tmp[0]), axis=0)
        indices[1] = np.concatenate((indices[1], idx_tmp[1]), axis=0)

    x,y = centeroidnp(indices[0], indices[1])

    if not slice_max[int(x), int(y)] in allen_labels:
        set_of_points = np.zeros((indices[0].shape[0],2))
        set_of_points[..., 0] = indices[0]
        set_of_points[..., 1] = indices[1]
        point_i = np.argmin(np.sum((set_of_points - np.asarray([x, y])) ** 2, axis=1))
        x, y = set_of_points[point_i]

    affine_matrix = np.zeros((4, 4))
    with open(os.path.join(data_dir, blocks[block_num], 'matrix.txt'), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        for it_row, row in enumerate(csvreader):
            affine_matrix[it_row, 0] = float(row[0])
            affine_matrix[it_row, 1] = float(row[1])
            affine_matrix[it_row, 2] = float(row[2])
            affine_matrix[it_row, 3] = float(row[3])

    mri_coords = np.dot(affine_matrix, np.asarray([x, y, slice_num, 1]))


    l_dict = {
        'label': fsd['name'],
        'data': {'blockNumber': blocks[block_num], 'xh': x, 'yh': y, 'zh': int(slice_num),
                 'xm': mri_coords[0], 'ym': mri_coords[1], 'zm': mri_coords[2]},
        'children': []
    }

    for label in allen_labels:
        # if label < 2000: continue

        # if label in [435]:
        #     pdb.set_trace()
        if np.sum(labels_block_dict[label]) == 0:
            x, y, slice_num, block_num = 0, 0, 1, 0

        else:

            block_num = np.argmax(labels_block_dict[label])
            data = block_dict[blocks[block_num]]
            v2r_init = np.load(join(RIGID_RESULTS, 'slices',  dict_block[blocks[block_num]], 'vox2ras0.npy'))
            # v2r_init = np.load(join(PROCESSED_DIR, dict_block[blocks[block_num]], 'LFB', 'vox2ras.npy'))
            affine_matrix = np.zeros((4, 4))
            with open(os.path.join(data_dir, blocks[block_num], 'matrix.txt'), newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ')
                for it_row, row in enumerate(csvreader):
                    affine_matrix[it_row, 0] = float(row[0])
                    affine_matrix[it_row, 1] = float(row[1])
                    affine_matrix[it_row, 2] = float(row[2])
                    affine_matrix[it_row, 3] = float(row[3])


            slice_total = []
            for it_sl in range(data.shape[-1]):
                slice_total.append(np.sum(data[..., it_sl] == label))


            valid_indices = []
            for slice_num in np.argsort(slice_total)[::-1]:
                slice_max = data[..., slice_num]
                slice_max = ORIENTATION_CLASS.change_orientation(slice_max, dict_block[blocks[block_num]], v2r_init, vox2ras_MRI, order=0)
                indices = np.where(slice_max == label)
                x, y = centeroidnp(indices[0], indices[1])

                for idx_x, idx_y in zip(indices[0], indices[1]):
                    m_coords = np.round(np.dot(affine_matrix, np.asarray([idx_x, idx_y, slice_num, 1])))
                    h_coords = np.dot(np.linalg.inv(affine_matrix), m_coords)
                    if np.round(h_coords[2]) == slice_num and slice_max[int(np.round(h_coords[0])), int(np.round(h_coords[1]))] == label and index_vol[int(m_coords[0]), int(m_coords[1]), int(m_coords[2])] == int(blocks[block_num]):
                        valid_indices.append([idx_x, idx_y])

                if valid_indices:
                    break

            if not valid_indices:
                print('valid indices not found for label ' + str(label))
                x, y, slice_num, block_num = 0, 0, 1, 0

            else:
                set_of_points = np.zeros((len(valid_indices), 2))
                set_of_points[..., 0] = [i[0] for i in valid_indices]
                set_of_points[..., 1] = [i[1] for i in valid_indices]
                point_i = np.argmin(np.sum((set_of_points - np.asarray([x, y])) ** 2, axis=1))
                x, y = set_of_points[point_i]

            mri_coords = np.dot(affine_matrix, np.asarray([x, y, slice_num, 1]))
            # if label in [2007]:
            #     pdb.set_trace()
            #     h_coords = np.dot(np.linalg.inv(affine_matrix), np.asarray([np.round(mri_coords[0]), np.round(mri_coords[1]), np.round(mri_coords[2]), 1]))
            #     h_coords = np.dot(np.linalg.inv(affine_matrix), np.round(np.dot(affine_matrix, np.asarray([x, y, slice_num, 1]))))
        al_dict = {
            'label': fsd['allen_labels'][label],
            'data': {'blockNumber': int(blocks[block_num]), 'xh': x, 'yh': y, 'zh': int(slice_num),
                     'xm': mri_coords[0], 'ym': mri_coords[1], 'zm': mri_coords[2]},

        }

        l_dict['children'].append(al_dict)

    to_write.append(l_dict)

print('')
print('')
print('')

json_object = json.dumps(to_write, indent=4)
with open(table_file, "w") as outfile:
    outfile.write(json_object)
