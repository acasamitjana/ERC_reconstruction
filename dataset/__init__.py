import csv
from os.path import join
import openpyxl

from setup_repo import BASE_DIR

ontology_file = join(BASE_DIR, 'Data', 'BUNGEE_TOOLS', 'Documents', 'FS_label_mapping.xlsx')
label_file = join(BASE_DIR, 'Data', 'BUNGEE_TOOLS', 'Documents', 'WholeHemisphereFS.txt')

def load_labels():
    dict_labels = {}
    with open(label_file, 'r') as txtfile:
        data = txtfile.readlines()
    for d in data[1:]:
        dsplit = d.split('  ')
        label = int(dsplit[0])
        label_name = dsplit[1]
        rgb = dsplit[2].split(' ')
        rgb = [int(c) for c in rgb[:3]]

        dict_labels[label] = {'num': label, 'name': label_name, 'rgb': rgb}

    return dict_labels


def load_ontology(unique_labels=None):
    labels_list = []
    h2_labels_dict = {}
    h1_labels_dict = {}

    # Read label mapping from Nellie and Juri: nested dictionaries with different ontologies form the top to the bottom.
    wb = openpyxl.load_workbook(ontology_file)
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


def labels_allen_to_fs():

    labels, h_dict = load_ontology()
    convert_dict = {}
    for fs_label, allen_d in h_dict[0].items():
        for allen_label, allen_name in allen_d['allen_labels'].items():
            convert_dict[allen_label] = fs_label

    return convert_dict

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
