# py
from os import makedirs
from argparse import ArgumentParser

# libraries imports

# project imports
from dataset.data_loader import DataLoader
from utils.deformation_utils import deform2D
from config import config_donors, config_dev, config_database
from utils.recon_utils import *


# ******************************************************************************************************************** #
#
# This file uses the computed latent SVF from the ST algorithm to deform the initial labeled sections at
# the original high-resolution (possibly with some --downsample_factor).
# It uses the SVF computed using the ST algorithm specified at --reg_algorithm
#
# ******************************************************************************************************************** #


####################
# Input parameters #
####################

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subject', default='P57-16', choices=list(config_donors.file_dict.keys()))
arg_parser.add_argument('--reg_algorithm', default='ST3_L1_RegNet_NN2', choices=['Linear', 'RegNet', 'ST3_L1_RegNet_NN2',  'ST3_L1_RegNet_NN4'])
arg_parser.add_argument('--disable_one_hot_flag', action='store_false')
arg_parser.add_argument('--downsample_factor', default=2, type=int)
arg_parser.add_argument('--final_labels', action='store_true')
arg_parser.add_argument('--bid', default=None, nargs='+')

arguments = arg_parser.parse_args()
SUBJECT = arguments.subject
ONE_HOT_FLAG = arguments.disable_one_hot_flag
FINAL_LABELS_FLAG = arguments.final_labels
reg_algorithm = arguments.reg_algorithm
downsample_factor = arguments.downsample_factor
initial_bid_list = arguments.bid

BT_DB_blocks = config_database.get_lin_dict(SUBJECT)
BT_DB_slices = config_database.get_nonlin_dict(SUBJECT)
data_dict = {SUBJECT: BT_DB_slices}
config_file_data = config_donors.file_dict[SUBJECT]
parameter_dict_MRI = config_dev.CONFIG_DICT['MRI']


REG_BLOCKS_DIR = join(BT_DB_blocks['RESULTS_DIR'], 'results', 'reg_blocks')
SLICES_DIR = join(BT_DB_blocks['RESULTS_DIR'], 'results', 'slices')

base_dir = join(config_file_data.BASE_DIR, 'NonlinearRegistration')
data_path_dict = {
    'RegNet': {'path': join(base_dir, 'Registration', 'RegNet'), 'stain': 'MRI_LFB'},
    'ST3_L1_RegNet_NN2': {'path': join(base_dir, 'ST3', 'l1', 'NN2'), 'stain': 'LFB'},
    'ST3_L1_RegNet_NN4': {'path': join(base_dir, 'ST3', 'l1', 'NN4'), 'stain': 'LFB'}
}

BLOCK_res = config_file_data.BLOCK_res
HISTO_res = config_file_data.HISTO_res
downsampleFactorHistoLabels = config_file_data.downsampleFactorHistoLabels

bid_list = initial_bid_list if initial_bid_list is not None else config_file_data.initial_bid_list
bid_list = {SUBJECT: bid_list}

data_loader = DataLoader(data_dict, included_blocks=bid_list)
subject_loader = data_loader.subject_dict[SUBJECT]
initial_bid_list = initial_bid_list if initial_bid_list is not None else config_file_data.initial_bid_list

for it_bid, bid in enumerate(initial_bid_list):
    print('')
    print('Block ID: ' + bid)

    block_loader = subject_loader.block_dict[bid]
    sbj_block = BT_DB_blocks['SUBJECT'] + '_' + bid
    BLOCK_DIR = join(SLICES_DIR, bid)

    if FINAL_LABELS_FLAG and ('C' not in bid and 'B' not in bid):
        LABELS_DIR = join(BT_DB_blocks['FINAL_HISTO_LABELS_DIR'], bid)
    else:
        LABELS_DIR = join(BT_DB_blocks['HISTO_LABELS_DIR'], bid)

    results_dir_sbj = join(data_path_dict[reg_algorithm]['path'], bid)
    proxy_flow = nib.load(join(results_dir_sbj, 'LFB.totalflow.nii.gz'))
    flow = np.asarray(proxy_flow.dataobj)

    if not exists(join(results_dir_sbj, 'LABELS')):
        makedirs(join(results_dir_sbj, 'LABELS'))

    labels_volume = np.zeros(flow.shape[1:])
    missing_slices = []
    rotation_angle = get_rotation_angle(SUBJECT, bid)
    print('   Slice number:', end=' ', flush=True)
    for slice in block_loader:
        sid = int(slice.sid)
        slice_num = sid - 1
        # if sid < 22: continue
        it_z = sid - 1
        sid_2str = "{:02d}".format(sid)

        print('#' + str(sid), end=' ', flush=True)

        # Get orig shape
        filename = BT_DB_blocks['SUBJECT'] + '_' + bid + '_LFB_' + sid_2str
        HISTO_DIR = join(BT_DB_blocks['SLIDES_DIR'], BT_DB_blocks['SUBJECT'] + '_' + bid)
        histo_mask_filepath = join(HISTO_DIR, 'LFB', filename + '.mask.png')
        orig_image = cv2.imread(histo_mask_filepath)
        orig_shape = orig_image.shape[:2]

        # Get label image
        label_image = get_label_image(sid, sbj_block, LABELS_DIR)
        if label_image is None:
            missing_slices.append(slice_num)
            continue

        # if SUBJECT == 'P57-16' and bid == 'B4.1':  # block B4.1 was flipped (up-down, so to find matching blocks we need to do that)
        #     resized_shape = tuple([int(i * downsampleFactorHistoLabels) for i in label_image.shape])
        #     L = cv2.resize(label_image, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_NEAREST)
        #     L = imrotate(L, -90, resize=True, order=0)  # rotate
        #     Lup = np.zeros_like(L)
        #     Lup[:-7500] = L[7500:]
        #     Lup = np.flipud(Lup)  # are 90 degrees rotated all labels
        #     Lup = imrotate(Lup, 90, resize=True, order=0)  # rotate
        #     label_image = cv2.resize(Lup, (label_image.shape[1], label_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        #     label_image = np.fliplr(label_image)
        #
        #     del L
        #     del Lup


        resized_shape = tuple([int(i / (HISTO_res * downsampleFactorHistoLabels) * BLOCK_res) for i in flow.shape[1:3]])
        factor = resized_shape[1] / flow.shape[2]
        field_i = factor * cv2.resize(flow[0, ..., it_z], (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
        factor = resized_shape[0] / flow.shape[1]
        field_j = factor * cv2.resize(flow[1, ..., it_z], (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)

        field = np.zeros((2,) + field_i.shape)
        field[0] = field_i
        field[1] = field_j

        del field_i
        del field_j

        if ONE_HOT_FLAG:
            unique_labels = np.unique(label_image)
            image_onehot_def = np.zeros(resized_shape + (len(unique_labels), ), dtype='float32')
            for it_ul, ul in enumerate(unique_labels):
                image_onehot = np.zeros_like(label_image).astype('float32')
                image_onehot[label_image == ul] = 1
                if not (FINAL_LABELS_FLAG and ('C' not in bid and 'B' not in bid)):
                    image_onehot = preprocess_label_image(image_onehot, SUBJECT, bid, rotation_angle, order=1)

                image_onehot_def[..., it_ul] = deform2D(image_onehot, field, mode='bilinear')
                del image_onehot

            image_def_unordered = np.argmax(image_onehot_def, axis=-1).astype('uint16')

            del image_onehot_def

            image_def = np.zeros_like(image_def_unordered, dtype = 'uint16')

            for it_ul, ul in enumerate(unique_labels): image_def[image_def_unordered == it_ul] = ul

            del image_def_unordered

            resized_shape_labels = tuple([int(i * downsampleFactorHistoLabels / downsample_factor) for i in image_def.shape])
            image_def = cv2.resize(image_def, (resized_shape_labels[1], resized_shape_labels[0]), interpolation=cv2.INTER_NEAREST)

        else:
            if not (FINAL_LABELS_FLAG and ('C' not in bid and 'B' not in bid)):
                label_image = preprocess_label_image(label_image, SUBJECT, bid, rotation_angle, order=0)

            image_def = deform2D(label_image, field, mode='nearest')
            image_def = image_def.astype(np.uint16)

            resized_shape_labels = tuple([int(i * downsampleFactorHistoLabels / downsample_factor) for i in image_def.shape])
            image_def = cv2.resize(image_def, (resized_shape_labels[1], resized_shape_labels[0]), interpolation=cv2.INTER_NEAREST)

        np.savez_compressed(join(results_dir_sbj, 'LABELS', 'slice_' + sid_2str + str(downsample_factor) + 'D.npz'), data=image_def)

print('')