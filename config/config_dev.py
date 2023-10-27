import copy

from setup_repo import *
from utils.transforms import ScaleNormalization, AffineParams, NonLinearParams, CropParams

# -------- #
# This file contains all the configuration parameters to train registration networks: spatial augmentation,
# intensity normalization, loss function, architecture, learning rate, etc... Feel free to change it
# -------- #

REGISTRATION_DIR = '/home/acasamitjana/Results/Registration/BUNGEE_Tools/Registration'
CONFIG_REGISTRATION = {
    'NORMALIZATION': ScaleNormalization(range=[0, 1]),
    'AFFINE': AffineParams(rotation=[2.5]*2, scaling=[0]*2, translation=[2]*2),
    'NONLINEAR': NonLinearParams(lowres_strength=3, lowres_shape_factor=0.04, distribution='uniform'),
    'TRANSFORM': [CropParams((896, 960))],

    'BATCH_SIZE': 5,
    'N_EPOCHS': 100,
    'LEARNING_RATE': 1e-3,
    'EPOCH_START_DECAY_LR': -1,
    'STARTING_EPOCH': 0,

    'ENC_NF': [16, 32, 32, 64, 64, 64],
    'DEC_NF': [64, 64, 64, 32, 32, 32, 16],

    'USE_GPU': True,
    'GPU_INDICES': [0],

    'LOG_INTERVAL': 10,
    'SAVE_MODEL_FREQUENCY': 100,

    'LOSS_REGISTRATION': {'name': 'NCC', 'params': {'kernel_var': [9, 9], 'kernel_type': 'mean'}, 'lambda': 1},
    'LOSS_REGISTRATION_SMOOTHNESS': {'name': 'Grad', 'params': {'dim': 2, 'penalty': 'l2'}, 'lambda': 1},
    'LOSS_REGISTRATION_LABELS': {'name': 'SemiDice', 'params': {'reduction': 'mean'}, 'lambda': 0.2},

    'NEIGHBOUR_DISTANCE': 4,
    'REF_MODALITY': None,
    'FLO_MODALITY': None,

    'UPSAMPLE_LEVELS': 4,
    'FIELD_TYPE': 'velocity',
    'RESULTS_DIR': REGISTRATION_DIR
}

CONFIG_INTERMODAL = copy.copy(CONFIG_REGISTRATION)
CONFIG_INTRAMODAL = copy.copy(CONFIG_REGISTRATION)

# MRI
CONFIG_MRI = copy.copy(CONFIG_INTRAMODAL)
CONFIG_MRI['REF_MODALITY'] = 'MRI'
CONFIG_MRI['FLO_MODALITY'] = 'MRI'
CONFIG_MRI['LOSS_REGISTRATION'] = {'name': 'NCC', 'params': {'kernel_var': [9, 9], 'kernel_type': 'mean'}, 'lambda': 1}

CONFIG_LFB = copy.copy(CONFIG_INTRAMODAL)
CONFIG_LFB['REF_MODALITY'] = 'LFB'
CONFIG_LFB['FLO_MODALITY'] = 'LFB'
# CONFIG_LFB['LOSS_REGISTRATION_SMOOTHNESS'] = {'name': 'Grad', 'params': {'dim': 2, 'penalty': 'l2'}, 'lambda': 0.1}
CONFIG_LFB['LOSS_REGISTRATION'] = {'name': 'NCC', 'params': {'kernel_var': [9, 9], 'kernel_type': 'mean'}, 'lambda': 1}

CONFIG_HE = copy.copy(CONFIG_INTRAMODAL)
CONFIG_HE['REF_MODALITY'] = 'HE'
CONFIG_HE['FLO_MODALITY'] = 'HE'
# CONFIG_HE['LOSS_REGISTRATION_SMOOTHNESS'] = {'name': 'Grad', 'params': {'dim': 2, 'penalty': 'l2'}, 'lambda': 0.1}
CONFIG_HE['LOSS_REGISTRATION'] = {'name': 'NCC', 'params': {'kernel_var': [9, 9], 'kernel_type': 'mean'}, 'lambda': 1}

CONFIG_MRI_LFB = copy.copy(CONFIG_MRI)
CONFIG_MRI_LFB['REF_MODALITY'] = 'MRI'
CONFIG_MRI_LFB['FLO_MODALITY'] = 'LFB'
CONFIG_MRI_LFB['LEARNING_RATE'] = 2e-4
CONFIG_MRI_LFB['LOSS_NCE'] = {'name': 'NCE', 'params': {'temp': 0.05}, 'lambda': 0.005}
CONFIG_MRI_LFB['N_EPOCHS'] = 50
# CONFIG_MRI_LFB['LOSS_REGISTRATION_SMOOTHNESS']['lambda'] = 0

CONFIG_MRI_HE = copy.copy(CONFIG_MRI)
CONFIG_MRI_HE['REF_MODALITY'] = 'MRI'
CONFIG_MRI_HE['FLO_MODALITY'] = 'HE'
CONFIG_MRI_HE['LEARNING_RATE'] = 2e-4
CONFIG_MRI_HE['LOSS_NCE'] = {'name': 'NCE', 'params': {'temp': 0.05}, 'lambda': 0.005}
CONFIG_MRI_HE['N_EPOCHS'] = 50
# CONFIG_MRI_HE['LOSS_REGISTRATION_SMOOTHNESS']['lambda'] = 0

CONFIG_LFB_HE = copy.copy(CONFIG_LFB)
CONFIG_LFB_HE['REF_MODALITY'] = 'LFB'
CONFIG_LFB_HE['FLO_MODALITY'] = 'HE'
CONFIG_LFB_HE['LEARNING_RATE'] = 2e-4
CONFIG_LFB_HE['LOSS_NCE'] = {'name': 'NCE', 'params': {'temp': 0.05}, 'lambda': 0.005}
CONFIG_LFB_HE['N_EPOCHS'] = 50
# CONFIG_LFB_HE['LOSS_REGISTRATION_SMOOTHNESS']['lambda'] = 0

CONFIG_DICT = {
    'REG': CONFIG_REGISTRATION,
    'MRI': CONFIG_MRI,
    'LFB': CONFIG_LFB,
    'HE': CONFIG_HE,
    'MRI_LFB': CONFIG_MRI_LFB,
    'MRI_HE': CONFIG_MRI_HE,
    'LFB_HE': CONFIG_LFB_HE
}


for key, CONFIG in CONFIG_DICT.items():
    if CONFIG['LOSS_REGISTRATION']['name'] == 'NCC':
        loss_dir = CONFIG['LOSS_REGISTRATION']['name'] + \
                   str(CONFIG['LOSS_REGISTRATION']['params']['kernel_var'][0]) + \
                   '_RegNet'
    else:
        loss_dir = CONFIG['LOSS_REGISTRATION']['name'] + '_RegNet'

    weight_dir = 'R' + str(CONFIG['LOSS_REGISTRATION']['lambda']) + \
                 '_S' + str(CONFIG['LOSS_REGISTRATION_SMOOTHNESS']['lambda'])
    downfactor_dir = 'DownFactor_' + str(CONFIG['UPSAMPLE_LEVELS'])
    CONFIG['RESULTS_DIR'] = join(RESULTS_DIR, 'Registration', loss_dir, weight_dir, downfactor_dir, key)

def get_params_dir(stain, config_dict):
    if config_dict['LOSS_REGISTRATION']['name'] == 'NCC':
        loss_dir = config_dict['LOSS_REGISTRATION']['name'] + \
                   str(config_dict['LOSS_REGISTRATION']['params']['kernel_var'][0]) + \
                   '_RegNet'
    else:
        loss_dir = config_dict['LOSS_REGISTRATION']['name'] + '_RegNet'

    weight_dir = 'R' + str(config_dict['LOSS_REGISTRATION']['lambda']) + \
                 '_S' + str(config_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda'])
    downfactor_dir = 'DownFactor_' + str(config_dict['UPSAMPLE_LEVELS'])
    config_dict['RESULTS_DIR'] = join(RESULTS_DIR, 'Registration', loss_dir, weight_dir, downfactor_dir, stain)

    return config_dict
# def get_config_dict():
#     for k, config in CONFIG_DICT.items():
#         if config['LOSS_REGISTRATION']['name'] == 'NCC':
#             loss_dir = config['LOSS_REGISTRATION']['name'] + str(config['LOSS_REGISTRATION']['params']['kernel_var'][0]) + \
#                        '_RegNet'
#         else:
#             loss_dir = config['LOSS_REGISTRATION']['name']
#
#         loss_dir = join(loss_dir, 'R' + str(config['LOSS_REGISTRATION']['lambda']) +
#                         '_S' + str(config['LOSS_REGISTRATION_SMOOTHNESS']['lambda']))
#         downfactor_dir = 'DownFactor_' + str(config['UPSAMPLE_LEVELS'])
#         # CONFIG_DICT[k]['DB_CONFIG'] = data_config
#         CONFIG_DICT[k]['RESULTS_DIR'] = join(RESULTS_DIR, 'Registration',  loss_dir, downfactor_dir, k)
#
#     return CONFIG_DICT