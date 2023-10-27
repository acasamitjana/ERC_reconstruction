# It is important that all files contain the same structure: prefix + '_' + slice_num + '.' + extension
from os.path import join
from setup_repo import *
SLICE_PREFIX = 'slice'



def get_nonlin_dict(subject):
    d = {
        'SLICE_PREFIX': 'slice_',
        'IMAGE_EXTENSION': '.png',
        'NAME': 'BUNGEE_Tools',
        'SUBJECT': subject,
    }

    d['BASE_DIR'] = join(RESULTS_DIR, subject, 'RigidRegistration', 'results', 'slices')
    d['LINEAR_RUN'] = '1'
    d['DATASET_SHEET'] = join(RESULTS_DIR, subject, 'RigidRegistration', 'results', 'slices', 'slice_id.txt')

    return d

def get_lin_dict(subject):
    d = {
        'NAME': 'BUNGEE_Tools',
        'SUBJECT': subject,
        'SUFIX': 'initialized',
        'IMAGE_EXTENSION': '.mgz',

        'RESULTS_DIR': join(RESULTS_DIR, subject, 'RigidRegistration'),

        'PREPROCESS_DIR': join(DATA_DIR, subject, 'mosaicPuzzle', 'rotation_files'),
        'SLIDES_DIR': join(DATA_DIR, subject, 'ScannedSlides'),
        'HISTO_LABELS_DIR': join(DATA_DIR, subject, 'Labels_substructure'),
        'FINAL_HISTO_LABELS_DIR': join(DATA_DIR, subject, 'Final_labels'),
        'HISTO_SUBSTRUCTURE_DIR': join(DATA_DIR, subject, 'Labels_substructure'),

        'BASE_DIR': join(DATA_DIR, subject),
        'INIT_BLOCKS_DIR': join(DATA_DIR, subject, 'initializedHistoBlocks'),
        'DOWNSAMPLE_DIR': join(DATA_DIR, subject, 'mosaicPuzzle', 'initHistoBlocksDownsampled1mm_GAP'),

        'MRI': {
            'MRI_FILE': join(DATA_DIR, subject, 'MRI', 'averageWithReg.stripped.bfcorr.rasoriented.nii.gz'),
            'MRI_APARC_FILE': join(DATA_DIR, subject, 'MRI', 'aparc+aseg.rasoriented.nii.gz'),
            'MRI_ASEG_FILE': join(DATA_DIR, subject, 'MRI', 'aseg.rasoriented.nii.gz'),
            'MRI_MASK_FILE': join(DATA_DIR, subject, 'MRI', 'mask_mri.rasoriented.nii.gz'),
            'MRI_MASK_DILATED_FILE': join(DATA_DIR, subject, 'MRI', 'mask_mri.dilated.rasoriented.nii.gz'),
            'MRI_MASK_CEREBELLUM_FILE': join(DATA_DIR, subject, 'MRI', 'mask_cerebellum.rasoriented.nii.gz'),
            'MRI_MASK_CEREBRUM_FILE': join(DATA_DIR, subject, 'MRI', 'mask_cerebrum.rasoriented.nii.gz'),
            'MRI_MASK_BS_FILE': join(DATA_DIR, subject, 'MRI', 'mask_brainstem.rasoriented.nii.gz'),
        },
    }

    d['DOWNSAMPLE_MRI'] = {
        'MRI_FILE': join(d['DOWNSAMPLE_DIR'], 'DownsampledMRI.mgz'),
        'MRI_APARC_FILE': join(DATA_DIR, subject, 'MRI', 'aparc+aseg.mgz'),
        'MRI_ASEG_FILE': join(DATA_DIR, subject, 'MRI', 'aseg.mgz'),
        'MRI_MASK_FILE': join(d['DOWNSAMPLE_DIR'], 'DownsampledMRI.mask.mgz'),
        'MRI_MASK_CEREBELLUM_FILE': join(d['DOWNSAMPLE_DIR'], 'DownsampledMRI.mask.cerebellum.mgz'),
        'MRI_MASK_CEREBRUM_FILE': join(d['DOWNSAMPLE_DIR'], 'DownsampledMRI.mask.cerebrum.mgz'),
        'MRI_MASK_BS_FILE': join(d['DOWNSAMPLE_DIR'], 'DownsampledMRI.mask.bs.mgz'),
        'MRI_MASK_DILATED_FILE': join(d['DOWNSAMPLE_DIR'], 'DownsampledMRI.mask.dilated.mgz'),
    }

    return d

