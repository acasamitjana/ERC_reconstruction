import csv
import pdb
from os.path import join, exists

import numpy as np
from PIL import Image

from dataset import labels_allen_to_fs
from utils import image_utils, io_utils

CONVERT_DICT = labels_allen_to_fs()
CONVERT_DICT[626] = 0# Found in P41-16_P4.1_33
CONVERT_DICT[577] = 0# Found in P41-16_P4.1_30-33
CONVERT_DICT[644] = 0# Found in P41-16_P4.1_28-33
CONVERT_DICT[578] = 0# Found in P41-16_P4.1
CONVERT_DICT[65] = 0# Found in P41-16_P3.3, P41_16_P4.1
CONVERT_DICT[473] = 0# Found in P41-16_P2.4_14/15/16/18/19/20/22/23/24
CONVERT_DICT[318] = 0# Found in P41-16_P2.4_20/21
CONVERT_DICT[506] = 0# Found in P41-16_P2.4_6/7/8/10/11/12
CONVERT_DICT[523] = 0# Found in P41-16_P2.4_2/3/4
CONVERT_DICT[413] = 0# Found in P41-16_P1.3_15
CONVERT_DICT[843] = 842# Found in P41-16_A1.2_28-34
CONVERT_DICT[24] = 16 # added here but need to be corrected by Nellie
CONVERT_DICT[1] = 1 # added here but need to be corrected by Nellie
CONVERT_DICT[0] = 0 # added here but need to be corrected by Nellie


class DataLoader(list):

    def __init__(self, database_config, **kwargs):
        '''
        This class is used to initialise a dictionary (subject_dict) containing all subjects specified.
        :param database_config: dictionary containing paths for all subjects e.g.: {SBJ_ID: SBJ_DICT}
        :param kwargs: e.g., excluded_blocks, included_blocks
        '''

        super().__init__()

        self.database_config = database_config # multiple parameter dicts
        self._initialize_dataset(**kwargs)


    def _initialize_dataset(self, excluded_blocks=None, included_blocks=None):
        '''
        If neither excluded_blocks nor included_blocks are specified, the dataset is initialised with all subjects and
        blocks available, as specified in the DATASET_SHEET field of the configuration files.
        :param excluded_blocks: (None, optional) it is a dictionary used to specify which subjects (keys) and
        blocks (values) to exclude.
        :param included_blocks: (None, optional) it is a dictionary used to specify which subjects (keys) and
        blocks (values) to inclue. :return: None
        '''

        subject_dict = {}
        for sbj_id, sjb_dict in self.database_config.items():

            block_dict = {}

            with open(sjb_dict['DATASET_SHEET'], 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:

                    if (excluded_blocks is not None) and (sbj_id in excluded_blocks.keys()) and \
                            (row['BLOCK_ID'] in excluded_blocks[sbj_id]): continue

                    if (included_blocks is not None):
                        if (sbj_id not in included_blocks.keys()):
                            continue
                        elif row['BLOCK_ID'] not in included_blocks[sbj_id]:
                            continue

                    if not exists(join(sjb_dict['BASE_DIR'], row['BLOCK_ID'])): continue

                    if row['BLOCK_ID'] not in block_dict.keys():
                        block_dict[row['BLOCK_ID']] = Block(bid=row['BLOCK_ID'],  data_path=sjb_dict['BASE_DIR'])

                    block_dict[row['BLOCK_ID']].add_slice(slice_id=row['SLICE_ID'],
                                                          id_mm=float(row['SLICE_ID_MM']),
                                                          slice_prefix=sjb_dict['SLICE_PREFIX'],
                                                          structures=row['STRUCTURES']
                                                          )

            sbj = Subject(sjb_dict['SUBJECT'])
            sbj._initialize_subject(block_dict)
            subject_dict[sjb_dict['SUBJECT']] = sbj

        self._subject_dict = subject_dict

    @property
    def image_shape(self):
        '''
        Global image shape for the sections/slices of all subjects and blocks.
        :return:
        '''
        return (896, 960)
        # ishape = [0, 0]
        # for sid, s in self._subject_dict.items():
        #     image_shape = s.image_shape
        #     if image_shape[0] > ishape[0]: ishape[0] = image_shape[0]
        #     if image_shape[1] > ishape[1]: ishape[1] = image_shape[1]
        #
        # return ishape

    @property
    def subject_dict(self):
        return self._subject_dict

    @property
    def sid_list(self):
        return [sbj for sbj in self._subject_dict.keys()]

    @property
    def subjects_list(self):
        return [sbj for sbj in self._subject_dict.values()]

    def items(self):
        for bid, block in self._subject_dict.items():
            yield bid, block

    def __getitem__(self, item):
        return self._subject_dict[item]

    def __len__(self):
        return len(self._subject_dict.keys())

    def __iter__(self):
        for block in self._subject_dict.values():
            yield block


class Subject(list):

    def __init__(self, name):
        '''
        This class represents a case from the BUNGEE-TOOLS dataset
        :param name:
        '''
        super().__init__()

        self._substructures = None
        self.name = name
        self._block_dict = {}

    def _initialize_subject(self, block_dict):
        substructures = []
        for bid, block in block_dict.items():
            block._initialize_block()
            substructures.extend(block.substructures)

            self._block_dict[bid] = block

        self._substructures = np.unique(substructures)

    @property
    def substructures(self):
        return self._substructures

    @property
    def block_dict(self):
        return self._block_dict

    @property
    def bid_list(self):
        return [sbj for sbj in self.block_dict.keys()]

    @property
    def block_list(self):
        return [sbj for sbj in self.block_dict.values()]

    @property
    def image_shape(self):
        ishape = [0,0]
        for bid, b in self._block_dict.items():
            image_shape = b.image_shape
            if image_shape[0] > ishape[0]: ishape[0] = image_shape[0]
            if image_shape[1] > ishape[1]: ishape[1] = image_shape[1]

        return ishape

    def __len__(self):
        return len(self._block_dict.keys())

    def __getitem__(self, item):
        # item = np.mod(item, len(self._block_dict.keys()))
        # return list(self._block_dict.values())[item]
        return self._block_dict[item]

    def __iter__(self):
        for block in self._block_dict.values():
            yield block


class Block(list):

    def __init__(self, bid, data_path):
        '''
        This class represents a block from a given subject
        :param bid:
        :param data_path:
        '''
        super().__init__()

        self.data_path = join(data_path, bid)
        self._bid = bid
        self._slice_dict = {}

        self._substructures = None

        self.modality_path = {
            'MRI': join(self.data_path, 'MRI'),
            'LFB': join(self.data_path, 'LFB'),
            'HE': join(self.data_path, 'HE'),

        }

    def __len__(self):
        return len(self.slice_list)

    def __getitem__(self, item):
        return self._slice_dict[item]

    def __iter__(self):
        for sl in self._slice_dict.values():
            yield sl

    def _initialize_block(self):
        substructures = []
        for sl in self.slice_list:
            substructures.extend(sl.substructures)

        self._substructures = np.unique(substructures)

    def add_slice(self, slice_id, **kwargs):
        slice = Slice(subject=self, sid=slice_id, **kwargs )
        self._slice_dict[slice_id] = slice

    @property
    def substructures(self):
        return self._substructures

    @property
    def id(self):
        return self._bid

    @property
    def nslices(self):
        return np.max([int(slid) for slid in self._slice_dict.keys()])

    @property
    def slid_list(self):
        return [slid for slid in self._slice_dict.keys()]

    @property
    def slice_list(self):
        return [sl for sl in self._slice_dict.values()]

    @property
    def vol_shape(self):
        return self.image_shape + (len(self.slice_list),)

    @property
    def image_shape(self):
        return self.slice_list[0].image_shape

    @property
    def num_classes(self):
        if self.substructures is None:
            return 0
        else:
            return len(self.substructures)

    @property
    def vox2ras0(self):
        if exists(join(self.data_path, 'vox2ras0.npy')):
            vox2ras0 = np.load(join(self.data_path, 'vox2ras0.npy'))
        else:
            vox2ras0 = io_utils.read_affine_matrix(join(self.data_path, 'vox2ras0.txt'))
            np.save(join(self.data_path, 'vox2ras0.npy'), vox2ras0)

        return vox2ras0


class Slice(object):

    def __init__(self, subject, sid, id_mm=None, slice_prefix=None, structures=None, use_fs_labels=True):
        '''
        This class represents a given slice of the datset with the loading methods needed.
        :param subject:
        :param sid: slice id
        :param id_mm: separation from the first slice of the block (in mm).
        :param slice_prefix:
        :param structures:
        :param use_fs_labels:
        '''
        self.subject = subject
        self._sid = sid
        self._id = subject.id + '_' + sid
        self._id_mm = id_mm if id_mm is not None else sid

        self._substructures = [int(l) for l in structures.split(' ')]
        if use_fs_labels:
            self._substructures = np.unique([CONVERT_DICT[int(l)] for l in self._substructures if int(l) in CONVERT_DICT.keys()])
            # self._substructures = np.unique([CONVERT_DICT[int(l)] for l in self._substructures])

        self.use_fs_labels = use_fs_labels
        if slice_prefix is None: slice_prefix = ''

        self.image_name = slice_prefix + sid + '.png'
        self.label_name = slice_prefix + sid + '.npy'
        self.affine_name = slice_prefix + sid + '.aff'

        # warnings.warn('MRI has no labels for this block. Instead, a zero-array will be returned.', category=DataWarning)
        # warnings.warn('HE has no labels for this block. Instead, a zero-array will be returned.', category=DataWarning)

    @property
    def substructures(self):
        return self._substructures

    @property
    def id(self):
        return self._id

    @property
    def sid(self):
        return self._sid

    @property
    def id_mm(self):
        return float(self._id_mm)

    @property
    def image_shape(self):
        mask = self.load_data(modality='MRI')
        return mask.shape

    def load_data(self, modality, **kwargs):
        data = Image.open(join(self.subject.modality_path[modality], 'images', self.image_name))
        data = np.array(data)
        return data

    def load_mask(self, modality, **kwargs):
        data = Image.open(join(self.subject.modality_path[modality], 'masks', self.image_name))
        data = np.array(data)
        if np.sum(data) == 0:
            return data
        else:
            return data / np.max(data)

    def load_labels(self, modality, **kwargs):
        return self.load_mask(modality, **kwargs)
        # path_npy = join(self.subject.modality_path[modality], 'labels', self.label_name)
        # path_png = join(self.subject.modality_path[modality], 'labels', self.image_name)
        #
        # if not exists(path_npy) and not exists(path_png):
        #     return np.zeros(self.image_shape)
        #
        # if exists(path_npy):
        #     data = np.load(path_npy)
        #
        # elif exists(path_png):
        #
        #     data = Image.open(path_png).convert('L')
        #     data = np.array(data)
        #
        # if self.use_fs_labels and modality == 'LFB':
        #     data_tmp = np.zeros_like(data)
        #     for l in np.unique(data):
        #         idx = np.where(data==l)
        #         data_tmp[idx] = CONVERT_DICT[l]
        #     data = data_tmp
        #
        # data = image_utils.normalize_target_tensor(data, class_labels=self.substructures)
        #
        # return data.astype('int16')

    def load_affine(self, modality):
        affine_matrix = np.zeros((2, 3))

        with open(join(self.subject.modality_path[modality], 'affine', self.affine_name), 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ')
            row = next(csvreader)
            affine_matrix[0, 0] = float(row[0])
            affine_matrix[0, 1] = float(row[1])
            affine_matrix[0, 2] = float(row[3])
            row = next(csvreader)
            affine_matrix[1, 0] = float(row[0])
            affine_matrix[1, 1] = float(row[1])
            affine_matrix[1, 2] = float(row[3])

        return affine_matrix
