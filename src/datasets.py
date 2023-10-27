# py

# third party imports
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from scipy.ndimage.morphology import binary_dilation
from skimage.exposure import match_histograms
from skimage.measure import label as CC

#project imports
from utils import transforms as tf
from utils.image_utils import one_hot_encoding


####  ----------  ####
####   Samplers   ####
####  ----------  ####

class InterModalSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)

        N = 0
        self.data_source = data_source #data_loader
        id_list = []
        for sid, subject in data_source.items():
            for bid, block in subject.block_dict.items():
                id_list.append(sid + '___' + bid)
                N += len(block)

        np.random.shuffle(id_list)
        self.id_list = id_list
        self.N = N

    def __iter__(self):
        for sbid in self.id_list:
            sid, bid = sbid.split('___')
            subject = self.data_source[sid]
            block = subject[bid]
            slid_list = block.slid_list
            np.random.shuffle(slid_list)
            for index_ref, rid_ref in enumerate(slid_list):
                yield sid, bid, rid_ref

    def __len__(self):
        return self.N

class IntraModalSampler(Sampler):
    def __init__(self, data_source, neighbor_distance=0):
        super().__init__(data_source)

        N = 0
        self.data_source = data_source #data_loader
        id_list = []
        for sid, subject in data_source.items():
            for bid, block in subject.block_dict.items():
                id_list.append(sid + '___' + bid)
                N += len(block)

        np.random.shuffle(id_list)
        self.id_list = id_list
        self.N = N
        self.neighbor_distance = neighbor_distance

    def __iter__(self):
        for sbid in self.id_list:
            sid, bid = sbid.split('___')
            subject = self.data_source[sid]
            block = subject[bid]
            slid_list = block.slid_list
            for index_ref, rid_ref in enumerate(slid_list):
                neigh_min = np.clip(index_ref - self.neighbor_distance, 0, len(slid_list)-1)
                neigh_max = np.clip(index_ref + self.neighbor_distance, 0, len(slid_list)-1)
                index_flo = int(np.random.choice(np.arange(neigh_min, neigh_max + 1), size=1))
                rid_flo = slid_list[index_flo]
                yield sid, bid, rid_ref, rid_flo

    def __len__(self):
        return self.N

class IntraModalFixedNeighSampler(Sampler):

    def __init__(self, data_source, neighbor_distance=0):
        super().__init__(data_source)
        N = 0
        self.data_source = data_source  # data_loader
        id_list = []
        for sid, subject in data_source.items():
            for bid, block in subject.block_dict.items():
                id_list.append(sid + '_' + bid)
                N += len(block)
                N -= neighbor_distance

        np.random.shuffle(id_list)
        self.id_list = id_list
        self.N = N
        self.neighbor_distance = neighbor_distance


    def __iter__(self):
        for sbid in self.id_list:
            sid, bid = sbid.split('_')
            subject = self.data_source[sid]
            block = subject[bid]
            slid_list = block.slid_list
            for index_ref, rid_ref in enumerate(slid_list):
                index_flo = index_ref + self.neighbor_distance
                index_flo = np.clip(index_flo, 0, len(slid_list))
                rid_flo = slid_list[index_flo]
                yield sid, bid, rid_ref, rid_flo

    def __len__(self):
        return self.N

class BlockInterModalSampler(Sampler):

    def __init__(self, block):
        super().__init__(block)
        self.N = len(block)
        self.block = block  # data_loader

    def __iter__(self):
        slid_list = self.block.slid_list
        for rid_ref in slid_list:
            yield rid_ref

    def __len__(self):
        return self.N

class BlockIntraModalFixedNeighSampler(Sampler):

    def __init__(self, block, neighbor_distance=0):
        super().__init__(block)
        self.block = block  # data_loader

        N = len(block)
        N -= neighbor_distance

        self.N = N
        self.neighbor_distance = neighbor_distance


    def __iter__(self):
        slid_list = self.block.slid_list
        if self.neighbor_distance == 0:
            for index_ref, rid_ref in enumerate(slid_list):
                index_flo = index_ref + self.neighbor_distance
                if index_flo >= len(slid_list):
                    break
                index_flo = np.clip(index_flo, 0, len(slid_list))
                rid_flo = slid_list[index_flo]
                yield rid_ref, rid_flo

        else:
            for index_ref, rid_ref in enumerate(slid_list[:-self.neighbor_distance]):
                index_flo = index_ref + self.neighbor_distance
                if index_flo >= len(slid_list):
                    break
                index_flo = np.clip(index_flo, 0, len(slid_list))
                rid_flo = slid_list[index_flo]
                yield rid_ref, rid_flo


    def __len__(self):
        return self.N

####  ----------  ####
####   Datasets   ####
####  ----------  ####
class RegistrationDataset(Dataset):

    def __init__(self, data_loader, affine_params, nonlinear_params, tf_params=None, norm_params=None,
                 hist_match=False, mask_dilation=None, to_tensor=True, landmarks=None, num_classes=False,
                 sbj_per_epoch=1, train=True):
        '''

        :param data_loader:
        :param rotation_params:
        :param nonlinear_params:
        :param tf_params:
        :param da_params:
        :param norm_params:
        :param hist_match:
        :param mask_dilation:
        :param to_tensor:
        :param landmarks:
        :param num_classes: (int) number of classes for one-hot encoding. If num_classes=-1, one-hot is not performed.
        :param train:
        '''

        self.data_loader = data_loader
        self.N_sbj = len(data_loader)
        self.N = None
        self.to_tensor = to_tensor
        self.landmarks = landmarks

        self.tf_params = tf.Compose(tf_params) if tf_params is not None else None
        self.norm_params = norm_params if norm_params is not None else lambda x: x
        image_shape = data_loader.image_shape
        self._image_shape = self.tf_params._compute_data_shape(image_shape) if tf_params is not None else image_shape


        self.hist_match = hist_match
        self.affine_params = affine_params
        self.nonlinear_params = nonlinear_params

        self.mask_dilation = mask_dilation
        self.num_classes = num_classes

        self.train = train
        self.sbj_per_epoch = sbj_per_epoch

        self.ref_modality = None
        self.flo_modality = None

    def mask_image(self, image, mask):
        ndim = len(image.shape)

        if ndim == 3:
            for it_z in range(image.shape[-1]):
                image[..., it_z] = image[..., it_z] * mask
        else:
            image = image*mask

        return image

    def get_ref_data(self, slice, *args, **kwargs):

        x_ref = slice.load_data(modality=self.ref_modality, *args, **kwargs)
        x_ref_mask = (slice.load_mask(modality=self.ref_modality, *args, **kwargs) > 0).astype('int')
        x_ref_labels = slice.load_labels(modality=self.ref_modality, *args, **kwargs)
        x_ref = self.mask_image(x_ref, x_ref_mask)

        if np.sum(x_ref_mask) > 0:
            x_ref = self.norm_params(x_ref)
            x_ref = self.mask_image(x_ref, x_ref_mask)

        return x_ref, x_ref_mask, x_ref_labels

    def get_flo_data(self, slice, rgb=False, *args, **kwargs):

        x_flo = slice.load_data(modality=self.flo_modality, *args, **kwargs)
        x_flo_mask = (slice.load_mask(modality=self.flo_modality, *args, **kwargs) > 0).astype('int')
        x_flo_labels = slice.load_labels(modality=self.flo_modality, *args, **kwargs)
        x_flo = self.mask_image(x_flo, x_flo_mask)

        if np.sum(x_flo_mask) > 0:
            x_flo = self.norm_params(x_flo)
            x_flo = self.mask_image(x_flo, x_flo_mask)

        return x_flo, x_flo_mask, x_flo_labels

    def get_intermodal_data(self, slice, rgb=False, *args, **kwargs):

        x_ref, x_ref_mask, x_ref_labels = self.get_ref_data(slice, rgb=rgb, *args, **kwargs)
        x_flo, x_flo_mask, x_flo_labels = self.get_flo_data(slice, rgb=rgb, *args, **kwargs)

        if self.hist_match and (np.sum(x_ref_mask) > 0 or np.sum(x_flo_mask) > 0) and self.flo_modality != 'MRI':
            x_flo = np.max(x_flo) - x_flo
            x_flo_vec = x_flo[x_flo_mask > 0]
            x_ref_vec = x_ref[x_ref_mask > 0]
            x_flo[x_flo_mask>0] = match_histograms(x_flo_vec, x_ref_vec)

        if self.tf_params is not None:
            img = [x_ref, x_flo, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels]
            img = self.tf_params(img)
            x_ref, x_flo, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels = img


        data_dict = {
            'x_ref': x_ref, 'x_flo': x_flo,
            'x_ref_mask_init': x_ref_mask, 'x_flo_mask_init': x_flo_mask,
            'x_ref_labels': x_ref_labels, 'x_flo_labels': x_flo_labels
        }

        if self.mask_dilation:
            se = np.ones((self.mask_dilation, self.mask_dilation))
            x_ref_mask = binary_dilation(x_ref_mask, structure=se)
            x_flo_mask = binary_dilation(x_flo_mask, structure=se)

        data_dict['x_ref_mask'] = x_ref_mask
        data_dict['x_flo_mask'] = x_flo_mask

        return data_dict

    def get_intramodal_data(self, slice_ref, slice_flo, rgb=False, *args, **kwargs):

        x_ref, x_ref_mask, x_ref_labels = self.get_ref_data(slice_ref, rgb=rgb, *args, **kwargs)
        x_flo, x_flo_mask, x_flo_labels = self.get_flo_data(slice_flo, rgb=rgb, *args, **kwargs)

        if self.tf_params is not None:
            img = [x_ref, x_flo, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels]
            img_tf = self.tf_params(img)
            x_ref, x_flo, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels = img_tf

        data_dict = {
            'x_ref': x_ref, 'x_flo': x_flo,
            'x_ref_mask_init': x_ref_mask, 'x_flo_mask_init': x_flo_mask,
            'x_ref_labels': x_ref_labels, 'x_flo_labels': x_flo_labels
        }

        if self.mask_dilation:
            x_ref_mask = binary_dilation(x_ref_mask, structure=self.mask_dilation)
            x_flo_mask = binary_dilation(x_flo_mask, structure=self.mask_dilation)

        data_dict['x_ref_mask'] = x_ref_mask
        data_dict['x_flo_mask'] = x_flo_mask

        return data_dict

    def get_deformation_field(self, num=2):
        affine_list = []
        nonlinear_field_list = []
        for it_i in range(num):
            if self.affine_params is not None:  # np.eye(4)
                affine = self.affine_params.get_affine(self._image_shape)
                affine_list.append(affine)

            if self.nonlinear_params is not None:
                nlf_xyz = self.nonlinear_params.get_lowres_strength(self._image_shape)
                svf = np.concatenate([nlf[np.newaxis] for nlf in nlf_xyz], axis=0)
                nonlinear_field_list.append(svf)

        return affine_list, nonlinear_field_list

        return affine_list, nonlinear_field_list

    def convert_to_tensor(self, data_dict):

        for k, v in data_dict.items():
            if 'landmarks' in k:
                continue

            elif 'labels' in k and self.num_classes:
                v = one_hot_encoding(v, self.num_classes)
                data_dict[k] = torch.from_numpy(v).float()

            elif isinstance(v, list):
                data_dict[k] = [torch.from_numpy(vl).float() for vl in v]

            else:
                data_dict[k] = torch.from_numpy(v[np.newaxis]).float()

        return data_dict

    def __len__(self):
        if self.N is None:
            return self.N_sbj
        else:
            return self.sbj_per_epoch * self.N


class InterModalRegistrationDataset(RegistrationDataset):
    '''
    Class for intermodal registration. This class recursively creates *items* as dictionaries containing data information:
    - reference images: intensity, mask
    - target images: intensity, mask
    - spatial augmentation parameters (to be used with src.models.TensorDeformation
    '''

    def __init__(self, data_loader, affine_params, nonlinear_params, ref_modality, flo_modality, tf_params=None,
                 norm_params=None, landmarks=None, hist_match=False, mask_dilation=False, train=True,
                 num_classes=False, to_tensor=True, sbj_per_epoch=1):
        '''
        :param data_loader: dictionary whose values have methods such as load_data, load_mask etc... (e.g., Slice class)
        :param affine_params: object of class utils.transforms.AffineParams
        :param nonlinear_params: object of class utils.transforms.NonLinearParams
        :param ref_modality: 'MRI', 'LFB' or 'HE'
        :param flo_modality: 'MRI', 'LFB' or 'HE'
        :param tf_params:
        :param da_params:
        :param norm_params:
        :param landmarks:
        :param hist_match:
        :param mask_dilation:
        :param train:
        :param num_classes:
        :param to_tensor:
        :param sbj_per_epoch:
        '''

        super().__init__(data_loader, affine_params, nonlinear_params, tf_params=tf_params,
                         norm_params=norm_params, train=train, hist_match=hist_match, mask_dilation=mask_dilation,
                         to_tensor=to_tensor, num_classes=num_classes, sbj_per_epoch=sbj_per_epoch, landmarks=landmarks)

        self.ref_modality = ref_modality
        self.flo_modality = flo_modality
        self.hist_match = hist_match

    def __getitem__(self, index):

        sbj_idx, block_idx, rid_ref = index
        subject = self.data_loader[sbj_idx]
        block = subject[block_idx]
        slice_ref = block[rid_ref]
        rid = sbj_idx + '_' + block_idx + '_' + rid_ref

        data_dict = self.get_intermodal_data(slice_ref)

        affine, nonlinear_field = self.get_deformation_field(num=2)
        data_dict['affine'] = affine
        data_dict['nonlinear'] = nonlinear_field
        data_dict = self.convert_to_tensor(data_dict)

        data_dict['flip_ud'] = 0
        data_dict['flip_lr'] = 0
        data_dict['rid'] = rid
        return data_dict

    @property
    def image_shape(self):
        return self._image_shape


class IntraModalRegistrationDataset(RegistrationDataset):
    '''
    Basic class for registration where input data, output target and the correponding transformations or data
    augmentation are specified
    '''

    def __init__(self, data_loader, affine_params, nonlinear_params, modality, tf_params=None,
                 norm_params=None, hist_match=False, mask_dilation=None, to_tensor=True, landmarks=None, train=True,
                 num_classes=False, sbj_per_epoch=1, neighbor_distance=-1, fix_neighbors=False):

        '''
        :param data_loader: dictionary whose values have methods such as load_data, load_mask etc... (e.g., Slice class)
        :param affine_params: object of class utils.transforms.AffineParams
        :param nonlinear_params: object of class utils.transforms.NonLinearParams
        :param modality: 'MRI', 'LFB' or 'HE'
        '''
        super().__init__(data_loader, affine_params, nonlinear_params, tf_params=tf_params,
                         norm_params=norm_params, train=train, hist_match=hist_match, mask_dilation=mask_dilation,
                         to_tensor=to_tensor, num_classes=num_classes, sbj_per_epoch=sbj_per_epoch, landmarks=landmarks)

        self.neighbor_distance = neighbor_distance
        self.fix_neighbors = fix_neighbors
        self.ref_modality = modality
        self.flo_modality = modality

    def __getitem__(self, index):


        sbj_idx, block_idx, rid_ref, rid_flo = index
        subject = self.data_loader[sbj_idx]
        block = subject[block_idx]
        slice_ref = block[rid_ref]
        slice_flo = block[rid_flo]
        rid = sbj_idx + '_' + block_idx + '_' + rid_ref + '_to_' + rid_flo

        data_dict = self.get_intramodal_data(slice_ref, slice_flo)

        affine, nonlinear_field = self.get_deformation_field(num=2)
        data_dict['affine'] = affine
        data_dict['nonlinear'] = nonlinear_field

        data_dict = self.convert_to_tensor(data_dict)

        data_dict['rid'] = rid
        data_dict['flip_ud'] = 0
        data_dict['flip_lr'] = 0

        return data_dict

    def __len__(self):
        if self.neighbor_distance < 0 and self.fix_neighbors:
            return super().__len__() + self.neighbor_distance
        else:
            return super().__len__()

    @property
    def image_shape(self):
        return self._image_shape


class BlockInterModalRegistrationDataset(InterModalRegistrationDataset):
    '''
    Class for intermodal registration. This class recursively creates *items* as dictionaries containing data information:
    - reference images: intensity, mask
    - target images: intensity, mask
    - spatial augmentation parameters (to be used with src.models.TensorDeformation
    '''

    def __getitem__(self, rid):
        slice_ref = self.data_loader[rid]

        data_dict = self.get_intermodal_data(slice_ref)

        affine, nonlinear_field = self.get_deformation_field(num=2)
        data_dict['affine'] = affine
        data_dict['nonlinear'] = nonlinear_field
        data_dict = self.convert_to_tensor(data_dict)

        data_dict['flip_ud'] = 0
        data_dict['flip_lr'] = 0
        data_dict['rid'] = rid
        return data_dict


class BlockIntraModalRegistrationDataset(IntraModalRegistrationDataset):
    '''
    Basic class for registration where input data, output target and the correponding transformations or data
    augmentation are specified
    '''


    def __getitem__(self, index):
        rid_ref, rid_flo = index
        slice_ref = self.data_loader[rid_ref]
        slice_flo = self.data_loader[rid_flo]
        rid = rid_ref + '_to_' + rid_flo

        data_dict = self.get_intramodal_data(slice_ref, slice_flo)

        affine, nonlinear_field = self.get_deformation_field(num=2)
        data_dict['affine'] = affine
        data_dict['nonlinear'] = nonlinear_field

        data_dict = self.convert_to_tensor(data_dict)

        data_dict['rid'] = rid
        data_dict['flip_ud'] = 0
        data_dict['flip_lr'] = 0

        return data_dict

class BlockRegistrationBTDataset(object):

    # Eugenio: changed default to NOT using edge maps; the LFB grayscale and the MRI have similar intensity profiles;
    # correlation on direct image intensities works well
    def __init__(self, data_loader, use_edges=False, **kwargs):

        self.data_loader = data_loader
        self.use_edges = use_edges
        self.read_images( **kwargs)

    def process_images(self, image_dict, mask_dict, **kwargs):

        mask_th = 0.01
        edges_dict = {}
        image_list = []
        for bid in image_dict.keys():
            image = image_dict[bid]
            image = np.clip(image, 0, np.percentile(image, 99))
            mask = mask_dict[bid]
            mask = np.double(mask > mask_th*np.max(mask))
            CCmask = CC(mask, connectivity=1)
            largestCC = CCmask == np.argmax(np.bincount(CCmask.flat)[1:]) + 1
            mask = np.double(largestCC)*np.double(mask/np.max(mask))
            mask[np.isnan(mask)] = 0

            image_list.append(image[mask>mask_th].reshape((-1,)))
            mask_dict[bid] = mask

        flo_image = np.concatenate(image_list)
        ref_image = np.reshape(self.MRI[self.MRI_mask>0], (-1,))
        flo_image = match_histograms(flo_image, ref_image)

        idx = [0,0]
        for bid in image_dict.keys():
            mask = mask_dict[bid]
            image = np.zeros_like(mask)

            idx[1] += int(len(np.where(mask>mask_th)[0]))
            image[mask>0.01] = flo_image[idx[0]:idx[1]]
            image = image/(np.max(image) - np.min(image))

            image_dict[bid] = image
            if self.use_edges:
                result = grad3d(image)
                gmodule = np.sqrt(result[1]**2 + result[2]**2)
                gmodule = gmodule/np.max(gmodule)
                image = np.log(1+10*gmodule)/np.log(11)


            edges_dict[bid] = image
            idx[0] = idx[1]
            mask_dict[bid] = mask#binary_dilation(mask, structure=np.ones((7,7,1))).astype('int')

        self.images_dict = image_dict
        self.edges_dict = edges_dict
        self.mask_dict = mask_dict

    def read_images(self, **kwargs):

        self.MRI = self.data_loader.load_MRI()
        self.MRI_mask = self.data_loader.load_MRI_mask()
        self.MRI_mask_cerebellum = self.data_loader.load_MRI_mask_cerebellum()
        self.MRI_mask_cerebrum = self.data_loader.load_MRI_mask_cerebrum()
        self.MRI_mask_brainstem = self.data_loader.load_MRI_mask_brainstem()

        self.MRI_mask_dilated = self.data_loader.load_MRI_mask_dilated()

        images_dict = {}
        mask_dict = {}
        for bid, block in self.data_loader.subject_dict.items():
            image = block.load_gray_slices()
            mask = block.load_mask_slices()
            image[np.isnan(mask)] = 0
            mask[np.isnan(mask)] = 0

            images_dict[bid] = image
            mask_dict[bid] = mask

        self.process_images(images_dict, mask_dict, **kwargs)

        if self.use_edges:
            result = grad3d(self.MRI)
            MRI = result[0]
            MRI = MRI/np.max(MRI)
            MRI = np.log(1+10*MRI) / np.log(11)
            self.MRI = MRI

    def block_header(self, bid = None):
        return self.data_loader.subject_dict[bid].affine

    def update_edges(self, image, bid):

        if self.use_edges:
            result = grad3d(image)
            gmodule = np.sqrt(result[1] ** 2 + result[2] ** 2)
            gmodule = gmodule / np.max(gmodule)
            image = np.log(1 + 10 * gmodule) / np.log(11)

        self.edges_dict[bid] = image

    @property
    def headers_dict(self):
        header_dict = {}
        for bid, block in self.data_loader.subject_dict.items():
            header_dict[bid] = block.affine

        return header_dict

    def __getitem__(self, bid):
        image = torch.tensor(np.reshape(self.edges_dict[bid],(1,1) + self.edges_dict[bid].shape), dtype=torch.float)
        mask = torch.tensor(np.reshape(self.mask_dict[bid],(1,1) + self.mask_dict[bid].shape), dtype=torch.float)
        header = torch.tensor(self.block_header(bid), dtype=torch.float)

        return image, mask, header