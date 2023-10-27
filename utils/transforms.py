import copy

import numpy as np

# ----------------- #
# Parameter classes #
# ----------------- #

class ResizeParams(object):
    def __init__(self, resize_shape):
        if isinstance(resize_shape, int):
            resize_shape = (resize_shape, resize_shape)
        self.resize_shape = resize_shape

class CropParams(object):
    def __init__(self, crop_shape, init_coordinates=None):
        if isinstance(crop_shape, int):
            crop_shape = (crop_shape, crop_shape)
        self.crop_shape = crop_shape
        self.init_coordinates = init_coordinates

class AffineParams(object):

    def __init__(self, rotation, scaling, translation):
        self.rotation = rotation
        self.scaling = scaling
        self.translation = translation

    def get_angles(self):
        angles = []
        for r in self.rotation:
            high, low = r, -r
            angles.append((np.random.rand(1) * (high - low) + low)/180*np.pi)

        return angles

    def get_scaling(self):
        angles = []
        for r in self.scaling:
            high, low = r, -r
            angles.append(np.random.rand(1) * (high - low) + low)

        return angles

    def get_translation(self):
        angles = []
        for r in self.translation:
            high, low = r, -r
            angles.append(np.random.rand(1) * (high - low) + low)

        return angles

    def get_affine(self, image_shape):
        if len(image_shape) == 2:
            return self._get_affine_2d(image_shape)
        else:
            return self._get_affine_3d(image_shape)

    def _get_affine_2d(self, image_shape):
        T1 = np.eye(3)
        T2 = np.eye(3)
        T3 = np.eye(3)
        T4 = np.eye(3)
        T5 = np.eye(3)

        cr = [i/2 for i in image_shape]
        scaling = self.get_scaling()
        angles = self.get_angles()
        translation = self.get_translation()

        T1[0, 2] = -cr[0]
        T1[1, 2] = -cr[1]

        T2[0, 0] += scaling[0]
        T2[1, 1] += scaling[1]

        T3[0, 0] = np.cos(angles[0])
        T3[0, 1] = -np.sin(angles[0])
        T3[1, 0] = np.sin(angles[0])
        T3[1, 1] = np.cos(angles[0])

        T4[0, 2] = cr[0]
        T4[1, 2] = cr[1]

        T5[0, 2] = translation[0]
        T5[1, 2] = translation[1]

        return T5 @ T4 @ T3 @ T2 @ T1

    def _get_affine_3d(self, image_shape):
        T1 = np.eye(4)
        T2 = np.eye(4)
        T3 = np.eye(4)
        T4 = np.eye(4)
        T5 = np.eye(4)
        T6 = np.eye(4)
        T7 = np.eye(4)

        cr = [i/2 for i in image_shape]
        scaling = self.get_scaling()
        angles = self.get_angles()
        translation = self.get_translation()

        T1[0, 3] = -cr[0]
        T1[1, 3] = -cr[1]
        T1[2, 3] = -cr[2]

        T2[0, 0] += scaling[0]
        T2[1, 1] += scaling[1]
        T2[2, 2] += scaling[2]

        T3[1, 1] = np.cos(angles[0])
        T3[1, 2] = -np.sin(angles[0])
        T3[2, 1] = np.sin(angles[0])
        T3[2, 2] = np.cos(angles[0])

        T4[0, 0] = np.cos(angles[1])
        T4[0, 2] = np.sin(angles[1])
        T4[2, 0] = -np.sin(angles[1])
        T4[2, 2] = np.cos(angles[1])

        T5[0, 0] = np.cos(angles[2])
        T5[0, 1] = -np.sin(angles[2])
        T5[1, 0] = np.sin(angles[2])
        T5[1, 1] = np.cos(angles[2])

        T6[0, 3] = cr[0]
        T6[1, 3] = cr[1]
        T6[2, 3] = cr[2]

        T7[0, 3] = translation[0]
        T7[1, 3] = translation[1]
        T7[2, 3] = translation[2]

        return T7 @ T6 @ T5 @ T4 @ T3 @ T2 @ T1

class PadParams(object):
    def __init__(self, psize, pfill=0, pmode='constant', dim=2):
        if isinstance(psize, int):
            psize = (psize, psize)
        self.psize = psize
        self.pmode = pmode
        self.pfill = pfill
        self.dim = dim

class NonLinearParams(object):
    def __init__(self, lowres_strength=1, lowres_shape_factor=0.04, distribution='normal', nstep=5):
        if isinstance(lowres_strength, int):
            self.lowres_strength = [lowres_strength, lowres_strength]

        self.lowres_shape_factor = lowres_shape_factor
        self.distribution = distribution
        self.nstep = nstep

    def get_lowres_size(self, image_shape):
        return tuple([int(s*self.lowres_shape_factor) for s in image_shape])

    def get_lowres_strength(self, image_shape):

        lowres_size = self.get_lowres_size(image_shape)

        ndim = len(image_shape)
        if self.distribution == 'normal':
            mean, std = self.lowres_strength[1], self.lowres_strength[0]
            lowres_strength = np.random.randn(1) * std + mean

        elif self.distribution == 'uniform':
            high, low = self.lowres_strength[1], self.lowres_strength[0]
            lowres_strength = np.random.rand(1) * (high - low) + low

        elif self.distribution == 'lognormal':
            mean, std = self.lowres_strength[1], self.lowres_strength[0]
            lowres_strength = np.random.randn(1) * std + mean
            lowres_strength = np.exp(lowres_strength)

        elif self.distribution is None:
            lowres_strength = [self.lowres_strength]

        else:
            raise ValueError("[src/utils/transformations: NonLinearDeformation]. Please, specify a valid distribution "
                             "for the low-res nonlinear distribution")

        if ndim ==2:
            field_lowres_x = lowres_strength * np.random.randn(lowres_size[0],
                                                               lowres_size[1])  # generate random noise.

            field_lowres_y = lowres_strength * np.random.randn(lowres_size[0],
                                                               lowres_size[1])  # generate random noise.

            return field_lowres_x, field_lowres_y

        else:
            field_lowres_x = lowres_strength * np.random.randn(lowres_size[0],
                                                               lowres_size[1],
                                                               lowres_size[2])  # generate random noise.

            field_lowres_y = lowres_strength * np.random.randn(lowres_size[0],
                                                               lowres_size[1],
                                                               lowres_size[2])  # generate random noise.

            field_lowres_z = lowres_strength * np.random.randn(lowres_size[0],
                                                               lowres_size[1],
                                                               lowres_size[2])  # generate random noise.

            return field_lowres_x, field_lowres_y, field_lowres_z



# ------------------ #
# Compose transforms #
# ------------------ #

class Compose(object):
    def __init__(self, transform_parameters):

        self.transform_parameters = transform_parameters if transform_parameters is not None else []
        self.img_shape = None

    def _compute_data_shape(self, init_shape):

        if isinstance(init_shape, list):
            n_shape = len(init_shape)
            final_shape = init_shape

        else:
            n_shape = 1
            final_shape = [init_shape]

        for t in self.transform_parameters:
            if isinstance(t, CropParams):
                final_shape = [t.crop_shape] * n_shape

            elif isinstance(t, PadParams):
                if t.psize is None:
                    final_shape = init_shape
                    #     psize = max([max([di.size for di in d]) for d in self.data])
                    #     t.psize = (1 << (psize[0] - 1).bit_length(), 1 << (psize[1] - 1).bit_length())
                else:
                    final_shape = [t.psize] * n_shape

            elif isinstance(t, ResizeParams):
                final_shape = [t.resize_shape] * n_shape

            else:
                raise ValueError(
                    str(type(t)) + 'is not a valid type for transformation. Please, specify a valid one')

        if isinstance(init_shape, list):
            return tuple(final_shape)
        else:
            return final_shape[0]

    def __call__(self, img):

        img_shape = [i.shape for i in img]

        for t in self.transform_parameters:
            if isinstance(t, CropParams):
                tf = RandomCropManyImages(t)
                img = tf(img)

            elif isinstance(t, PadParams):
                img = [Padding(t, i.shape)(i) for i in img]

            else:
                raise ValueError(
                    str(type(t)) + 'is not a valid type for transformation. Please, specify a valid one')

        self.img_shape = img_shape

        return img

    def inverse(self, img, img_shape=None):

        if img_shape is None:
            if self.img_shape is None:
                raise ValueError("You need to provide the initial image shape or call the forward transform function"
                                 "before calling the inverse")
            else:
                img_shape = self.img_shape


        for t in self.transform_parameters:
            if isinstance(t, CropParams):
                tf = RandomCropManyImages(t)
                img = tf.inverse(img, img_shape)

            elif isinstance(t, PadParams):
                img = [Padding(t, i.shape).inverse(i, img_shape) for i in img]

            else:
                raise ValueError(
                    str(type(t)) + 'is not a valid type for transformation. Please, specify a valid one')

        return img

class Compose_DA(object):
    def __init__(self, data_augmentation_parameters):
        self.data_augmentation_parameters = data_augmentation_parameters if data_augmentation_parameters is not None else []

    def __call__(self, img, mask_flag = None, **kwargs):
        '''
        Mask flag is used to indicate which elements of the list are not used in intensity-based transformations, and
        only in deformation-based transformations.
        '''

        islist = True
        if not isinstance(img, list):
            img = [img]
            islist = False

        if mask_flag is None:
            mask_flag = [False] * len(img)
        elif not isinstance(mask_flag, list):
            mask_flag = [mask_flag] * len(img)


        for da in self.data_augmentation_parameters:

            if isinstance(da, NonLinearParams):
                tf = NonLinearDifferomorphismManyImages(da)
                img = tf(img, mask_flag)
                continue

            else:
                raise ValueError(str(type(da)) + 'is not a valid type for data augmentation. Please, specify a valid one')

            img_tf = tf([i for i, m in zip(img, mask_flag) if not m])

            it_img_tf = 0
            for it_img in range(len(img)):
                if not mask_flag[it_img]:
                    img[it_img] = img_tf[it_img_tf]
                    it_img_tf += 1

        if not islist:
            img = img[0]

        return img


# ---------- #
# Transforms #
# ---------- #
class NormalNormalization(object):
    def __init__(self, mean = 0, std = 1, dim = None, inplace = False):

        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.dim = None

    def __call__(self, data, *args, **kwargs):
        if not self.inplace:
            data = copy.deepcopy(data)

        mean_d = np.mean(data, axis = self.dim)
        std_d = np.std(data, axis = self.dim)

        assert len(mean_d) == self.mean

        d_norm = (data - mean_d) / std_d
        out_data = (d_norm + self.mean) * self.std

        return out_data

class ScaleNormalization(object):
    def __init__(self, scale=1.0, range = None, percentile=None):

        self.scale = scale
        self.range = range
        self.percentile = percentile

    def __call__(self, data, *args, **kwargs):

        if self.range is not None:
            if self.percentile is not None:
                dmax = np.percentile(data,self.percentile)
                dmin = np.percentile(data,100-self.percentile)

            else:
                dmax = np.max(data)
                dmin = np.min(data)

            data = (data - dmin) / (dmax-dmin) * (self.range[1] - self.range[0]) + self.range[0]
        else:
            data = data * self.scale

        return data

class Padding(object):
    def __init__(self, parameters, isize, dim=2):


        if len(isize) > dim+1:
            raise ValueError("Please, specify a valid dimension and size")

        osize = parameters.psize
        assert len(osize) == dim

        pfill = parameters.pfill
        pmode = parameters.pmode

        psize = []
        for i, o in zip(isize, osize):
            if o - i > 0:
                pfloor = int(np.floor((o - i) / 2))
                pceil = pfloor if np.mod(o - i, 2) == 0 else pfloor + 1
            else:
                pfloor = 0
                pceil = 0

            psize.append((pfloor, pceil))

        pad_tuple = psize

        self.padding = pad_tuple
        self.fill = pfill
        self.padding_mode = pmode
        self.dim = dim
        self.osize = osize

    def __call__(self, data):

        if len(data.shape) == self.dim+1:
            nchannels = data.shape[-1]
            output_data = np.zeros(self.osize + (nchannels, ))
            for idim in range(nchannels):
                output_data[..., idim] = np.pad(data[..., idim], pad_width=self.padding, mode=self.padding_mode,
                                                constant_values=self.fill)
            return output_data
        else:
            return np.pad(data, pad_width=self.padding, mode=self.padding_mode, constant_values=self.fill)

    def inverse(self, img, img_shape):

        oshape = img.shape
        sshape = []
        for o, i in zip(oshape, img_shape):
            d = o - i
            d1 = int(np.floor(d / 1))
            d2 = d - d1
            sshape.append(slice(d1, o - d2))
        a = img[sshape[0], sshape[1]]
        return a

class RandomCropManyImages(object):
    """Crop the given numpy array at a random location.
    Images are cropped at from the center as follows:


    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (d1, d2, ... , dN), a square crop (size, size, ..., size) is
            made.

        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding.
    """

    def __init__(self, parameters, pad_if_needed=True, fill=0, padding_mode='constant'):

        self.parameters = parameters
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def get_params(self, data_shape, output_shape):

        if all([a==b for a,b in zip(data_shape, output_shape)]):
            return [0]*len(data_shape), data_shape

        if self.parameters.init_coordinates is None:
            init_coordinates = []
            for a, b in zip(data_shape, output_shape):
                init_coordinates.append(int((a-b)//2))
        else:
            init_coordinates = self.parameters.init_coordinates

        return init_coordinates, output_shape

    def __call__(self, data_list):
        """
        Args:
            data_list : list of numpy arrays. Each numpy array has the following size: (num_channels, d1, ..., dN)

        Returns:
            output_list: list of cropped numpy arrays.
        """
        size = self.parameters.crop_shape
        n_dims = len(size)
        padded_data_list = []
        for i in range(len(data_list)):
            data = data_list[i]
            # pad the width if needed
            pad_width = []
            for it_dim in range(n_dims):
                if self.pad_if_needed and data.shape[it_dim] < size[it_dim]:
                    pad0 = (size[it_dim] - data.shape[it_dim]) // 2
                    pad_width.append((pad0, size[it_dim] - data.shape[it_dim]-pad0))
                else:
                    pad_width.append((0,0))
            data = np.pad(data, pad_width=pad_width, mode=self.padding_mode, constant_values=self.fill)
            padded_data_list.append(data)

        init_coord, output_shape = self.get_params(padded_data_list[0].shape, size)

        self.init_coord = init_coord
        self.output_shape = output_shape

        output = []
        for i in range(len(padded_data_list)):
            padded_data = padded_data_list[i]
            for it_dim in range(n_dims):
                idx = (slice(None),) * (it_dim) + \
                      (slice(init_coord[it_dim], init_coord[it_dim] + output_shape[it_dim], 1), )
                padded_data = padded_data[idx]
            output.append(padded_data)

        return output

    def inverse(self, data_list, data_shape):
        size = self.parameters.crop_shape
        n_dims = len(size)

        cropped_data_list = []
        for data, dshape in zip(data_list, data_shape):
            cropped_data = data
            for it_dim in range(n_dims):
                init_coord = (size[it_dim] - dshape[it_dim]) // 2
                if init_coord < 0:
                    init_coord = 0

                idx = (slice(None),) * (it_dim) + (slice(init_coord, init_coord + dshape[it_dim], 1),)
                cropped_data = cropped_data[idx]
            cropped_data_list.append(cropped_data)

        init_coord, _ = self.get_params(data_shape[0], size)

        output = []
        for data, dshape in zip(cropped_data_list, data_shape):
            pad_width = []
            for it_dim in range(n_dims):
                if size[it_dim] < dshape[it_dim]:
                    pad_width.append((int(init_coord[it_dim]), int(dshape[it_dim] - size[it_dim] - init_coord[it_dim])))
                else:
                    pad_width.append((0, 0))

            data = np.pad(data, pad_width=pad_width, mode=self.padding_mode, constant_values=self.fill)
            output.append(data)

        return output

