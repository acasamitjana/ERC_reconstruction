import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_filter(filt_size=3):

    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


#########################################
############ Learning layers ############
#########################################
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm_layer='none', activation='relu', use_bias=True):
        super(LinearBlock, self).__init__()

        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        self.initialize_normalization(norm_layer, norm_dim=output_dim)
        self.initialize_activation(activation)


    def initialize_normalization(self, norm_layer, norm_dim):
        if norm_layer == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm_layer == 'none' or norm_layer == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm_layer)

    def initialize_activation(self, activation):
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class BaseConvBlock2D(nn.Module):

    def initialize_padding(self, pad_type, padding):
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zeros':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

    def initialize_normalization(self, norm_layer, norm_dim):
        if norm_layer == 'bn':
            self.norm_layer = nn.BatchNorm2d(norm_dim)
        elif norm_layer == 'in':
            self.norm_layer = nn.InstanceNorm2d(norm_dim, affine=False)
        elif norm_layer == 'none' or norm_layer == 'sn':
            self.norm_layer = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm_layer)

    def initialize_activation(self, activation):
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

class ConvBlock2D(BaseConvBlock2D):
    '''
    2D ConvlutionBlock performing the following operations:
        Conv2D --> BatchNormalization -> Activation function
    :param Conv2D input parameters: see nn.Conv2D
    :param norm_layer (None, PyTorch normalization layer): it can be either None if no normalization is applied or a
    Pytorch normalization layer (nn.BatchNorm2d, nn.InstanceNorm2d)
    :param activation (None or PyTorch activation): it can be either None for linear activation or any other activation
    in PyTorch (nn.ReLU, nn.LeakyReLu(alpha), nn.Sigmoid, ...)
    '''

    def __init__(self, input_filters, output_filters, kernel_size=3, padding=0, stride=1, bias=True,
                 norm_layer='bn', activation='relu', pad_type='zeros'):

        super().__init__()
        # initialize padding
        self.initialize_padding(pad_type, padding)
        self.initialize_normalization(norm_layer,norm_dim=output_filters)
        self.initialize_activation(activation)
        self.conv_layer = nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size,  stride=stride, bias=bias)


    def forward(self, inputs):
        outputs = self.conv_layer(self.pad(inputs))
        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

class ConvTransposeBlock2D(BaseConvBlock2D):
    '''
    2D ConvTransposeBlock2D performing the following operations:
        Conv2D --> BatchNormalization -> Activation function
    :param ConvTranspose2D input parameters: see nn.ConvTranspose2d
    :param norm_layer (None, PyTorch normalization layer): it can be either None if no normalization is applied or a
    Pytorch normalization layer (nn.BatchNorm2d, nn.InstanceNorm2d)
    :param activation (None or PyTorch activation): it can be either None for linear activation or any other activation
    in PyTorch (nn.ReLU, nn.LeakyReLu(alpha), nn.Sigmoid, ...)
    '''

    def __init__(self, input_filters, output_filters, kernel_sizeT=4, kernel_size=3, output_padding=0, padding=0,
                 stride=2, bias=True, norm_layer='bn', activation='relu', pad_type='zeros'):

        super().__init__()
        self.initialize_padding(pad_type, padding, int(np.floor((kernel_size-1)/2)))
        self.initialize_normalization(norm_layer,norm_dim=output_filters)
        self.initialize_activation(activation)

        self.convT_layer = nn.ConvTranspose2d(input_filters, input_filters, kernel_size=kernel_sizeT,
                                              output_padding=output_padding, stride=stride, bias=bias)

        self.conv_layer = nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size,
                                    stride=1, bias=bias)


    def initialize_padding(self, pad_type, padding1, padding2):
        # initialize padding
        if pad_type == 'reflect':
            self.pad1 = nn.ReflectionPad2d(padding1)
            self.pad2 = nn.ReflectionPad2d(padding2)

        elif pad_type == 'replicate':
            self.pad1 = nn.ReplicationPad2d(padding1)
            self.pad2 = nn.ReplicationPad2d(padding2)

        elif pad_type == 'zeros':
            self.pad1 = nn.ZeroPad2d(padding1)
            self.pad2 = nn.ZeroPad2d(padding2)

        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)


    def forward(self, inputs):
        outputs = self.convT_layer(self.pad1(inputs))
        outputs = self.conv_layer(self.pad2(outputs))

        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]

class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

###############################################
############ Transformation layers ############
###############################################
class AffineTransformer(nn.Module):
    def __init__(self, vol_shape, input_channels, enc_features):
        super(AffineTransformer, self).__init__()

        # Spatial transformer localization-network
        out_shape = [v for v in vol_shape]
        nf_list = [input_channels] + enc_features
        localization_layers = []
        for in_nf, out_nf in zip(nf_list[:-1], nf_list[1:]):
            localization_layers.append(nn.Conv2d(in_nf, out_nf, kernel_size=3, stride=2, padding=1))
            localization_layers.append(nn.LeakyReLU(0.2))
            out_shape = [o/2 for o in out_shape]

        self.localization = nn.Sequential(*localization_layers)
        self.out_shape = int(enc_features[-1]*np.prod(out_shape))

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.out_shape, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 3 * 2)
        )

    # Spatial transformer network forward function
    def forward(self, x):
        x_floating = x[:,0:1]
        xs = self.localization(x)
        xs = xs.view(-1, self.out_shape)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x_floating.size())

        return F.grid_sample(x_floating, grid), theta

class SpatialInterpolation(nn.Module):
    """
    [SpatialInterpolation] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample

    This is copied from voxelmorph code, so for more information and credit
    visit https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """

    def __init__(self, mode='bilinear', padding_mode='zeros'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super().__init__()

        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, src, new_locs, **kwargs):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        if 'padding_mode' in kwargs:
            self.padding_mode = kwargs['padding_mode']
        if 'mode' in kwargs:
            self.mode = kwargs['mode']

        shape = src.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode=self.padding_mode, align_corners=True)

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample

    This is copied from voxelmorph code, so for more information and credit
    visit https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """

    def __init__(self, size, mode='bilinear', padding_mode='border', torch_dtype=torch.float):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super().__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s).type(torch_dtype) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch_dtype)

        self.register_buffer('grid', grid)

        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, src, flow, **kwargs):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        padding_mode = kwargs['padding_mode'] if 'padding_mode' in kwargs else self.padding_mode
        mode = kwargs['mode'] if 'mode' in kwargs else self.mode

        new_locs = self.grid + flow
        # new_locs = self.grid.to("cuda:0") + flow # TODO: Eugenio: must be a better way, I shouldn't need this...

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=mode, padding_mode=padding_mode, align_corners=True)

class SpatialTransformerAffine(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample

    This is copied from voxelmorph code, so for more information and credit
    visit https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """

    def __init__(self, size, mode='bilinear', padding_mode='border', torch_dtype = torch.float):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super().__init__()

        ndims = len(size)

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = grid.type(torch_dtype)


        flat_mesh = torch.reshape(grid, (ndims,-1))
        ones_vec = torch.ones((1, np.prod(size))).type(torch_dtype)
        mesh_matrix = torch.cat((flat_mesh, ones_vec), dim=0)

        # grid = torch.unsqueeze(grid, 0)  # add batch
        # grid = grid.type(torch_dtype)
        # self.register_buffer('grid', grid)

        mesh_matrix = mesh_matrix.type(torch_dtype)
        self.register_buffer('mesh_matrix', mesh_matrix)

        self.size = size
        self.mode = mode
        self.padding_mode = padding_mode
        self.torch_dtype = torch_dtype

    def _get_locations(self, affine_matrix):
        batch_size = affine_matrix.shape[0]
        ndims = len(self.size)
        vol_shape = self.size


        # compute locations
        loc_matrix = torch.matmul(affine_matrix, self.mesh_matrix)  # N x nb_voxels
        loc = torch.reshape(loc_matrix[:,:ndims], [batch_size, ndims] + list(vol_shape))  # *volshape x N

        return loc.float()

    def forward(self, src, affine_matrix, **kwargs):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image of size [batch_size, n_dims, *volshape]
            :param flow: the output from the U-Net [batch_size, n_dims, *volshape]
        """

        padding_mode = kwargs['padding_mode'] if 'padding_mode' in kwargs else self.padding_mode
        mode = kwargs['mode'] if 'mode' in kwargs else self.mode

        affine_matrix = affine_matrix.type(self.torch_dtype)

        new_locs = self._get_locations(affine_matrix)
        new_locs = new_locs.type(self.torch_dtype)

        if 'shape' in kwargs.keys():
            shape = kwargs['shape']
        else:
            shape = new_locs.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]


        return F.grid_sample(src, new_locs, mode=mode, padding_mode=padding_mode)

class VecInt(nn.Module):
    """
    Vector Integration Layer

    Enables vector integration via several methods
    (ode or quadrature for time-dependent vector fields,
    scaling and squaring for stationary fields)

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(self, field_shape, int_steps=7, int_end=1,**kwargs):
        """
        Parameters:
            int_steps is the number of integration steps
        """
        super().__init__()
        self.int_steps = int_steps
        self.int_end = int_end
        self.scale = 1 / (2 ** self.int_steps) * int_end
        self.transformer = SpatialTransformer(field_shape)

    def forward(self, field, **kwargs):


        output = field * self.scale

        nsteps = self.int_steps
        if 'nsteps' in kwargs:
            nsteps = nsteps - kwargs['nsteps']

        for _ in range(nsteps):
            a = self.transformer(output, output)
            output = output + a

        return output

class RescaleTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    Credit to voxelmorph: https://github.com/voxelmorph/voxelmorph/blob/redesign/voxelmorph/torch/layers.py
    """

    def __init__(self, inshape, factor=None, target_size=None):
        '''

        :param vol_size:
        :param factor:
                :param latent_size: it only applies if factor is None

        '''
        super().__init__()

        self.ndims = len(inshape)
        self.mode = 'linear'
        self.inshape = inshape

        if factor is None:
            assert target_size is not None
            self.factor = tuple([b/a for a, b in zip(inshape, target_size)])
        elif isinstance(factor, list) or isinstance(factor, tuple):
            self.factor = list(factor)
        else:
            self.factor = [factor for _ in range(self.ndims)]

        if self.ndims == 2:
            self.mode = 'bi' + self.mode
        elif self.ndims == 3:
            self.mode = 'tri' + self.mode

        if self.factor[0] < 1:
            kernel_sigma = [0.44 * 1 / f for f in self.factor]
            kernel = self.gaussian_filter_2d(kernel_sigma=kernel_sigma)

            self.register_buffer('kernel', kernel)

    def gaussian_filter_2d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))


        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX = np.meshgrid(coord[0], coord[1], indexing='ij')
        xy_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1,1,1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1,1,1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)

        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xy_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1])

        # Total kernel
        total_kernel = np.zeros((2, 2) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel

        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def forward(self, x):

        x = x.clone()
        if self.factor[0] < 1:
            padding = [int((s - 1) // 2) for s in self.kernel.shape[2:]]
            x = F.conv2d(x, self.kernel, stride=(1, 1), padding=padding)

            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            for i in range(self.ndims):
                x[:, i] = x[:, i] * self.factor[i]

        elif self.factor[0] > 1:
            # multiply first to save memory
            for i in range(self.ndims):
                x[:, i] = x[:, i] * self.factor[i]
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    Credit to voxelmorph: https://github.com/voxelmorph/voxelmorph/blob/redesign/voxelmorph/torch/layers.py
    """

    def __init__(self, inshape, target_size=None, factor=None, gaussian_filter_flag=True):
        '''

        :param vol_size:
        :param factor: if factor<1 the shape is reduced and viceversa.
        :param latent_size: it only applies if factor is None
        '''
        super().__init__()

        self.ndims = len(inshape)
        self.mode = 'linear'
        self.inshape = inshape
        self.gaussian_filter_flag = gaussian_filter_flag
        if self.ndims == 2:
            self.mode = 'bi' + self.mode
        elif self.ndims == 3:
            self.mode = 'tri' + self.mode

        if target_size is None:
            self.factor = factor
            if isinstance(factor, float) or isinstance(factor, int):
                self.factor = [factor for _ in range(self.ndims)]
        else:
            self.factor = tuple([b/a for a, b in zip(inshape, target_size)])

        if self.gaussian_filter_flag:

            kernel_sigma = [0.44 * f for f in self.factor]
            kernel = self.gaussian_filter_2d(kernel_sigma=kernel_sigma)

            self.register_buffer('kernel', kernel)

    def gaussian_filter_2d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))


        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX = np.meshgrid(coord[0], coord[1], indexing='ij')
        xy_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1,1,1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1,1,1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)

        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xy_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1])

        # Total kernel
        total_kernel = np.zeros((2, 2) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel

        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def forward(self, x):

        x = x.clone()
        if self.gaussian_filter_flag and self.factor[0] < 1:
            padding = [int((s - 1) // 2) for s in self.kernel.shape[2:]]
            x = F.conv2d(x, self.kernel, stride=(1, 1), padding=padding)

        x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x


