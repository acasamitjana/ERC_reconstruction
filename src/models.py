import math
import functools
from os.path import exists

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.autograd import Variable

from src.layers import ConvBlock2D, SpatialTransformer, VecInt, ResizeTransform, SpatialTransformerAffine, \
    RescaleTransform, Normalize, ResnetBlock, Upsample, Downsample

class BaseModel(nn.Module):
    pass

class Init_net(object):


    def __init__(self):
        pass

    def init_net(self, net, init_type='normal', init_gain=0.02, device='cpu', gpu_ids=[]):
        """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
        Parameters:
            net (network)      -- the network to be initialized
            init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)       -- scaling factor for normal, xavier and orthogonal.
            gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        Return an initialized network.
        """
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        self._init_weights(net, init_type, init_gain=init_gain)
        net = net.to(device)

        return net

    def _init_weights(self, net, init_type='normal', init_gain=0.02):
        """Initialize network weights.
        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                nn.init.normal_(m.weight.data, 1.0, init_gain)
                nn.init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>


class TensorDeformation(object):

    def __init__(self, image_shape, nonlinear_field_size, device):

        self.image_shape = image_shape
        self.imshape_half = tuple([v // 2 for v in self.image_shape])
        self.ndims = len(image_shape)
        self.device = device

        model = {}
        model['resize_half'] = ResizeTransform(inshape=nonlinear_field_size, target_size=self.imshape_half)
        model['vecInt'] = VecInt(self.imshape_half)
        model['resize_full'] = ResizeTransform(inshape=self.imshape_half, target_size=self.image_shape)
        model['warper'] = SpatialTransformer(image_shape, padding_mode='zeros')
        for m in model.values():
            m.requires_grad_(False)
            m.to(device)
            m.eval()

        self.model = model

        vectors = [torch.arange(0, s) for s in self.image_shape]
        grids = torch.meshgrid(vectors)
        self.grid = torch.stack(grids).to(device)  # y, x, z
        self.ones_vec = torch.ones((1, np.prod(self.image_shape)), device=self.device)


    def get_nonlin_field(self, field):
        return self.model['resize_full'](self.model['vecInt'](self.model['resize_half'](field))).float()

    def get_lin_field(self, affine):

        batch_size = affine.shape[0]

        ones_vec = torch.ones((batch_size, 1, np.prod(self.image_shape)), device=self.device)
        flat_mesh = torch.reshape(self.grid, (batch_size, self.ndims,-1))
        mesh_matrix = torch.cat((flat_mesh, ones_vec), dim=0)

        loc_matrix = torch.matmul(affine, mesh_matrix)  # N x nb_voxels
        loc = torch.reshape(loc_matrix[:,:self.ndims], [batch_size, self.ndims] + list(self.image_shape))  # *volshape x N

        return loc - mesh_matrix

    def get_lin_nonlin_field(self, affine, field):
        deformation = self.model['resize_full'](self.model['vecInt'](self.model['resize_half'](field))).float()

        batch_size = affine.shape[0]

        def_list = torch.unbind(deformation, dim=0)
        flat_mesh_list = [torch.reshape(self.grid + d, (self.ndims, -1)) for d in def_list]
        mesh_matrix_list = [torch.cat((flat_mesh, self.ones_vec), dim=0) for flat_mesh in flat_mesh_list]
        mesh_matrix = torch.stack(mesh_matrix_list, dim=0)
        loc_matrix = torch.matmul(affine, mesh_matrix)  # N x nb_voxels
        loc = torch.reshape(loc_matrix[:,:self.ndims], [batch_size, self.ndims] + list(self.image_shape))  # *volshape x N

        return loc - self.grid

    def transform(self, image, affine=None, low_res_nonfield=None, **kwargs):

        if affine is not None and low_res_nonfield is not None:
            deformation = self.get_lin_nonlin_field(affine, low_res_nonfield)

        elif affine is not None:
            deformation = self.get_lin_field(affine)

        elif low_res_nonfield is not None:
            deformation = self.get_nonlin_field(low_res_nonfield)

        else:
            return image

        image = self.model['warper'](image, deformation, **kwargs)

        return image


#############
# UNet like #
#############

class ResnetGenerator(BaseModel):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None,
                 n_downsampling=2, tanh=True):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()] if tanh else [nn.Sigmoid()]

        self.model = nn.Sequential(*model)


    def init_net(self, device, weightsfile=None, model_key = None, init_type='xavier', init_gain=0.02):
        init_net = Init_net()

        if weightsfile is not None:

            model_key = '' if model_key is None else '_' + model_key
            self.model = self.model.to(device)
            checkpoint = torch.load(weightsfile, map_location=device)
            self.load_state_dict(checkpoint['state_dict' + model_key])

        else:
            self.model = init_net.init_net(self.model, init_type, init_gain, device=device)


    def forward(self, input, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake


class Unet(BaseModel):
    """
    Voxelmorph Unet. For more information see voxelmorph.net
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1, activation='lrelu',
                 cpoints_level=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = self._default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        nlayers_uparm = len(self.enc_nf) - int(np.log2(cpoints_level))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock2D(prev_nf, nf, stride=2, activation=activation, norm_layer='none', padding=1))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:nlayers_uparm]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock2D(channels, nf, stride=1, activation=activation, norm_layer='none', padding=1))

            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf = prev_nf + 2 if cpoints_level == 1 else prev_nf + enc_history[nlayers_uparm]
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock2D(prev_nf, nf, stride=1, activation=activation, norm_layer='none', padding=1))
            prev_nf = nf

    def init_net(self, device, weightsfile=None, init_type='xavier', init_gain=0.02):
        init_net = Init_net()

        if weightsfile is not None:
            self.upsample = self.upsample.to(device)
            self.uparm = self.uparm.to(device)
            self.downarm = self.downarm.to(device)
            self.extras = self.extras.to(device)
            checkpoint = torch.load(weightsfile, map_location=device)
            self.load_state_dict(checkpoint['state_dict'])

        else:
            self.upsample = self.upsample.to(device)
            self.uparm = init_net.init_net(self.uparm, device=device)
            self.downarm = init_net.init_net(self.downarm, device=device)
            self.extras = init_net.init_net(self.extras, init_type, init_gain, device=device)

    def _default_unet_features(self):
        nb_features = [
            [16, 32, 32, 32],  # encoder
            [32, 32, 32, 32, 32, 16, 16]  # decoder
        ]
        return nb_features

    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class PatchSampleF(BaseModel):
    def __init__(self, use_mlp=False, nc=256, init_type='normal', init_gain=0.02, device='cpu'):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.device = device

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc, bias=True), nn.ReLU(True), nn.Linear(self.nc, self.nc)])
            setattr(self, 'mlp_%d' % mlp_id, mlp)

        init_net = Init_net()
        init_net.init_net(self, self.init_type, self.init_gain, device=self.device, init_bias=0.0001)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None, mask_sampling=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)

        for feat_id, feat in enumerate(feats):

            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            if B>1 and mask_sampling is not None:
                raise ValueError('Mask sampling is only available for batch_size=1')
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    if mask_sampling is None:
                        patch_id = torch.randperm(feat_reshape.shape[1], device=self.device)
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                    else:
                        mask = mask_sampling[feat_id]
                        mask_reshape = mask.permute(0, 2, 3, 1).flatten(1, 2)
                        idx = torch.nonzero(mask_reshape[0,:,0])[:, 0]
                        idx_perm = torch.randperm(idx.shape[0], device=self.device)
                        idx_perm = idx_perm[:int(min(num_patches, idx_perm.shape[0]))]
                        patch_id = idx[idx_perm]
                        # patch_id = torch.randperm(feat_reshape.shape[1], device=self.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # shape: patches X channels
            else:
                x_sample = feat_reshape
                patch_id = []

            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                # print('MLP')
                # print(x_sample.mean())
                # print([a + '_' + str(b.mean()) for a,b in mlp.named_parameters()])
                x_sample = mlp(x_sample)
                # print(x_sample.mean())

            return_ids.append(patch_id)

            x_sample = self.l2norm(x_sample)# channelwise normalization by the l2norm

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)

        return return_feats, return_ids


#######################################
# Non-linear Registration/Deformation #
#######################################

class RegNet(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        use_probs=False,
        device='cpu'):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            cpoints_level=int_downsize
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        # self.resize = RescaleTransform(inshape, factor=1 / int_downsize, gaussian_filter_flag=gaussian_filter_flag) if resize else None
        self.resize = None
        self.fullsize = RescaleTransform(inshape, factor=int_downsize) if resize else None

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def init_net(self, device, weightsfile=None, model_key = None,):
        init_net = Init_net()

        self.fullsize = self.fullsize.to(device)
        self.integrate = self.integrate.to(device)
        self.transformer = self.transformer.to(device)
        self.flow = self.flow.to(device)

        if weightsfile is not None:
            if not exists(weightsfile):
                raise ValueError("No trained weights are found for " + weightsfile)

            model_key = '' if model_key is None else '_' + model_key
            checkpoint = torch.load(weightsfile, map_location=device)

            self.unet_model = self.unet_model.to(device)
            self.load_unet(checkpoint['state_dict' + model_key])
            self.load_state_dict(checkpoint['state_dict' + model_key], strict=True)
            self.eval()
        else:
            # init flow layer with small weights and bias
            self.flow = self.flow.to(device)
            self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape), requires_grad=True)
            self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape), requires_grad=True)
            self.unet_model = init_net.init_net(self.unet_model, device=device)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        if self.resize:
            flow_field = self.resize(flow_field)

        preint_flow = flow_field

        # integrate to produce diffeomorphic warp
        if self.integrate:
            flow_field = self.integrate(flow_field)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        # warp image with flow field
        y_source = self.transformer(source, flow_field)

        # return non-integrated flow field if training
        if not registration:
            return y_source, flow_field, preint_flow
        else:
            return y_source, flow_field

    def predict(self, image, flow, svf=True, **kwargs):

        if svf:
            flow = self.integrate(flow)

            if self.fullsize:
                flow = self.fullsize(flow)

        return self.transformer(image, flow, **kwargs)

    def get_flow_field(self, flow_field):
        if self.integrate:
            flow_field = self.integrate(flow_field)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        return flow_field

    def load_unet(self, state_dict):

        own_state = self.unet_model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

###################################
# Linear Registration/Deformation #
###################################
class InstanceAffineModel(nn.Module):
    def __init__(self, vol_shape, device='cpu'):
        super().__init__()

        self.vol_shape = list(vol_shape)
        self.ndims = len(vol_shape)
        self.device = device
        self.warper = SpatialTransformer(self.vol_shape)

        self.T = torch.nn.Parameter(torch.zeros(self.ndims, self.ndims+1))
        self.T.requires_grad = True
        self.eye = torch.eye(self.ndims+1, device=device)[:self.ndims]


    def _get_dense_field(self, affine_matrix):

        vectors = [torch.arange(0, end=s, step=1, device=self.device, dtype=torch.float) for s in self.vol_shape]
        mesh = torch.meshgrid(*vectors)  # grid of vectors
        mesh = [f.float() for f in mesh]
        # mesh = [mesh[f] - (self.vol_shape[f] - 1) / 2 for f in range(self.ndims)]  # shift center

        # add an all-ones entry and transform into a large matrix
        flat_mesh = [torch.reshape(f, (-1,)) for f in mesh]
        flat_mesh.append(torch.ones(np.prod(self.vol_shape), device=self.device, dtype=torch.float))
        mesh_matrix = torch.transpose(torch.stack(flat_mesh, dim=1), 1, 0)  # 4 x nb_voxels

        # compute locations
        loc_matrix = torch.matmul(affine_matrix, mesh_matrix)  # N+1 x nb_voxels
        loc = torch.reshape(loc_matrix[:self.ndims, :], [self.ndims] + list(self.vol_shape))  # *volshape x N

        # get shifts and return

        shift = loc - torch.stack(mesh, dim=0)
        return shift.float()


    def forward(self, inputs):

        #Reshape params to matrix and add identity to learn only the shift

        #RigidRegistration matrix to dense
        params = self.T + self.eye
        dense_field = self._get_dense_field(params)
        dense_field = dense_field.view((1,) + dense_field.shape)
        dense_field = torch.cat(inputs.shape[0] * [dense_field])
        output = self.warper(inputs, dense_field)
        return output, dense_field, self.T


class InstanceBlockModel(nn.Module):

    def __init__(self, vol_shape, vol_affine, device='cpu'):
        super().__init__()

        self.vol_affine = Variable(torch.Tensor(vol_affine).to(device))
        self.vol_shape = vol_shape
        self.device = device
        self.normvecs = []
        self.crs = []
        # self.warper = SpatialTransformerAffine(self.vol_shape)

        self.crs_var = []
        self.normvecs_var = []


    def compute_affine_matrix(self):
        raise NotImplementedError


    def _get_dense_field(self, mesh_matrix, affine_matrix):
        # vectors = [torch.arange(0, end=s, step=1, device=self.device, dtype=torch.float) for s in self.vol_shape]
        # mesh = torch.meshgrid(*vectors)  # grid of vectors
        # mesh = [f.float() for f in mesh]
        #
        #
        # # add an all-ones entry and transform into a large matrix
        # flat_mesh = [torch.reshape(f, (-1,)) for f in mesh]
        # flat_mesh.append(torch.ones(np.prod(self.vol_shape), device=self.device, dtype=torch.float))
        # mesh_matrix = torch.transpose(torch.stack(flat_mesh, dim=1), 1, 0)  # 4 x nb_voxels

        ndims = len(self.vol_shape)
        vol_shape = self.vol_shape

        # compute locations
        loc_matrix = torch.matmul(affine_matrix, mesh_matrix)  # N x nb_voxels
        loc = torch.reshape(loc_matrix, [ndims] + list(vol_shape))  # *volshape x N

        # get shifts and return
        shift = loc - mesh_matrix[:ndims].view([ndims] + list(vol_shape))
        return shift.float()

    def forward(self, header, image_shape, mode='bilinear', *args, **kwargs):

        # Reshape params to matrix and add identity to learn only the shift
        ndims = len(self.vol_shape)
        params_affine = self.compute_affine_matrix()
        new_header = torch.matmul(params_affine, header)
        affine = torch.matmul(torch.inverse(new_header), self.vol_affine)
        affine = torch.unsqueeze(affine[:ndims], dim=0)

        return affine[:ndims], new_header


class Instance3DGlobalScalingModel(InstanceBlockModel):

    def __init__(self, vol_shape, vol_affine, scaling, device='cpu'):
        super().__init__(vol_shape, vol_affine, device)

        self.translation = torch.nn.Parameter(torch.zeros(3, 1))
        self.angle = torch.nn.Parameter(torch.zeros(3, 1))
        self.scaling = scaling
        self.translation.requires_grad = True
        self.angle.requires_grad = True
        self.scaling.requires_grad = True

    def compute_affine_matrix(self):
        cr = np.mean(self.crs, axis=0)
        normvecs = self.normvecs
        if len(normvecs) > 1:
            for it_nv, nv in enumerate(normvecs[1:]):
                if np.dot(nv.T,normvecs[0]) < 0:
                    normvecs[it_nv+1] = -nv
            normvec = np.mean(self.normvecs, axis=0)
        else:
            normvec = normvecs

        normvec = normvec/np.linalg.norm(normvec)

        cr = torch.FloatTensor(cr)
        normvec = torch.FloatTensor(normvec)
        norm_xy = torch.norm(normvec[:2])

        T1 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T2 = Variable(torch.eye(4).cuda(),  requires_grad=False)
        T3 = Variable(torch.eye(4).cuda(),  requires_grad=False)
        T1inv = Variable(torch.eye(4).cuda(),  requires_grad=False)
        T2inv = Variable(torch.eye(4).cuda(),  requires_grad=False)
        T3inv = Variable(torch.eye(4).cuda(),  requires_grad=False)
        T4a = Variable(torch.eye(4).cuda(),  requires_grad=False)
        T4b1 = Variable(torch.eye(4).cuda(),  requires_grad=False)
        T4b2 = Variable(torch.eye(4).cuda(),  requires_grad=False)
        T4b3 = Variable(torch.eye(4).cuda(),  requires_grad=False)
        Ttrans = Variable(torch.eye(4).cuda(),  requires_grad=False)

        translation = self.translation
        angles = self.angle / 180 * math.pi
        scalingXY = torch.exp(self.scaling[0]/100)
        scalingZ = torch.exp(self.scaling[1]/100)

        T1[0, 3] = -cr[0]
        T1[1, 3] = -cr[1]
        T1[2, 3] = -cr[2]

        T1inv[0, 3] = cr[0]
        T1inv[1, 3] = cr[1]
        T1inv[2, 3] = cr[2]

        if norm_xy != 0:
            T2[0,0] = normvec[0]/norm_xy
            T2[0,1] = normvec[1]/norm_xy
            T2[1,0] = -normvec[1]/norm_xy
            T2[1,1] = normvec[0]/norm_xy

            T2inv[0, 0] = normvec[0] / norm_xy
            T2inv[0, 1] = -normvec[1] / norm_xy
            T2inv[1, 0] = normvec[1] / norm_xy
            T2inv[1, 1] = normvec[0] / norm_xy

        T3[0, 0] = normvec[2]
        T3[0, 2] = -norm_xy
        T3[2, 0] = norm_xy
        T3[2, 2] = normvec[2]

        T3inv[0, 0] = normvec[2]
        T3inv[0, 2] = norm_xy
        T3inv[2, 0] = -norm_xy
        T3inv[2, 2] = normvec[2]

        T4a[0, 0] = scalingXY
        T4a[1, 1] = scalingXY
        T4a[2, 2] = scalingZ

        T4b1[1, 1] = torch.cos(angles[0])
        T4b1[1, 2] = -torch.sin(angles[0])
        T4b1[2, 1] = torch.sin(angles[0])
        T4b1[2, 2] = torch.cos(angles[0])

        T4b2[0, 0] = torch.cos(angles[1])
        T4b2[0, 2] = torch.sin(angles[1])
        T4b2[2, 0] = -torch.sin(angles[1])
        T4b2[2, 2] = torch.cos(angles[1])

        T4b3[0, 0] = torch.cos(angles[2])
        T4b3[0, 1] = -torch.sin(angles[2])
        T4b3[1, 0] = torch.sin(angles[2])
        T4b3[1, 1] = torch.cos(angles[2])

        Ttrans[0, 3] = translation[0]
        Ttrans[1, 3] = translation[1]
        Ttrans[2, 3] = translation[2]

        T = torch.chain_matmul(Ttrans, T1inv, T2inv, T3inv, T4b3, T4b2, T4b1, T4a, T3, T2, T1)

        return T


class InstanceGlobalScalingModel(InstanceBlockModel):
    def __init__(self, vol_shape, vol_affine, shared_scaling, device='cpu'):
        super().__init__(vol_shape, vol_affine, device)

        self.translation = torch.nn.Parameter(torch.zeros(3, 1))
        self.angle = torch.nn.Parameter(torch.zeros(1, 1))
        self.scaling = shared_scaling

        self.translation.requires_grad = True
        self.angle.requires_grad = True
        self.scaling.requires_grad = True


    def compute_affine_matrix(self):

        cr = np.mean(self.crs, axis=0)
        for it_nv, nv in enumerate(self.normvecs[1:]):
            if np.dot(nv.T, self.normvecs[0]) < 0:
                self.normvecs[it_nv + 1] = -nv

        normvec = np.mean(self.normvecs, axis=0)
        normvec = normvec / np.linalg.norm(normvec)

        cr = torch.FloatTensor(cr)
        normvec = torch.FloatTensor(normvec)
        norm_xy = torch.norm(normvec[:2])


        T1 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T2 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T3 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T1inv = Variable(torch.eye(4).cuda(), requires_grad=False)
        T2inv = Variable(torch.eye(4).cuda(), requires_grad=False)
        T3inv = Variable(torch.eye(4).cuda(), requires_grad=False)
        T4a = Variable(torch.eye(4).cuda(), requires_grad=False)
        T4b = Variable(torch.eye(4).cuda(), requires_grad=False)
        Ttrans = Variable(torch.eye(4).cuda(), requires_grad=False)

        translation = self.translation
        angles = self.angle / 180 * math.pi
        scalingXY = torch.exp(self.scaling[0] / 100)
        scalingZ = torch.exp(self.scaling[1] / 100)

        T1[0, 3] = -cr[0]
        T1[1, 3] = -cr[1]
        T1[2, 3] = -cr[2]

        T1inv[0, 3] = cr[0]
        T1inv[1, 3] = cr[1]
        T1inv[2, 3] = cr[2]

        if norm_xy != 0:
            T2[0, 0] = normvec[0] / norm_xy
            T2[0, 1] = normvec[1] / norm_xy
            T2[1, 0] = -normvec[1] / norm_xy
            T2[1, 1] = normvec[0] / norm_xy

            T2inv[0, 0] = normvec[0] / norm_xy
            T2inv[0, 1] = -normvec[1] / norm_xy
            T2inv[1, 0] = normvec[1] / norm_xy
            T2inv[1, 1] = normvec[0] / norm_xy

        T3[0, 0] = normvec[2]
        T3[0, 2] = -norm_xy
        T3[2, 0] = norm_xy
        T3[2, 2] = normvec[2]

        T3inv[0, 0] = normvec[2]
        T3inv[0, 2] = norm_xy
        T3inv[2, 0] = -norm_xy
        T3inv[2, 2] = normvec[2]

        T4a[0, 0] = scalingXY
        T4a[1, 1] = scalingXY
        T4a[2, 2] = scalingZ

        T4b[0, 0] = torch.cos(angles[0])
        T4b[0, 1] = -torch.sin(angles[0])
        T4b[1, 0] = torch.sin(angles[0])
        T4b[1, 1] = torch.cos(angles[0])

        Ttrans[0, 3] = translation[0]
        Ttrans[1, 3] = translation[1]
        Ttrans[2, 3] = translation[2]


        T = torch.chain_matmul(Ttrans, T1inv, T2inv, T3inv, T4b, T4a, T3, T2, T1)

        return T


class InstanceIndividualScalesModel(InstanceBlockModel):
    def __init__(self, vol_shape, vol_affine, device='cpu'):
        super().__init__(vol_shape, vol_affine, device)

        self.translation = torch.nn.Parameter(torch.zeros(3, 1))
        self.angle = torch.nn.Parameter(torch.zeros(1, 1))
        self.scaling = torch.nn.Parameter(torch.zeros(2, 1))
        self.scaling.requires_grad = True
        self.translation.requires_grad = True
        self.angle.requires_grad = True

    def compute_affine_matrix(self):

        cr = np.mean(self.crs, axis=0)
        for it_nv, nv in enumerate(self.normvecs[1:]):
            if np.dot(nv.T, self.normvecs[0]) < 0:
                self.normvecs[it_nv + 1] = -nv

        normvec = np.mean(self.normvecs, axis=0)
        normvec = normvec / np.linalg.norm(normvec)

        cr = torch.FloatTensor(cr)
        normvec = torch.FloatTensor(normvec)
        norm_xy = torch.norm(normvec[:2])

        T1 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T2 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T3 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T1inv = Variable(torch.eye(4).cuda(), requires_grad=False)
        T2inv = Variable(torch.eye(4).cuda(), requires_grad=False)
        T3inv = Variable(torch.eye(4).cuda(), requires_grad=False)
        T4a = Variable(torch.eye(4).cuda(), requires_grad=False)
        T4b = Variable(torch.eye(4).cuda(), requires_grad=False)
        Ttrans = Variable(torch.eye(4).cuda(), requires_grad=False)

        translation = self.translation
        angles = self.angle / 180 * math.pi
        scalingXY = torch.exp(self.scaling[0] / 100)
        scalingZ = torch.exp(self.scaling[1] / 100)

        T1[0, 3] = -cr[0]
        T1[1, 3] = -cr[1]
        T1[2, 3] = -cr[2]

        T1inv[0, 3] = cr[0]
        T1inv[1, 3] = cr[1]
        T1inv[2, 3] = cr[2]

        if norm_xy != 0:
            T2[0, 0] = normvec[0] / norm_xy
            T2[0, 1] = normvec[1] / norm_xy
            T2[1, 0] = -normvec[1] / norm_xy
            T2[1, 1] = normvec[0] / norm_xy

            T2inv[0, 0] = normvec[0] / norm_xy
            T2inv[0, 1] = -normvec[1] / norm_xy
            T2inv[1, 0] = normvec[1] / norm_xy
            T2inv[1, 1] = normvec[0] / norm_xy

        T3[0, 0] = normvec[2]
        T3[0, 2] = -norm_xy
        T3[2, 0] = norm_xy
        T3[2, 2] = normvec[2]

        T3inv[0, 0] = normvec[2]
        T3inv[0, 2] = norm_xy
        T3inv[2, 0] = -norm_xy
        T3inv[2, 2] = normvec[2]

        T4a[0, 0] = scalingXY
        T4a[1, 1] = scalingXY
        T4a[2, 2] = scalingZ

        T4b[0, 0] = torch.cos(angles[0])
        T4b[0, 1] = -torch.sin(angles[0])
        T4b[1, 0] = torch.sin(angles[0])
        T4b[1, 1] = torch.cos(angles[0])

        Ttrans[0, 3] = translation[0]
        Ttrans[1, 3] = translation[1]
        Ttrans[2, 3] = translation[2]

        T = torch.chain_matmul(Ttrans, T1inv, T2inv, T3inv, T4b, T4a, T3, T2, T1)

        return T


class InstanceIndividualSimilarityModel(InstanceBlockModel):
    def __init__(self, vol_shape, vol_affine, device='cpu'):
        super().__init__(vol_shape, vol_affine, device)

        self.translation = torch.nn.Parameter(torch.zeros(3, 1))
        self.angle = torch.nn.Parameter(torch.zeros(3, 1))
        self.scaling = torch.nn.Parameter(torch.zeros(2, 1))
        self.scaling.requires_grad = True
        self.translation.requires_grad = True
        self.angle.requires_grad = True


    def compute_affine_matrix(self):

        cr = np.mean(self.crs, axis=0)
        cr = Variable(torch.FloatTensor(cr).to(self.device), requires_grad = False)


        T1 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T2 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T3 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T4 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T5 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T6 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T7 = Variable(torch.eye(4).cuda(), requires_grad=False)

        translation = self.translation
        angles = self.angle / 180 * math.pi
        scalingXY = torch.exp(self.scaling[0] / 100)
        scalingZ = torch.exp(self.scaling[1] / 100)

        T1[0,3] = -cr[0]
        T1[1,3] = -cr[1]
        T1[2,3] = -cr[2]

        T2[0,0] = scalingXY
        T2[1,1] = scalingXY
        T2[2,2] = scalingZ

        T3[1, 1] = torch.cos(angles[0])
        T3[1, 2] = -torch.sin(angles[0])
        T3[2, 1] = torch.sin(angles[0])
        T3[2, 2] = torch.cos(angles[0])

        T4[0, 0] = torch.cos(angles[1])
        T4[0, 2] = torch.sin(angles[1])
        T4[2, 0] = -torch.sin(angles[1])
        T4[2, 2] = torch.cos(angles[1])

        T5[0, 0] = torch.cos(angles[2])
        T5[0, 1] = -torch.sin(angles[2])
        T5[1, 0] = torch.sin(angles[2])
        T5[1, 1] = torch.cos(angles[2])

        T6[0, 3] = cr[0]
        T6[1, 3] = cr[1]
        T6[2, 3] = cr[2]

        T7[0, 3] = translation[0]
        T7[1, 3] = translation[1]
        T7[2, 3] = translation[2]

        T = torch.chain_matmul(T7, T6, T5, T4, T3, T2, T1)

        return T


class InstanceIndividualSimilarityModelWithRigid(InstanceIndividualSimilarityModel):
    def __init__(self, vol_shape, vol_affine, block_shape, device='cpu', torch_dtype=torch.float):
        super().__init__(vol_shape, vol_affine, device)

        self.block_shape = block_shape[:2]
        self.num_slices = block_shape[-1]

        self.slice_scaling = torch.nn.Parameter(torch.zeros(self.num_slices, 1))
        self.slice_translation = torch.nn.Parameter(torch.zeros(self.num_slices, 3, 1))
        self.slice_angle = torch.nn.Parameter(torch.zeros(self.num_slices, 3, 1))
        self.slice_scaling.requires_grad = True
        self.slice_translation.requires_grad = True
        self.slice_angle.requires_grad = True
        self.transform = SpatialTransformerAffine(self.block_shape, torch_dtype=torch_dtype)


    def warp(self, image, affine_matrix, **kwargs):
        image = torch.permute(image[0], [3, 0, 1, 2])
        out_image = self.transform(image, affine_matrix, **kwargs)
        return torch.unsqueeze(torch.permute(out_image, [1, 2, 3, 0]), dim=0)


    def forward(self, header, image_shape, mode='bilinear', *args, **kwargs):

        affine, new_header = super().forward(header, image_shape, mode, *args, **kwargs)

        # Container for all affine matrices, grouped in "batch size"
        total_affine = Variable(torch.zeros(self.num_slices, 3, 4).cuda(), requires_grad=False)

        # Reshape params to matrix and add identity to learn only the shift
        ndims = len(self.vol_shape)

        for it_s in range(self.num_slices):
            params_affine = self.compute_params(self.slice_angle[it_s], self.slice_angle[it_s], self.slice_scaling[it_s])
            total_affine[it_s] = params_affine[:ndims]

        return affine, new_header, total_affine


    def compute_params(self, translation, angle, scaling):

        cr = np.mean(self.crs, axis=0)
        cr = Variable(torch.FloatTensor(cr).to(self.device), requires_grad = False)


        T1 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T2 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T3 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T4 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T5 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T6 = Variable(torch.eye(4).cuda(), requires_grad=False)
        T7 = Variable(torch.eye(4).cuda(), requires_grad=False)

        angles = angle / 180 * math.pi
        scalingXY = torch.exp(scaling[0] / 100)

        T1[0,3] = -cr[0]
        T1[1,3] = -cr[1]
        T1[2,3] = -cr[2]

        T2[0,0] = scalingXY
        T2[1,1] = scalingXY

        T3[1, 1] = torch.cos(angles[0])
        T3[1, 2] = -torch.sin(angles[0])
        T3[2, 1] = torch.sin(angles[0])
        T3[2, 2] = torch.cos(angles[0])

        T4[0, 0] = torch.cos(angles[1])
        T4[0, 2] = torch.sin(angles[1])
        T4[2, 0] = -torch.sin(angles[1])
        T4[2, 2] = torch.cos(angles[1])

        T5[0, 0] = torch.cos(angles[2])
        T5[0, 1] = -torch.sin(angles[2])
        T5[1, 0] = torch.sin(angles[2])
        T5[1, 1] = torch.cos(angles[2])

        T6[0, 3] = cr[0]
        T6[1, 3] = cr[1]
        T6[2, 3] = cr[2]

        T7[0, 3] = translation[0]
        T7[1, 3] = translation[1]
        T7[2, 3] = translation[2]

        T = torch.chain_matmul(T7, T6, T5, T4, T3, T2, T1)

        return T



class InstanceIndividualSimilarityModelWithNonlinear(InstanceIndividualSimilarityModel):
    '''
    It adds slice-wide (2d) nonlinear deformations to InstanceIndividualSimilarityModel
    '''
    def __init__(self, vol_shape, vol_affine, block_shape, cp_spacing=10, device='cpu', torch_dtype=torch.float):
        '''
        :param vol_shape: tuple. Shape of the block
        :param vol_affine:
        :param block_shape:
        :param cp_spacing:
        :param device:
        '''
        super().__init__(vol_shape, vol_affine, device)

        self.cp_spacing = np.array(cp_spacing) # Spacing between control points in voxels
        self.field_size = np.ceil(block_shape[0:2]/self.cp_spacing).astype('int') # Size of field image

        # Eugenio: Adria, in the next line I'm assuming batchsize=1. Feel free to edit to make it cleaner, if you want
        self.nonlin_fields = torch.nn.Parameter(torch.zeros(1,2,self.field_size[0],self.field_size[1],block_shape[-1]))
        self.nonlin_fields.requires_grad = True

        # This resizer is specific for each block, since they've got different sizes...
        self.resizer = ResizeTransform(self.nonlin_fields.shape[2:],target_size=block_shape,gaussian_filter_flag=False)

        # The resampling is also specific for each block, as they all have different sizes.
        self.transform = SpatialTransformer(block_shape, torch_dtype=torch_dtype)

    def forward(self, header, image_shape, mode='bilinear', *args, **kwargs):

        # Reshape params to matrix and add identity to learn only the shift
        ndims = len(self.vol_shape)
        params_affine = self.compute_affine_matrix()
        new_header = torch.matmul(params_affine, header)
        affine = torch.matmul(torch.inverse(new_header), self.vol_affine)
        affine = torch.unsqueeze(affine[:ndims], dim=0)

        # Eugenio: rather than having Nslice 2D deformation fields, it's just easier to append a zero field in z and use existing code for 3D  ;-)
        # Also: at one point I thought of dividing the fields by cp_spacing to precondition a bit, since ResizeTransform
        # will multiply by the scaling factors, but in the end it worked OK without. If gradients because unstable,
        # though, we could think of switching it back on.
        # And also: I keep assuming batchsize=1; feel free to improve if you want.
        # fields_3d = torch.cat((self.nonlin_fields / torch.from_numpy(self.cp_spacing).to(self.device),
        #                        torch.zeros(1,1,self.field_size[0],self.field_size[1], image_shape[-1]).to(self.device)), 1)
        fields_3d = torch.cat((self.nonlin_fields, torch.zeros(1, 1, self.field_size[0], self.field_size[1], image_shape[-1]).to(self.device)), 1)

        # Eugenio: maybe add integration here?

        nonlin_fields_fullsiz = self.resizer(fields_3d)

        return affine[:ndims], new_header, nonlin_fields_fullsiz

    def warp(self, image, field, **kwargs):
        return self.transform(image, field, **kwargs)
