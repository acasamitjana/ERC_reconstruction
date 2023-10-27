import pdb
from os.path import join
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from src import models
from src.callbacks import History, ModelCheckpoint, PrinterCallback, ToCSVCallback, EarlyStopping

DOWNSAMPLING_DICT = {
    0: 0, #Padding
    4: 1, #Input size
    8: 2, #Downsamplig x2
    12: 3, #Downsamplig x4
    16: 4, #Downsamplig x8
    20: 5, #Downsamplig x8
    24: 6, #Downsamplig x8

}
torch_dtype = torch.float

class Registration(object):

    def __init__(self, device, loss_function_dict, loss_weight_dict, callbacks, p_dict, da_model=None):
        '''
        This class trains a registration network using SVFs and a composite loss function of (i) intensity loss + (ii)
        regularisation loss.
        :param device: 'cpu' or 'cuda:0'
        :param loss_function_dict:
        :param loss_weight_dict:
        :param callbacks: (list) extra callbacks to add - if any: it could be an empty list.
        :param p_dict: parameter dictionary specifying some training parameters
        :param da_model: a models.TensorDeformation instance
        '''

        self.loss_function_dict = loss_function_dict
        self.loss_weight_dict = loss_weight_dict
        self.log_keys = ['loss_' + l for l in loss_function_dict.keys()] \
                        + ['w_loss_' + l for l in loss_function_dict.keys()]\
                        + ['loss', 'time_duration (s)']

        attach = True if p_dict['STARTING_EPOCH'] > 0 else False
        logger = History(self.log_keys)
        mcheck = ModelCheckpoint(join(p_dict['RESULTS_DIR'], 'checkpoints'), p_dict['SAVE_MODEL_FREQUENCY'])
        results_path = join(p_dict['RESULTS_DIR'], 'results', 'training_results.csv')
        training_tocsv = ToCSVCallback(filepath=results_path, keys=self.log_keys, attach=attach)
        training_printer = PrinterCallback(log_interval=p_dict['LOG_INTERVAL'])
        early_stop = EarlyStopping()
        callback_list = [logger, mcheck, training_printer, training_tocsv, early_stop]
        self.callbacks = callbacks + callback_list

        self.da_model = da_model if da_model is not None else False

        self.parameter_dict = p_dict
        self.device = device
        self.state_dict = {'best_loss': 100000, 'best_metric': 0, 'early_stop': False}

    # def iterate_bidir(self, generator, model, optimizer, epoch, mask_flag, da_model=None, **kwargs):
    #
    #     N = len(generator.dataset)
    #     rid_epoch_list = []
    #     total_iter = 0
    #     for batch_idx, data_dict in enumerate(generator):
    #         flo_image, ref_image = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
    #         nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
    #         affine_field = [aff.to(self.device) for aff in data_dict['affine']]
    #         rid_epoch_list.extend(data_dict['rid'])
    #         model.zero_grad()
    #         if mask_flag:
    #             flo_mask, ref_mask = data_dict['x_flo_mask'].to(self.device), data_dict['x_ref_mask'].to(self.device)
    #
    #
    #         if da_model is not None:
    #             flip_lr = 0
    #             flip_ud = 0
    #             ref_image = da_model.transform(ref_image, affine_field[0], nonlinear_field[0], flip_ud, flip_lr)
    #             flo_image = da_model.transform(flo_image, affine_field[1], nonlinear_field[1], flip_ud, flip_lr)
    #
    #             if mask_flag:
    #                 ref_mask = da_model.transform(ref_mask, affine_field[0], nonlinear_field[0], flip_ud, flip_lr, mode='nearest')
    #                 flo_mask = da_model.transform(flo_mask, affine_field[1], nonlinear_field[1], flip_ud, flip_lr, mode='nearest')
    #
    #
    #         reg_flo_image, flow_image, v_image = model(flo_image, ref_image)
    #         flow_image_rev = model.get_flow_field(-v_image)
    #
    #         reg_ref_image = model.predict(ref_image, flow_image_rev, svf=False)
    #
    #         loss_dict = {}
    #         if mask_flag:
    #             loss_dict['registration'] = self.loss_function_dict['registration'](ref_image, reg_flo_image, mask=ref_mask)
    #             loss_dict['registration'] += self.loss_function_dict['registration'](flo_image, reg_ref_image, mask=flo_mask)
    #
    #         else:
    #             loss_dict['registration'] = self.loss_function_dict['registration'](ref_image, reg_flo_image)
    #             loss_dict['registration'] += self.loss_function_dict['registration'](flo_image, reg_ref_image)
    #
    #         loss_dict['registration'] = 0.5 * loss_dict['registration']
    #         loss_dict['registration_smoothness'] = self.loss_function_dict['registration_smoothness'](v_image)
    #
    #         log_dict = {'loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()}
    #
    #         for k, v in loss_dict.items():
    #             loss_dict[k] = self.loss_weight_dict[k] * v
    #
    #         total_loss = sum([l for l in loss_dict.values()])
    #
    #         total_loss.backward()
    #         # plot_grad_flow(model.named_parameters(), save_dir='model_reg')
    #         optimizer.step()
    #
    #         log_dict = {**log_dict,
    #                     **{'w_loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()},
    #                     **{'loss': total_loss.item()}}
    #
    #         total_iter += len(flo_image)
    #         for cb in self.callbacks:
    #             cb.on_step_fi(log_dict, model, epoch, iteration=total_iter, N=N)
    #
    #     return self.callbacks
    #
    # def iterate(self, generator, model, optimizer, epoch,  mask_flag, da_model=None, **kwargs):
    #
    #     N = len(generator.dataset)
    #     rid_epoch_list = []
    #     total_iter = 0
    #     for batch_idx, data_dict in enumerate(generator):
    #
    #         flo_image, ref_image = data_dict['x_flo'].to(self.device), data_dict['x_ref'].to(self.device)
    #         nonlinear_field = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
    #         affine_field = [aff.to(self.device) for aff in data_dict['affine']]
    #         rid_epoch_list.extend(data_dict['rid'])
    #         model.zero_grad()
    #
    #         if da_model is not None:
    #             flip_lr = 0
    #             flip_ud = 0
    #             ref_image = da_model.transform(ref_image, affine_field[0], nonlinear_field[0], flip_ud, flip_lr)
    #             flo_image = da_model.transform(flo_image, affine_field[1], nonlinear_field[1], flip_ud, flip_lr)
    #             if mask_flag:
    #                 ref_mask = data_dict['x_ref_mask'].to(self.device)
    #                 ref_mask = da_model.transform(ref_mask, affine_field[0], nonlinear_field[0], flip_ud, flip_lr, mode='nearest')
    #
    #         reg_flo_image, flow_image, v_image = model(flo_image, ref_image)
    #
    #         loss_dict = {}
    #         if mask_flag:
    #             loss_dict['registration'] = self.loss_function_dict['registration'](ref_image, reg_flo_image, mask=ref_mask)
    #         else:
    #             loss_dict['registration'] = self.loss_function_dict['registration'](ref_image, reg_flo_image)
    #
    #         loss_dict['registration_smoothness'] = self.loss_function_dict['registration_smoothness'](v_image)
    #
    #         log_dict = {'loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()}
    #         for k, v in loss_dict.items():
    #             loss_dict[k] = self.loss_weight_dict[k] * v
    #
    #         total_loss = sum([l for l in loss_dict.values()])
    #
    #         total_loss.backward()
    #         optimizer.step()
    #
    #         log_dict = {**log_dict,
    #                     **{'w_loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()},
    #                     **{'loss': total_loss.item()}}
    #
    #         total_iter += len(flo_image)
    #         for cb in self.callbacks:
    #             cb.on_step_fi(log_dict, model, epoch, iteration=total_iter, N=N)
    #
    #     return self.callbacks

    def compute_loss(self, tensor_dict, log_dict):
        total_loss = 0.
        for k, loss in self.loss_function_dict.items():
            if k == 'registration':
                lnum = loss(tensor_dict['x_ref'],tensor_dict['x_flo_reg'], mask=tensor_dict['x_ref_mask'])
            elif k == 'registration_labels':
                lnum = loss(tensor_dict['x_ref_labels'],tensor_dict['x_flo_labels_reg'], mask=tensor_dict['x_ref_mask'])

            elif k == 'smoothness':
                lnum = loss(tensor_dict['vel'])

            else:
                raise ValueError('Loss dict keys do not match. Valid options: registration, smoothness.')

            log_dict['loss_' + k] = lnum.item()
            lnum = self.loss_weight_dict[k] * lnum
            log_dict['w_loss_' + k] = lnum.item()

            total_loss += lnum

        total_loss.backward()
        log_dict['loss'] = total_loss.item()

    def compute_loss_bidir(self, tensor_dict, log_dict):
        total_loss = 0.
        for k, loss in self.loss_function_dict.items():
            if k == 'registration':
                # from src.utils.visualization import slices
                # slices([tensor_dict['ref_image'][0,0], tensor_dict['reg_flo_image'][0,0], tensor_dict['ref_image'][0,0] - tensor_dict['flo_image'][0,0], tensor_dict['ref_mask'][0,0]])
                # slices([tensor_dict['flo_image'][0,0], tensor_dict['reg_ref_image'][0,0], tensor_dict['flo_image'][0,0] - tensor_dict['reg_ref_image'][0,0], tensor_dict['flo_mask'][0,0]])
                lnum = loss(tensor_dict['x_ref'],tensor_dict['x_flo_reg'], mask=tensor_dict['x_ref_mask'])
                lnum += loss(tensor_dict['x_flo'], tensor_dict['x_ref_reg'], mask=tensor_dict['x_flo_mask'])
                lnum = lnum / 2

            elif k == 'registration_mask':
                lnum = loss(tensor_dict['x_ref_mask'],tensor_dict['x_flo_mask_reg'])
                lnum += loss(tensor_dict['x_flo_mask'],tensor_dict['x_ref_mask_reg'])
                lnum = lnum / 2

            elif k == 'smoothness':
                lnum = loss(tensor_dict['vel'], mask=tensor_dict['x_ref_mask'])

            else:
                raise ValueError('Loss dict keys do not match. Valid options: registration, smoothness.')

            log_dict['loss_' + k] = lnum.item()
            lnum = self.loss_weight_dict[k] * lnum
            log_dict['w_loss_' + k] = lnum.item()

            total_loss += lnum

        if total_loss != 0:
            total_loss.backward()
            log_dict['loss'] = total_loss.item()
        else:
            log_dict['loss'] = 0.

    def geometric_augmentation(self, data_dict, def_dict):
        if self.da_model:
            flip_ud = sum(def_dict['flip_ud']) if len(def_dict['flip_ud']) > 0 else def_dict['flip_ud']
            flip_lr = sum(def_dict['flip_lr']) if len(def_dict['flip_lr']) > 0 else def_dict['flip_lr']


            for k, v in data_dict.items():
                mod_idx = 0 if 'ref' in k else 1
                if ('mask' in k or 'labels' in k) and v is not None:
                    data_dict[k] = self.da_model.transform(v, def_dict['affine'][mod_idx],
                                                           def_dict['nonlinear'][mod_idx],
                                                           mode='nearest')

                else:
                    data_dict[k] = self.da_model.transform(v, def_dict['affine'][mod_idx],
                                                           def_dict['nonlinear'][mod_idx])

            del def_dict
            # data_dict['x_ref'] = self.da_model.transform(data_dict['x_ref'], affine[0], nonlinear[0], flip_ud, flip_lr)
            # data_dict['x_flo'] = self.da_model.transform(data_dict['x_flo'], affine[1], nonlinear[1], flip_ud, flip_lr)
            #
            # if data_dict['x_ref_mask'] is not None:
            #     data_dict['x_ref_mask'] = self.da_model.transform(data_dict['x_ref_mask'], affine[0], nonlinear[0],
            #                                                       flip_ud, flip_lr, mode='nearest')
            #     data_dict['x_ref_mask_init'] = self.da_model.transform(data_dict['x_ref_mask_init'], affine[0], nonlinear[0],
            #                                                            flip_ud, flip_lr, mode='nearest')
            #
            # if data_dict['x_flo_mask'] is not None:
            #     data_dict['x_flo_mask'] = self.da_model.transform(data_dict['x_flo_mask'], affine[1], nonlinear[1],
            #                                                       flip_ud, flip_lr, mode='nearest')
            #     data_dict['x_flo_mask_init'] = self.da_model.transform(data_dict['x_flo_mask_init'], affine[1], nonlinear[1],
            #                                                            flip_ud, flip_lr, mode='nearest')

        return data_dict

    def step(self, data_dict, model, optimizer,  mask_flag=False, bidir_flag=False):

        ###############
        #  Input data #
        ###############
        log_dict = {}

        tensor_dict = {}
        def_dict = {'flip_ud': data_dict['flip_ud'], 'flip_lr': data_dict['flip_lr']}
        with torch.no_grad():
            tensor_dict['x_ref'] = data_dict['x_ref'].to(self.device)
            tensor_dict['x_flo'] = data_dict['x_flo'].to(self.device)
            tensor_dict['x_ref_mask'] = data_dict['x_ref_mask'].to(self.device) if mask_flag else None
            tensor_dict['x_flo_mask'] = data_dict['x_flo_mask'].to(self.device) if (mask_flag and bidir_flag) else None
            def_dict['nonlinear'] = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            def_dict['affine'] = [aff.to(self.device) for aff in data_dict['affine']]

            tensor_dict = self.geometric_augmentation(tensor_dict, def_dict)

        # pdb.set_trace()
        # from utils.visualization import slices
        # slices([tensor_dict['x_ref'][0,0], tensor_dict['x_flo'][0,0], tensor_dict['x_ref_mask'][0,0], tensor_dict['x_flo_mask'][0,0]])
        ##################
        #  Forward pass #
        ##################
        for p in model.parameters():
            p.requires_grad = True

        optimizer.zero_grad()

        tensor_dict['x_flo_reg'], f, tensor_dict['vel'] = model(tensor_dict['x_flo'], tensor_dict['x_ref'])
        if 'registration_mask' in self.loss_function_dict:
            tensor_dict['x_flo_mask_reg'] = model.predict(tensor_dict['x_flo'], f, svf=False)

        if bidir_flag:
            flow_image_rev = model.get_flow_field(-tensor_dict['vel'])
            tensor_dict['x_ref_reg'] = model.predict(tensor_dict['x_ref'], flow_image_rev, svf=False)
            if 'registration_mask' in self.loss_function_dict:
                tensor_dict['x_ref_mask_reg'] = model.predict(tensor_dict['x_ref_mask'], flow_image_rev, svf=False)

            self.compute_loss_bidir(tensor_dict, log_dict)

        else:
            self.compute_loss(tensor_dict, log_dict)

        optimizer.step()
        del tensor_dict

        return log_dict

    def init_train(self, model):
        model.train()

    def train(self, model, optimizer, generator, **kwargs):
        for cb in self.callbacks:
            cb.on_train_init(model, starting_epoch=self.parameter_dict['STARTING_EPOCH'])

        self.init_train(model)

        batch_size = generator.batch_size
        N = len(generator.dataset)
        for epoch in range(self.parameter_dict['STARTING_EPOCH'], self.parameter_dict['N_EPOCHS']):
            epoch_start_time = time.time()
            for cb in self.callbacks:
                cb.on_epoch_init(model, epoch)

            logs_dict = {}
            for batch_idx, data_dict in enumerate(generator):
                log_dict = self.step(data_dict, model, optimizer, **kwargs)

                if log_dict is not None:
                    for k, v in log_dict.items():
                        if k in logs_dict.keys():
                            logs_dict[k] += v
                        else:
                            logs_dict[k] = v

                # Callbacks
                for cb in self.callbacks:
                    cb.on_step_fi(log_dict, model, epoch, iteration=batch_idx + 1, N=N, batch_size=batch_size)
                    if isinstance(cb, EarlyStopping):
                        self.state_dict['early_stop'] = cb.early_stop

                if self.state_dict['early_stop']:
                    print(data_dict['rid'])
                    break

            for k, v in logs_dict.items():
                logs_dict[k] = v / (N / batch_size)

            epoch_end_time = time.time()
            logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

            for cb in self.callbacks:
                cb.on_epoch_fi(logs_dict, model, epoch, optimizer=optimizer)

            if self.state_dict['early_stop']: break


        for cb in self.callbacks:
            cb.on_train_fi(model)

class WeaklyRegistration(Registration):

    def step(self, data_dict, model, optimizer, mask_flag=False, bidir_flag=False):

        ###############
        #  Input data #
        ###############
        log_dict = {}

        tensor_dict = {}
        def_dict = {}
        def_dict['flip_ud'] = data_dict['flip_ud']
        def_dict['flip_lr'] = data_dict['flip_lr']
        with torch.no_grad():
            tensor_dict['x_ref'] = data_dict['x_ref'].to(self.device)
            tensor_dict['x_flo'] = data_dict['x_flo'].to(self.device)
            tensor_dict['x_ref_labels'] = data_dict['x_ref_labels'].to(self.device)
            tensor_dict['x_flo_labels'] = data_dict['x_flo_labels'].to(self.device)
            tensor_dict['x_ref_mask'] = data_dict['x_ref_mask'].to(self.device) if mask_flag else None
            tensor_dict['x_flo_mask'] = data_dict['x_flo_mask'].to(self.device) if (mask_flag and bidir_flag) else None

            def_dict['nonlinear'] = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            def_dict['affine'] = [aff.to(self.device) for aff in data_dict['affine']]

            tensor_dict = self.geometric_augmentation(tensor_dict, def_dict)

        ##################
        #  Forward pass #
        ##################
        for p in model.parameters():
            p.requires_grad = True

        optimizer.zero_grad()

        tensor_dict['x_flo_reg'], flow_image, tensor_dict['vel'] = model(tensor_dict['x_flo'], tensor_dict['x_ref'])
        tensor_dict['x_flo_labels_reg'] = model.predict(tensor_dict['x_flo_labels'], flow_image, svf=False, mode='nearest')

        if bidir_flag:
            flow_image_rev = model.get_flow_field(-tensor_dict['vel'])
            tensor_dict['x_ref_ref'] = model.predict(tensor_dict['x_ref'], flow_image_rev, svf=False)
            tensor_dict['x_ref_labels_reg'] = model.predict(tensor_dict['x_ref_labels'], flow_image_rev, svf=False, mode='nearest')

            self.compute_loss_bidir(tensor_dict, log_dict)

        else:
            self.compute_loss(tensor_dict, log_dict)

        optimizer.step()

        del tensor_dict

        return log_dict

class SbR(Registration):

    def __init__(self, device, loss_function_dict, loss_weight_dict, callbacks, nce_layers, num_patches,
                 clip_grad, f_lr, p_dict, mask_nce_flag=True, da_model=None):

        '''
        This class trains the SbR framework.
        :param device: 'cpu' or 'cuda:0'
        :param loss_function_dict:
        :param loss_weight_dict:
        :param callbacks: (list) extra callbacks to add - if any: it could be an empty list.
        :param p_dict: parameter dictionary specifying some training parameters
        :param da_model: a models.TensorDeformation instance

        :param nce_layers: which layers should I take features from
        :param num_patches: number of pathces per layer
        :param clip_grad: (bool, default=False). If clipping gradients
        :param f_lr: learning rate for the headers
        :param mask_nce_flag: (bool, default=True). Whether to sample only pixels within a mask on the InfoNCE loss
        '''
        super().__init__(device=device, loss_function_dict=loss_function_dict, loss_weight_dict=loss_weight_dict,
                         callbacks=callbacks, p_dict=p_dict, da_model=da_model)

        self.nce_layers = nce_layers
        self.num_patches = num_patches

        self.mask_nce_flag = mask_nce_flag

        self.f_lr = f_lr
        self.clip_grad = clip_grad

        self.init_F_flag = False

    def initialize_F(self, tensor_dict, model_dict, optimizer_dict, weightsfile=None, **kwargs):

        '''
        Networks A/B are defined on the domain they apply (e.g: G_A transforms from B to A, while DX_A is used to
        discriminate between real/fake in domain A (generated A and real A))
        '''


        _ = self.compute_G_loss(tensor_dict, model_dict, {}, **kwargs)
        # G_loss.backward()

        if weightsfile is not None:
            checkpoint = torch.load(weightsfile)
            for model_key, model in model_dict.items():
                if model_key == 'F_M':
                    model.load_state_dict(checkpoint['state_dict_' + model_key])

        elif self.parameter_dict['STARTING_EPOCH'] > 0:
            weightsfile = 'model_checkpoint.' + str(self.parameter_dict['STARTING_EPOCH'] - 1) + '.pth'
            checkpoint = torch.load(join(self.parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile))
            for optimizer_key, optimizer in optimizer_dict.items():
                if optimizer_key == 'F_M':
                    optimizer.load_state_dict(checkpoint['optimizer_' + optimizer_key])

            for model_key, model in model_dict.items():
                if model_key == 'F_M':
                    model.load_state_dict(checkpoint['state_dict_' + model_key])

        if self.loss_weight_dict['nce'] > 0:
            optimizer_F_M = torch.optim.Adam(model_dict['F_M'].parameters(), lr=self.f_lr, betas=(0.5, 0.999))
            optimizer_dict['F_M'] = optimizer_F_M

        self.init_F_flag = True

    def compute_NCE_loss(self, data, model_dict, loss_list, nce_mask):
        src, trgt = data[:2]
        n_layers = len(self.nce_layers)

        feat_trgt = model_dict['G_M'](trgt, self.nce_layers, encode_only=True)
        feat_src = model_dict['G_M'](src, self.nce_layers, encode_only=True)
        if nce_mask:
            src_m = data[2]
            mp = nn.ReflectionPad2d(3)
            mm = nn.MaxPool2d(2)
            kernel = torch.ones((1, 1, 3, 3), device=self.device, requires_grad=False)

            downsampling_masks = [mp(src_m), src_m]

            for it in range(2, 5):
                m_tmp = (F.conv2d(downsampling_masks[it - 1], kernel, padding=1) > 0).float()
                downsampling_masks.append(mm(m_tmp))
            for it in range(5, 6):
                m_tmp = (F.conv2d(downsampling_masks[it - 1], kernel, padding=1) > 0).float()
                downsampling_masks.append(m_tmp)

            feat_src_m = []
            for layer in self.nce_layers:
                feat_src_m.append(downsampling_masks[DOWNSAMPLING_DICT[layer]])

            feat_src_pool, patch_ids = model_dict['F_M'](feat_src, num_patches=self.num_patches, patch_ids=None,
                                                         mask_sampling=feat_src_m)
            feat_trgt_pool, _ = model_dict['F_M'](feat_trgt, num_patches=self.num_patches, patch_ids=patch_ids)

        else:
            feat_trgt = model_dict['G_M'](trgt, self.nce_layers, encode_only=True)
            feat_src = model_dict['G_M'](src, self.nce_layers, encode_only=True)
            feat_src_pool, patch_ids = model_dict['F_M'](feat_src, num_patches=self.num_patches, patch_ids=None)
            feat_trgt_pool, _ = model_dict['F_M'](feat_trgt, num_patches=self.num_patches, patch_ids=patch_ids)

        NCE_loss = 0.0
        for f_q, f_k, crit in zip(feat_trgt_pool, feat_src_pool, loss_list):
            loss = crit(f_q, f_k)
            NCE_loss += loss.mean()
        return NCE_loss / n_layers

    def compute_G_loss(self, tensor_dict, model_dict, log_dict, bidir_flag=False, mask_flag=False, nce_mask=True):

        total_loss = 0.
        for k, loss in self.loss_function_dict.items():
            if self.loss_weight_dict[k] == 0:
                log_dict['loss_' + k] = 0.
                log_dict['w_loss_' + k] = 0.
                continue

            if k == 'registration':
                mask = tensor_dict['x_ref_mask'] if mask_flag else None
                lnum = loss(tensor_dict['x_ref'], tensor_dict['x_flo_G_reg'], mask=mask)
                if bidir_flag:
                    mask = tensor_dict['x_flo_mask'] if mask_flag else None
                    lnum += loss(tensor_dict['x_flo_G'], tensor_dict['x_ref_reg'], mask=mask)
                    lnum = lnum/2

            elif k == 'smoothness':
                lnum = loss(tensor_dict['vel'])

            elif k == 'nce':
                data_in = tensor_dict['x_flo'], tensor_dict['x_flo_G'], tensor_dict['x_flo_mask']#_init']
                lnum = self.compute_NCE_loss(data_in, model_dict, loss, nce_mask=nce_mask)

                data_in = tensor_dict['x_ref'], tensor_dict['x_flo_reg'], tensor_dict['x_ref_mask']#_init']
                lnum += self.compute_NCE_loss(data_in, model_dict, loss, nce_mask=nce_mask)
            else:
                raise ValueError('Loss dict keys do not match. Valid options: registration, smoothness, nce.')

            log_dict['loss_' + k] = lnum.item()
            lnum = self.loss_weight_dict[k] * lnum
            log_dict['w_loss_' + k] = lnum.item()

            total_loss += lnum

        total_loss.backward()
        log_dict['loss'] = total_loss.item()

    def step(self, data_dict, model_dict, optimizer_dict, bidir_flag=False, mask_flag=False, nce_mask=True):

        log_dict = {}

        ###############
        #  Input data #
        ###############


        tensor_dict = {}
        def_dict = {}
        def_dict['flip_ud'] = data_dict['flip_ud']
        def_dict['flip_lr'] = data_dict['flip_lr']
        with torch.no_grad():
            tensor_dict['x_ref'] = data_dict['x_ref'].to(self.device)
            tensor_dict['x_flo'] = data_dict['x_flo'].to(self.device)
            tensor_dict['x_ref_mask'] = data_dict['x_ref_mask'].to(self.device)
            tensor_dict['x_flo_mask'] = data_dict['x_flo_mask'].to(self.device)
            # tensor_dict['x_ref_mask_init'] = data_dict['x_ref_mask_init'].to(self.device)
            # tensor_dict['x_flo_mask_init'] = data_dict['x_flo_mask_init'].to(self.device)
            def_dict['nonlinear'] = [nlf.to(self.device) for nlf in data_dict['nonlinear']]
            def_dict['affine'] = [nlf.to(self.device) for nlf in data_dict['affine']]

            if torch.sum(tensor_dict['x_ref_mask']) == 0 or torch.sum(tensor_dict['x_flo_mask']) == 0:
                return {}

            tensor_dict = self.geometric_augmentation(tensor_dict, def_dict)

        ###################
        #  Initialization #
        ###################

        if not self.init_F_flag:
            tensor_dict['x_flo_G'] = model_dict['G_M'](tensor_dict['x_flo'])
            tensor_dict['x_flo_G_reg'], flow, tensor_dict['vel'] = model_dict['R_M'](tensor_dict['x_flo_G'], tensor_dict['x_ref'])
            tensor_dict['x_flo_reg'] = model_dict['R_M'].predict(tensor_dict['x_flo'], flow, svf=False)

            if bidir_flag:
                tensor_dict['x_ref_reg'] = model_dict['R_M'].predict(tensor_dict['x_ref'], -tensor_dict['vel'], svf=True)

            self.initialize_F(tensor_dict, model_dict, optimizer_dict, bidir_flag=bidir_flag,
                              mask_flag=mask_flag, nce_mask=nce_mask)

        #################
        #  Forward pass #
        #################
        for p in model_dict['G_M'].parameters():
            p.requires_grad = True
        optimizer_dict['G_M'].zero_grad()

        if 'F_M' in optimizer_dict.keys():
            for p in model_dict['F_M'].parameters():
                p.requires_grad = True
            optimizer_dict['F_M'].zero_grad()

        tensor_dict['x_flo_G'] = model_dict['G_M'](tensor_dict['x_flo'])
        tensor_dict['x_flo_G_reg'], flow, tensor_dict['vel'] = model_dict['R_M'](tensor_dict['x_flo_G'],
                                                                                 tensor_dict['x_ref'])
        tensor_dict['x_flo_reg'] = model_dict['R_M'].predict(tensor_dict['x_flo'], flow, svf=False)

        if bidir_flag:
            tensor_dict['x_ref_reg'] = model_dict['R_M'].predict(tensor_dict['x_ref'], -tensor_dict['vel'], svf=True)

        self.compute_G_loss(tensor_dict, model_dict, log_dict, bidir_flag, mask_flag, nce_mask)

        optimizer_dict['G_M'].step()
        if 'F_M' in optimizer_dict.keys():
            optimizer_dict['F_M'].step()

        return log_dict

    def init_train(self, model):
        model['G_M'].train()
        model['F_M'].train()

class InfoNCE2D(SbR):

    def compute_NCE_loss(self, data, model_dict, loss_list, nce_mask):
        src, trgt = data[:2]
        n_layers = len(self.nce_layers)

        feat_trgt = model_dict['G_M'](trgt, self.nce_layers, encode_only=True)
        feat_src = model_dict['G_M'](src, self.nce_layers, encode_only=True)
        if nce_mask:
            src_m = data[2]
            mp = nn.ReflectionPad2d(3)
            mm = nn.MaxPool2d(2)
            kernel = torch.ones((1, 1, 3, 3), device=self.device, requires_grad=False)

            downsampling_masks = [mp(src_m), src_m]

            for it in range(2, 5):
                m_tmp = (F.conv2d(downsampling_masks[it - 1], kernel, padding=1) > 0).float()
                downsampling_masks.append(mm(m_tmp))
            for it in range(5, 6):
                m_tmp = (F.conv2d(downsampling_masks[it - 1], kernel, padding=1) > 0).float()
                downsampling_masks.append(m_tmp)

            feat_src_m = []
            for layer in self.nce_layers:
                feat_src_m.append(downsampling_masks[DOWNSAMPLING_DICT[layer]])

            feat_src_pool, patch_ids = model_dict['F_M'](feat_src, num_patches=self.num_patches, patch_ids=None,
                                                         mask_sampling=feat_src_m)
            feat_trgt_pool, _ = model_dict['F_M'](feat_trgt, num_patches=self.num_patches, patch_ids=patch_ids)

        else:
            feat_trgt = model_dict['G_M'](trgt, self.nce_layers, encode_only=True)
            feat_src = model_dict['G_M'](src, self.nce_layers, encode_only=True)
            feat_src_pool, patch_ids = model_dict['F_M'](feat_src, num_patches=self.num_patches, patch_ids=None)
            feat_trgt_pool, _ = model_dict['F_M'](feat_trgt, num_patches=self.num_patches, patch_ids=patch_ids)

        NCE_loss = 0.0
        for f_q, f_k, crit in zip(feat_trgt_pool, feat_src_pool, loss_list):
            loss = crit(f_q, f_k)
            NCE_loss += loss.mean()
        return NCE_loss / n_layers

class LinearTraining(object):

    def __init__(self, dataset, loss_function_dict, loss_weight_dict, callbacks, device):
        '''
        This instance is used in the hierarchical linear alignment from all stacks of histology images (aligned using
        blockface photographs) and the MRI
        :param dataset: dataset object with loading functions
        :param loss_function_dict:
        :param loss_weight_dict:
        :param callbacks:
        :param device:
        '''
        self.dataset = dataset
        self.loss_function_dict = loss_function_dict
        self.loss_weight_dict = loss_weight_dict
        self.callbacks = callbacks
        self.device = device

    def iterate(self, model_dict, optimizer,epoch, **kwargs):

        for bid in self.dataset.data_loader.subject_dict.keys():
            model_dict[bid].train()
            model_dict[bid].zero_grad()

        ref_image = torch.FloatTensor(self.dataset.MRI[np.newaxis, np.newaxis])
        ref_image = ref_image.to(self.device)

        ref_mask_cerebellum = torch.FloatTensor(self.dataset.MRI_mask_cerebellum[np.newaxis, np.newaxis])
        ref_mask_cerebellum = ref_mask_cerebellum.to(self.device)

        mask_cr_bs = self.dataset.MRI_mask_cerebrum[np.newaxis, np.newaxis] + self.dataset.MRI_mask_brainstem[
            np.newaxis, np.newaxis]
        ref_mask_cerebrum_brainstem = torch.FloatTensor(mask_cr_bs)
        ref_mask_cerebrum_brainstem = ref_mask_cerebrum_brainstem.to(self.device)

        ref_mask_dilated = torch.FloatTensor(self.dataset.MRI_mask_dilated[np.newaxis, np.newaxis])
        ref_mask_dilated = ref_mask_dilated.to(self.device)

        def closure():

            if torch.is_grad_enabled():
                optimizer.zero_grad()

            accum_shape = (1, 1) + self.dataset.data_loader.vol_shape
            accum_image = torch.zeros(accum_shape, dtype=torch_dtype, device=self.device)
            accum_mask = torch.zeros(accum_shape, dtype=torch_dtype, device=self.device)
            accum_mask_cerebellum = torch.zeros(accum_shape, dtype=torch_dtype, device=self.device)
            accum_mask_cerebrum_brainstem = torch.zeros(accum_shape, dtype=torch_dtype, device=self.device)
            header_list = []

            for it_bid, bid in enumerate(self.dataset.data_loader.subject_dict.keys()):
                image, mask, header = self.dataset[bid]
                image, mask, header = image.to(self.device), mask.to(self.device), header.to(self.device)
                image, mask = image.type(torch_dtype), mask.type(torch_dtype)
                image_shape = image.shape

                model = model_dict[bid]
                if isinstance(model, models.InstanceIndividualSimilarityModelWithNonlinear):
                    affine, new_header, fields = model(header, image.shape[2:])
                    image = model.warp(image, fields, mode='bilinear')
                    mask = model.warp(mask, fields, mode='bilinear')

                else:
                    affine, new_header = model(header, image.shape[2:])

                affine = affine.type(torch_dtype)

                reg_mask = model_dict['warper'](mask, affine, shape=image_shape[2:])
                accum_mask += reg_mask

                reg_image = model_dict['warper'](image, affine, shape=image_shape[2:])
                accum_image += reg_image * reg_mask

                if 'C' in bid:
                    accum_mask_cerebellum += reg_mask
                else:
                    accum_mask_cerebrum_brainstem += reg_mask

                header_list.append(new_header)

                # loss_i +=loss_function_dict['intensities'](reg_image, ref_image,reg_mask > 0)

            accum_image = accum_image / (accum_mask + 1e-5)
            loss_i = self.loss_function_dict['intensities'](accum_image, ref_image, ref_mask_dilated > 0)
            loss_i = self.loss_weight_dict['intensities'] * loss_i

            overlap_cerebellum = self.loss_function_dict['overlap'](accum_mask_cerebellum, ref_mask_cerebellum)
            overlap_cerebrum_brainstem = self.loss_function_dict['overlap'](accum_mask_cerebrum_brainstem,
                                                                            ref_mask_cerebrum_brainstem)

            loss_overlap_overlap_cerebellum = self.loss_weight_dict['overlap_cerebellum'] * overlap_cerebellum
            loss_overlap_overlap_cerebrum_brainstem = self.loss_weight_dict[
                                                          'overlap_cerebrum_brainstem'] * overlap_cerebrum_brainstem
            loss_overlap = self.loss_weight_dict['overlap'] * (
                        loss_overlap_overlap_cerebellum + loss_overlap_overlap_cerebrum_brainstem)
            loss_scale = self.loss_function_dict['scale'](header_list)
            loss_scale = self.loss_weight_dict['scale'] * loss_scale

            total_loss = loss_i + loss_overlap + loss_scale
            total_loss.backward()

            return total_loss

        optimizer.step(closure=closure)

        accum_image = torch.zeros((1, 1) + self.dataset.data_loader.vol_shape, dtype=torch_dtype, device=self.device)
        accum_mask = torch.zeros((1, 1) + self.dataset.data_loader.vol_shape, dtype=torch_dtype, device=self.device)
        accum_mask_cerebellum = torch.zeros((1, 1) + self.dataset.data_loader.vol_shape, dtype=torch_dtype, device=self.device)
        accum_mask_cerebrum_brainstem = torch.zeros((1, 1) + self.dataset.data_loader.vol_shape, dtype=torch_dtype,
                                                    device=self.device)

        header_list = []
        loss_dict = {'intensities': 0}
        for it_bid, bid in enumerate(self.dataset.data_loader.subject_dict.keys()):
            image, mask, header = self.dataset[bid]
            image, mask, header = image.to(self.device), mask.to(self.device), header.to(self.device)
            image = image.type(torch_dtype)
            mask = mask.type(torch_dtype)
            image_shape = image.shape

            model = model_dict[bid]

            if isinstance(model, models.InstanceIndividualSimilarityModelWithNonlinear):
                affine, new_header, fields = model(header, image.shape[2:])
                image = model.warp(image, fields, mode='bilinear')
                mask = model.warp(mask, fields, mode='bilinear')
            else:
                affine, new_header = model(header, image.shape[2:])

            affine = affine.type(torch_dtype)

            reg_mask = model_dict['warper'](mask, affine, shape=image_shape[2:])
            accum_mask += reg_mask

            reg_image = model_dict['warper'](image, affine, shape=image_shape[2:])
            accum_image += reg_image * reg_mask
            # loss_dict['intensities'] += loss_function_dict['intensities'](reg_image, ref_image, ref_mask_dilated > 0)

            if 'C' in bid:
                accum_mask_cerebellum += reg_mask
            else:
                accum_mask_cerebrum_brainstem += reg_mask

            header_list.append(new_header)

        accum_image = accum_image / (accum_mask + 1e-5)
        loss_dict['intensities'] = self.loss_function_dict['intensities'](accum_image, ref_image, ref_mask_dilated > 0)
        loss_dict['intensities'] = self.loss_weight_dict['intensities'] * loss_dict['intensities']

        overlap_cerebellum = self.loss_function_dict['overlap'](accum_mask_cerebellum, ref_mask_cerebellum)
        overlap_cerebrum_brainstem = self.loss_function_dict['overlap'](accum_mask_cerebrum_brainstem,
                                                                   ref_mask_cerebrum_brainstem)

        loss_dict['overlap'] = self.loss_weight_dict['overlap_cerebellum'] * overlap_cerebellum
        loss_dict['overlap'] += self.loss_weight_dict['overlap_cerebrum_brainstem'] * overlap_cerebrum_brainstem
        loss_dict['overlap'] = self.loss_weight_dict['overlap'] * loss_dict['overlap']

        loss_dict['scale'] = self.loss_function_dict['scale'](header_list)
        loss_dict['scale'] = self.loss_weight_dict['scale'] * loss_dict['scale']

        # TODO: think about whether it'd be worth adding landmarks to nonlinear version in the future
        # loss_dict['landmarks'] = 0
        # loss_dict['landmarks'] = loss_weight_dict['landmarks'] * loss_dict['landmarks']

        total_loss = sum([l for l in loss_dict.values()])
        # total_loss.backward()
        # optimizer.step()

        log_dict = {**{'loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()},
                    **{'loss': total_loss.item()}
                    }

        print('')
        for cb in self.callbacks:
            cb.on_step_fi(log_dict, model_dict, epoch, iteration=1, N=1, batch_size=1)

        return log_dict

    def update_headers(self, model_dict, **kwargs):

        for bid in self.dataset.data_loader.subject_dict.keys():
            model_dict[bid].eval()

        for it_bid, bid in enumerate(self.dataset.data_loader.subject_dict.keys()):
            image, mask, header = self.dataset[bid]

            image, mask, header = image.to(self.device), mask.to(self.device), header.to(self.device)

            model = model_dict[bid]

            if hasattr(model, 'nonlin_fields'):  # Eugenio: there's probably a cleaner way...
                affine, new_header, fields = model(header, image.shape[2:])
            else:
                affine, new_header = model(header, image.shape[2:])

            self.dataset.data_loader.subject_dict[bid]._affine = new_header.cpu().detach().numpy()

    def update_headers_images(self, model_dict):

        for it_bid, bid in enumerate(self.dataset.data_loader.subject_dict.keys()):
            image, mask, header = self.dataset[bid]
            image, mask, header = image.to(self.device), mask.to(self.device), header.to(self.device)

            model = model_dict[bid]
            affine, new_header, fields = model(header.to(self.device), image.shape)

            fields = fields.type(torch_dtype)
            nonlin_mask = model.warp(mask, fields)
            # nonlin_mask = filter_3d(nonlin_mask, kernel_sigma=[2,2,4])
            mask_block = np.squeeze(nonlin_mask.to('cpu').detach().numpy())

            self.dataset.data_loader.subject_dict[bid]._affine = new_header.cpu().detach().numpy()

            self.dataset.mask_dict[bid] = mask_block

