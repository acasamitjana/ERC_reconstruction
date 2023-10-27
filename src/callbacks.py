from os.path import join
import csv

import numpy as np
import torch


class Callback(object):

    def on_train_init(self, model, **kwargs):
        pass

    def on_train_fi(self, model, **kwargs):
        pass

    def on_epoch_init(self, model, epoch, **kwargs):
        pass

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        pass

    def on_step_init(self, logs_dict, model, epoch, **kwargs):
        pass

    def on_step_fi(self, logs_dict, model, epoch,**kwargs):
        pass

class History(Callback):

    def __init__(self, keys=None):
        self.logs = {}

        if keys is None:
            self.keys = []
        else:
            self.keys = keys

    def on_train_init(self, model, **kwargs):
        self.logs['Train'] = {}
        self.logs['Validation'] = {}


    def on_epoch_init(self, model, epoch, **kwargs):
        self.logs['Train'][epoch] = {}
        self.logs['Validation'][epoch] = {}


        for k in self.keys:
            self.logs['Train'][epoch][k] = []


    def on_step_fi(self, logs_dict, model, epoch, **kwargs):
        for k,v in logs_dict.items():
            self.logs['Train'][epoch][k].append(v)


    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        for k,v in logs_dict.items():
            self.logs['Validation'][epoch][k]=v

class ModelCheckpoint(Callback):

    def __init__(self, dirpath, save_model_frequency):

        self.dirpath = dirpath
        self.save_model_frequency = save_model_frequency

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):

        optimizer = kwargs['optimizer']
        checkpoint = {
            'epoch': epoch + 1,
        }
        if isinstance(model, dict):
            for model_name, model_instance in model.items():
                checkpoint['state_dict_' + model_name] = model_instance.state_dict()
        else:
            checkpoint['state_dict'] = model.state_dict()

        if isinstance(optimizer, dict):
            for optimizer_name, optimizer_instance in optimizer.items():
                checkpoint['optimizer_' + optimizer_name] = optimizer_instance.state_dict()
        else:
            checkpoint['optimizer'] = optimizer.state_dict()

        filepath = join(self.dirpath, 'model_checkpoint.LAST.pth')
        torch.save(checkpoint, filepath)
        if self.save_model_frequency != -1 and np.mod(epoch, self.save_model_frequency) == 0:
            filepath = join(self.dirpath, 'model_checkpoint.' + str(epoch) + '.pth')
            torch.save(checkpoint, filepath)


    def on_train_fi(self, model, **kwargs):
        checkpoint = {}
        if isinstance(model, dict):
            for model_name, model_instance in model.items():
                checkpoint['state_dict_' + model_name] = model_instance.state_dict()
        else:
            checkpoint['state_dict'] = model.state_dict()

        filepath = join(self.dirpath, 'model_checkpoint.FI.pth')
        torch.save(checkpoint, filepath)

class PrinterCallback(Callback):

    def __init__(self, log_interval=1, keys=None):
        self.keys = keys
        self.log_interval = log_interval
        self.logs = {}

    def on_train_init(self, model, **kwargs):
        print('######################################')
        print('########## Training started ##########')
        print('######################################')
        print('\n')

    def on_epoch_init(self, model, epoch, **kwargs):
        print('------------- Epoch: ' + str(epoch))

    def on_step_fi(self, logs_dict, model, epoch, **kwargs):

        for k, v in logs_dict.items():
            if k in self.logs.keys():
                self.logs[k] += v/self.log_interval
            else:
                self.logs[k] = v/self.log_interval

        if np.mod(kwargs['iteration'], self.log_interval) == 0:
            to_print = 'Iteration: (' + str(kwargs['iteration']*kwargs['batch_size']) + '/' + str(kwargs['N']) + '). ' + \
                       ', '.join([k + ': ' + str(round(v, 4)) for k, v in self.logs.items()])
            print(to_print)
            self.logs = {}

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        to_print = 'Epoch summary: ' + ','.join([k + ': ' + str(round(v, 6)) for k, v in logs_dict.items()])
        print(to_print)


    def on_train_fi(self, model, **kwargs):
        print('#######################################')
        print('########## Training finished ##########')
        print('#######################################')

class ToCSVCallback(Callback):

    def __init__(self, filepath, keys, attach=False):
        mode = 'a' if attach else 'w'
        fieldnames = ['Phase','epoch','iteration'] + keys
        csvfile = open(filepath, mode)
        self.csvwriter = csv.DictWriter(csvfile, fieldnames)
        self.csvwriter.writeheader()

    def on_step_fi(self, logs_dict, model, epoch, **kwargs):
        write_dict = {**{'Phase': 'Train', 'epoch': epoch, 'iteration': kwargs['iteration']*kwargs['batch_size']}, **logs_dict}
        self.csvwriter.writerow(write_dict)

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        write_dict = {**{'Phase': 'Validation', 'epoch': epoch}, **logs_dict}
        self.csvwriter.writerow(write_dict)

class LRDecay(Callback):

    def __init__(self, optimizer, n_iter_start, n_iter_finish):
        # self.optimizer = optimizer # Eugenio
        self.n_iter_start = n_iter_start
        self.n_iter_finish = n_iter_finish
        self.optimizer = optimizer
        self.lr_init = optimizer.param_groups[0]['lr']

    def on_train_init(self, model, **kwargs):
        init_epoch = kwargs['starting_epoch'] if 'starting_epoch' in kwargs.keys() else 0
        if init_epoch >= self.n_iter_start:
            # Eugenio added 0.75 so we end at 25% of the original LR rather than 0
            updated_lr = (1 - 0.75 * (init_epoch - self.n_iter_start)/(self.n_iter_finish-self.n_iter_start)) * self.lr_init
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = updated_lr

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        if epoch >= self.n_iter_start:
            # Eugenio added 0.75 so we end at 25% of the original LR rather than 0
            updated_lr = (1 - 0.75 * (epoch - self.n_iter_start)/(self.n_iter_finish-self.n_iter_start)) * self.lr_init
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = updated_lr

class EarlyStopping(Callback):

    def __init__(self):
        self.early_stop = False

    def on_step_fi(self, logs_dict, model, epoch, **kwargs):

        # First condition: loss is nan:
        if sum([np.isnan(v) for k, v in logs_dict.items()]) > 0:
            self.early_stop = True


