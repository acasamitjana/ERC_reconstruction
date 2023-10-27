from datetime import datetime, date
from os.path import join, exists
from os import makedirs
import yaml
import csv
import openpyxl
import os

import numpy as np
import torch


class DebugWriter(object):

    def __init__(self, debug_flag, filename = None, attach = False):
        self.filename = filename
        self.debug_flag = debug_flag
        if filename is not None:
            date_start = date.today().strftime("%d/%m/%Y")
            time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if not attach:
                with open(self.filename, 'w') as writeFile:
                    writeFile.write('############################\n')
                    writeFile.write('###### NEW EXPERIMENT ######\n')
                    writeFile.write('############################\n')
                    writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                    writeFile.write('\n')
            else:
                with open(self.filename, 'a') as writeFile:
                    for i in range(4):
                        writeFile.write('\n')
                    writeFile.write('############################\n')
                    writeFile.write('###### NEW EXPERIMENT ######\n')
                    writeFile.write('############################\n')
                    writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                    writeFile.write('\n')

    def write(self, to_write):
        if self.debug_flag:
            if self.filename is None:
                print(to_write, end=' ')
            else:
                with open(self.filename, 'a') as writeFile:
                    writeFile.write(to_write)

class ResultsWriter(object):

    def __init__(self, filename = None, attach = False):
        self.filename = filename
        if filename is not None:
            date_start = date.today().strftime("%d/%m/%Y")
            time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if not attach:
                with open(self.filename, 'w') as writeFile:
                    writeFile.write('############################\n')
                    writeFile.write('###### NEW EXPERIMENT ######\n')
                    writeFile.write('############################\n')
                    writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                    writeFile.write('\n')
            else:
                with open(self.filename, 'a') as writeFile:
                    for i in range(4):
                        writeFile.write('\n')
                    writeFile.write('############################\n')
                    writeFile.write('###### NEW EXPERIMENT ######\n')
                    writeFile.write('############################\n')
                    writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                    writeFile.write('\n')

    def write(self, to_write):
        if self.filename is None:
            print(to_write, end=' ')
        else:
            with open(self.filename, 'a') as writeFile:
                writeFile.write(to_write)

class ExperimentWriter(object):
    def __init__(self, filename = None, attach = False):
        self.filename = filename
        date_start = date.today().strftime("%d/%m/%Y")
        time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        if filename is not None:
            method = 'a' if attach else 'w'
            with open(filename, method) as writeFile:
                writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                writeFile.write('\n')

    def write(self, to_write):
        if self.filename is None:
            print(to_write, end=' ')
        else:
            with open(self.filename, 'a') as writeFile:
                writeFile.write(to_write)

class ConfigFile(object):
    """
    Loads the subjects and the configuration of a study given the path to the configuration file for this study
    """

    def __init__(self, configuration_path):
        """
        Initializes a DataLoader with the given configuration file

        Parameters
        ----------
        configuration_path : String
            Path to the YAMP configuration file with the configuration parameters expected for a study.
            See config/exampleConfig.yaml for more information about the format of configuration files.
        """
        # Load the configuration
        with open(configuration_path, 'r') as conf_file:
            conf = yaml.load(conf_file, yaml.FullLoader)
        self._conf = conf

        self.output_dir = conf['data_management']['output_dir']
        self.use_GPU = conf['general_options']['use_GPU']

        self.learning_rate = conf['training']['learning_rate']
        self.batch_size = conf['training']['batch_size']
        self.n_epochs = conf['training']['n_epochs']
        self.log_interval = conf['training']['log_interval']

        self.momentum = conf['optimizer']['momentum']

class DataWarning(UserWarning):
    pass

def worker_init_fn(wid):
    np.random.seed(np.mod(torch.utils.data.get_worker_info().seed, 2**32-1))

def get_from_cl(obtain_input_from, apply_function, try_ntimes, default_value, show_text, show_error_text):
    """
    Generic method to ask for user input

    Parameters
    ----------
    obtain_input_from : function
        Function used to ask for use input. Default: input
    apply_function : Function
        Function applied to the input if it is correct.
    try_ntimes : int
        Number of times that the user is allowed to provide an incorrect value
    default_value : built-in type
        Default value to be used if the user does not provide a correct value
    show_text : String
        Text to be shown to the user when asking for input
    show_error_text : String
        Error text to be shown when the user fails to input a correct value
    Returns
    -------
    built-in type
        Value that the user has inputted or default value if user failed to input a correct value
    """
    if try_ntimes <= 0:
        try_ntimes = -1

    while try_ntimes != 0:
        s = obtain_input_from(show_text)
        if not s:
            print('Default value selected.')
            return default_value
        else:
            try:
                return apply_function(s)
            except Exception as exc:
                print(show_error_text(exc))

        if try_ntimes < 0:
            print('Infinite')
        else:
            try_ntimes -= 1
            print(try_ntimes)
        print('attempts left.')

        if try_ntimes == 0:
            print('Default value selected.')
        else:
            print('Please, try again.')

    return default_value

def getoneof(option_list,default_value=None,try_ntimes=3,
             show_text='Please, select one of the following (enter index, or leave blank to set by default):',
             obtain_input_from=input,):
    """
    Static method to ask the user to select one option out of several
    Parameters
    ----------
    option_list : List
        List of options provided to the user to pick one out of it
    default_value : int
        The default value to be returned if the user does not provide a correct value
    try_ntimes : int
        Number of times that the user is allowed to provide an incorrect value
    show_text : String
        Text to be shown to the user when asking for input
    obtain_input_from : function
        Function used to ask for use input. Default: input
    Returns
    -------
    built-in type
        The selected option or the default value if the user does not provide a correct selection
    """
    opt_list = list(option_list)
    lol = len(opt_list)
    lslol = len(str(lol))
    right_justify = lambda s: ' ' * (lslol - len(str(s))) + str(s)

    new_show_text = show_text
    for i in range(lol):
        new_show_text += '\n  ' + right_justify(i) + ':  ' + str(opt_list[i])
    new_show_text += '\nYour choice: '

    def get_index(s, ls=lol):
        index = int(s)
        if index < 0 or index >= ls:
            raise IndexError('Index ' + s + ' is out of the accepted range [0, ' + str(ls) + '].')
        return index

    index = get_from_cl(
        obtain_input_from,
        get_index,
        try_ntimes,
        None,
        new_show_text,
        lambda e: 'Could not match input with index: ' + str(e)
    )
    if index == None:
        return default_value

    return opt_list[index]

def get_memory_used():
    import sys
    local_vars = list(locals().items())
    for var, obj in local_vars: print(var, sys.getsizeof(obj) / 1000000000)


def write_affine_matrix(file, matrix):
    with open(file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        for it_r in range(matrix.shape[0]):
            csvwriter.writerow(matrix[it_r])

def read_affine_matrix(file):
    matrix= np.zeros((4,4))
    with open(file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        for it_row, row in enumerate(csvreader):
            matrix[it_row] = [float(r) for r in row]

    return matrix
def create_results_dir(results_dir, subdirs=None):
    if subdirs is None:
        subdirs = ['checkpoints', 'results']
    if not exists(results_dir):
        for sd in subdirs:
            makedirs(join(results_dir, sd))
    else:
        for sd in subdirs:
            if not exists(join(results_dir, sd)):
                makedirs(join(results_dir, sd))


def load_ontology(label_file, unique_labels=None):
    labels_list = []
    h2_labels_dict = {}
    h1_labels_dict = {}

    # Read label mapping from Nellie and Juri: nested dictionaries with different ontologies form the top to the bottom.
    wb = openpyxl.load_workbook(label_file)
    ws = wb.active
    is_title = True
    max_label = 0
    for row in ws.iter_rows(values_only=True):
        if is_title:
            is_title = False
            continue

        if row[0] is not None:
            fs_label = row[0]
            fs_name = row[1]
            h1_labels_dict[fs_label] = {'name': fs_name, 'allen_labels': {}}

        if row[2] is not None:
            allen_label = row[2]
            if unique_labels is not None:
                if allen_label not in unique_labels:
                    continue

            allen_name = row[3]
            if allen_label > max_label:
                max_label = allen_label
            labels_list.append({'h2_name': allen_name, 'h1_name': fs_name, 'h2_num': allen_label, 'h1_num': fs_label})
            h2_labels_dict[allen_label] = {'name': allen_name}
            h1_labels_dict[fs_label]['allen_labels'][allen_label] = allen_name

    h1_labels_dict[24] = {'name': 'CSF', 'allen_labels': {20001: 'CSF'}}
    h1_labels_dict[165] = {'name': 'Skull', 'allen_labels': {20002: 'Skull'}}
    h1_labels_dict[258] = {'name': 'Head-ExtraCerebral', 'allen_labels': {20003: 'Head-ExtraCerebral'}}
    h1_labels_dict[259] = {'name': 'SkullApprox', 'allen_labels': {20004: 'SkullApprox'}}

    h2_labels_dict[20001] = {'name': 'CSF'}
    h2_labels_dict[20002] = {'name': 'Skull'}
    h2_labels_dict[20003] = {'name': 'Head-ExtraCerebral'}
    h2_labels_dict[20004] = {'name': 'SkullApprox'}

    return labels_list, [h1_labels_dict, h2_labels_dict]

def load_lut(filepath):
    label_dict = {}
    with open(filepath, 'r') as f:
        for row in f.readlines():
            feat = row.split('  ')
            lab = int(feat[0])
            if lab == 0:
                rgbt = [0,0,0,0]
            else:
                rgbt = [int(c) for c in feat[-1].split(' \n')[0].split(' ')]
            label_dict[lab] = rgbt[:-1]

    label_list = np.zeros((np.max(list(label_dict.keys()))+1,3))
    for k,v in label_dict.items():
        label_list[k] = v

    return label_list

def mkdir(directory):
    if not os.path.exists(directory): os.makedirs(directory)
