import csv
from os.path import join, dirname

import numpy as np
import torch
import PIL
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable # plotting

color_code = ['b', 'r', 'k', 'g', 'm', 'y', 'c', 'pink']

def plot_results(filepath, keys=None, show=False, restrict_ylim=False):
    past_backend = plt.get_backend()
    plt.switch_backend('Agg')
    if keys is None:
        keys = ['loss']
    x_axis = []
    x_axis_val = []
    starting_epoch = []
    n_epochs = -1
    results_dict = {'Train': {k: [] for k in keys}, 'Validation': {k: [] for k in keys}}
    with open(filepath, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for it_row, row in enumerate(csvreader):
            if row['Phase'] == 'Train':
                x_axis.append(it_row + 1)
            else:
                x_axis_val.append(it_row + 1)

            if int(row['epoch']) > n_epochs:
                n_epochs = int(row['epoch'])
                starting_epoch.append(it_row)

            for k in keys:
                try:
                    results_dict[row['Phase']][k].append(float(row[k]))
                except:
                    n_iterations = it_row - starting_epoch[-1]
                    results_dict[row['Phase']][k].append(np.mean(results_dict['Train'][k][-n_iterations:]))

    # n_iter = len(results_dict['Train'][keys[0]])
    # n_iter_per_epoch = 1.0 * n_iter / n_epochs

    # x_axis = np.arange(0, n_iter )
    # x_axis_val = np.arange(0, n_iter, n_iter_per_epoch)
    delta_epoch = int(np.ceil(n_epochs/100)*10)
    print(n_epochs)
    plt.figure()
    for k in keys:
        print(k)
        it_color_code = 0
        if results_dict['Train'][k]:
            plt.plot(x_axis, results_dict['Train'][k], color=color_code[it_color_code], marker='*')
            it_color_code +=1

        if results_dict['Validation'][k]:
            plt.plot(x_axis_val, results_dict['Validation'][k], color=color_code[it_color_code], marker='*')
            it_color_code +=1

        if restrict_ylim:
            # ymin, ymax = np.min(results_dict['Validation'][k]), np.max(results_dict['Validation'][k])
            ymin, ymax = np.percentile(results_dict['Train'][k],1), np.percentile(results_dict['Train'][k],99)

            if ymin < 0:
                ymin = ymin*1.5
            else:
                ymin=ymin*0.5

            if ymax < 0:
                ymax = ymax*0.5
            else:
                ymax=ymax*1.5

            plt.ylim([ymin,ymax])


        plt.xticks(starting_epoch[::delta_epoch], np.arange(0,n_epochs,delta_epoch))
        plt.xlabel('Number of epochs')
        plt.ylabel(k)
        plt.grid()

        if show:
            plt.show()
        else:
            plt.savefig(join(dirname(filepath), k + '_results.png'))

        plt.close()

    plt.switch_backend(past_backend)

def plot_multiple_results(filepath_list, keys=None, show=False, restrict_ylim=False):
    past_backend = plt.get_backend()
    plt.switch_backend('Agg')

    if keys is None:
        keys = ['loss']


    n_rows = 4 if len(filepath_list) > 4 else 2
    n_cols = len(filepath_list) // 4 + 1 if len(filepath_list) > 4 else 2

    for k in keys:
        print(k)
        fig, ax = plt.subplots(n_rows, n_cols)

        total_ymin = 1000000
        total_ymax = -100000
        for it_filepath, filepath in enumerate(filepath_list):
            x_axis = []
            x_axis_val = []
            starting_epoch = []
            n_epochs = -1
            results_dict = {'Train': {k: []}, 'Validation': {k: []}}

            plt_row = np.mod(it_filepath,n_rows)
            plt_col = it_filepath//n_rows
            with open(filepath, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)

                for it_row, row in enumerate(csvreader):
                    if row['Phase'] == 'Train':
                        x_axis.append(it_row + 1)
                    else:
                        x_axis_val.append(it_row + 1)

                    if int(row['epoch']) > n_epochs:
                        n_epochs = int(row['epoch'])
                        starting_epoch.append(it_row)

                        try:
                            results_dict[row['Phase']][k].append(float(row[k]))
                        except:
                            n_iterations = it_row - starting_epoch[-1]
                            results_dict[row['Phase']][k].append(np.mean(results_dict['Train'][k][-n_iterations:]))


            it_color_code = 0
            if results_dict['Train'][k]:
                ax[plt_row, plt_col].plot(x_axis, results_dict['Train'][k], color=color_code[it_color_code], marker='*')
                it_color_code += 1

            if results_dict['Validation'][k]:
                ax[plt_row, plt_col].plot(x_axis_val, results_dict['Validation'][k], color=color_code[it_color_code], marker='*')
                it_color_code += 1

            ymin, ymax = np.percentile(results_dict['Train'][k],10), np.percentile(results_dict['Train'][k],90)

            total_ymin = min(total_ymin, ymin)
            total_ymax = max(total_ymax, ymax)

            ax[plt_row, plt_col].grid()

        if restrict_ylim:
            for plt_row in range(n_rows):
                for plt_col in range(n_cols):
                    ax[plt_row, plt_col].set_ylim([total_ymin, total_ymax])
                    if plt_row != 0 or plt_col != 0:
                        ax[plt_row, plt_col].get_xaxis().set_visible(False)
                        ax[plt_row, plt_col].get_yaxis().set_visible(False)
        if show:
            plt.show()
        else:
            plt.savefig(join(dirname(dirname(dirname(filepath_list[0]))), k + '_results.png'))

    plt.close()

    plt.switch_backend(past_backend)

def slices(slices_in,  # the 2D slices
           titles=None,  # list of titles
           cmaps=None,  # list of colormaps
           norms=None,  # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           grid=False,  # option to plot the images in a grid or a single row
           width=15,  # width in in
           show=True,  # option to actually show the plot (plt.show())
           axes_off=True,
           imshow_args=None):
    '''
    plot a grid of slices (2d images)
    '''

    # input processing
    if isinstance(slices_in,torch.Tensor):
        slices_in = [slices_in.cpu().detach().numpy()]

    elif isinstance(slices_in, np.ndarray):
        slices_in = [slices_in]

    elif isinstance(slices_in, PIL.Image.Image):
        slices_in = [np.asarray(slices_in)]

    elif isinstance(slices_in, list):
        for it_s, s  in enumerate(slices_in):
            if isinstance(s, torch.Tensor):
                if s.is_cuda:
                    s = s.cpu()
                slices_in[it_s] = s.detach().numpy()

            elif isinstance(s, PIL.Image.Image):
                slices_in[it_s] = np.asarray(s)

    nb_plots = len(slices_in)
    for si, slice_in in enumerate(slices_in):
        if len(slice_in.shape) != 2:
            assert len(slice_in.shape) == 3 and slice_in.shape[-1] == 3, 'each slice has to be 2d or RGB (3 channels)'
        slices_in[si] = slice_in.astype('float')

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i], interpolation="nearest", norm=norms[i], **imshow_args[i])

        # colorbars
        # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        if do_colorbars and cmaps[i] is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_ax, cax=cax)

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        if axes_off:
            ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)

    if show:
        plt.tight_layout()
        plt.show()

    return (fig, axs)


def plot_grad_flow(named_parameters, show=False, save_dir=None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            # layers.append(n.split('.')[1])
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    if show:
        plt.show()
    else:
        save_dir = 'gradflow.png' if save_dir is None else save_dir
        plt.savefig(save_dir)
        plt.close()
