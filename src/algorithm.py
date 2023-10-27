from os.path import join

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_dilation
import gurobipy as gp
import nibabel as nib
from skimage.transform import resize

# Read st2 graph
def init_st2(input_dir, image_shape, nslices, nneighbours = 2, se = None, ref='MRI', c1='LFB'):

    nk = [0]

    w = np.zeros((nslices*2*(1+nneighbours), 2*nslices-1), dtype='int')
    obs_mask = np.zeros(image_shape + (nslices*3*(1+nneighbours),))

    d_Ref = np.zeros((nslices*2*(1+nneighbours),), dtype='int')
    d_C1 = np.zeros((nslices*2*(1+nneighbours),), dtype='int')
    d_inter = np.zeros((nslices*2*(1+nneighbours),), dtype='int')

    phi = np.zeros((2,) + image_shape + (2*nslices*(1+2*nneighbours),))

    # Intermodality

    proxy = nib.load(join(input_dir, ref + '_' + c1 + '.0N.velocity_field.nii.gz'))
    nk.append(nk[-1] + nslices)
    field = np.asarray(proxy.dataobj)
    phi[0, :, :, :nslices] = resize(field[0], image_shape, anti_aliasing=True)
    phi[1, :, :, :nslices] = resize(field[1], image_shape, anti_aliasing=True)

    # Masks
    proxy = nib.load(join(input_dir, ref + '_' + c1 + '.0N.mask.nii.gz'))
    mask_mov = np.double(np.asarray(proxy.dataobj) > 0)
    mask_mov = mask_mov[..., :nslices]
    mask_index = np.sum(mask_mov, axis=(0, 1))

    proxy = nib.load(join(input_dir, ref + '.mask.nii.gz'))
    mask_ref = np.double(np.asarray(proxy.dataobj) > 0)
    mask_ref = mask_ref[..., :nslices]


    mask = mask_mov * mask_ref
    mask = resize(mask, image_shape, anti_aliasing=False, order=0)
    if se is not None:
        for it_shape in range(mask.shape[-1]):
            if mask_index[it_shape] > 0:
                mask[..., it_shape] = np.double(binary_dilation(mask[..., it_shape], se))
            else:
                mask[..., it_shape] = 0

    obs_mask[..., :nslices] = mask
    w[:nslices, :nslices] +=  np.diag((mask_index>0).astype('uint8'))
    d_inter[:nslices] = 1

    # Intramodality
    for zneighbour in range(1,1+nneighbours):
        # Ref
        filename = ref + '.' + str(zneighbour) + 'N'
        proxy = nib.load(join(input_dir, filename + '.velocity_field.nii.gz'))
        nslices_tmp = nslices - zneighbour
        nk.append(nk[-1] + nslices_tmp)

        field = np.asarray(proxy.dataobj)[..., :nslices_tmp]
        phi[0, :, :, nk[-2]:nk[-1]] = resize(field[0], image_shape, anti_aliasing=True)
        phi[1, :, :, nk[-2]:nk[-1]] = resize(field[1], image_shape, anti_aliasing=True)

        # Masks
        proxy = nib.load(join(input_dir, filename + '.mask.nii.gz'))
        mask_mov = np.double(np.asarray(proxy.dataobj) > 0)
        mask_mov = mask_mov[..., :nslices]
        mask_mov = mask_mov[..., :-(zneighbour)]

        proxy = nib.load(join(input_dir, ref + '.mask.nii.gz'))
        mask_ref = np.double(np.asarray(proxy.dataobj) > 0)
        mask_ref = mask_ref[..., :nslices]
        mask_ref = mask_ref[..., :-(zneighbour)]

        mask_index = (np.sum(mask_mov, axis=(0, 1)) > 0) & (np.sum(mask_ref, axis=(0, 1)) > 0)
        mask = mask_ref * mask_mov
        mask = resize(mask, image_shape, anti_aliasing=False, order=0)

        if se is not None:
            for it_shape in range(mask.shape[-1]):
                if mask_index[it_shape] > 0:
                    mask[..., it_shape] = np.double(binary_dilation(mask[..., it_shape], se))
                else:
                    mask[..., it_shape] = 0

        obs_mask[..., nk[-2]:nk[-1]] = mask
        for it_sl in range(zneighbour):
            w[nk[-2]:nk[-1], nslices + it_sl:2*nslices - zneighbour + it_sl] += 1 * np.eye(nslices-zneighbour, dtype=int)
        d_Ref[nk[-2]:nk[-1]] = zneighbour

        # C1
        filename = c1 + '.' + str(zneighbour) + 'N'
        proxy = nib.load(join(input_dir, filename + '.velocity_field.nii.gz'))
        nslices_tmp = nslices - zneighbour
        nk.append(nk[-1] + nslices_tmp)

        field = np.asarray(proxy.dataobj)[..., :nslices_tmp]
        phi[0, :, :, nk[-2]:nk[-1]] = resize(field[0], image_shape, anti_aliasing=True)
        phi[1, :, :, nk[-2]:nk[-1]] = resize(field[1], image_shape, anti_aliasing=True)

        # Masks
        proxy = nib.load(join(input_dir, filename + '.mask.nii.gz'))
        mask_mov = np.double(np.asarray(proxy.dataobj) > 0)
        mask_mov = mask_mov[..., :nslices]
        mask_mov = mask_mov[..., :-(zneighbour)]

        proxy = nib.load(join(input_dir,  c1 + '.mask.nii.gz'))
        mask_ref = np.double(np.asarray(proxy.dataobj) > 0)
        mask_ref = mask_ref[..., :nslices]
        mask_ref = mask_ref[..., :-(zneighbour)]

        mask_index = (np.sum(mask_mov, axis=(0, 1))>0) & (np.sum(mask_ref, axis=(0, 1))>0)
        mask = mask_ref * mask_mov
        mask = resize(mask, image_shape, anti_aliasing=False, order=0)
        if se is not None:
            for it_shape in range(mask.shape[-1]):
                if mask_index[it_shape] > 0:
                    mask[..., it_shape] = np.double(binary_dilation(mask[..., it_shape], se))
                else:
                    mask[..., it_shape] = 0

        obs_mask[..., nk[-2]:nk[-1]] = mask

        w[nk[-2]:nk[-1], :nslices-zneighbour] += -1 * np.diag(mask_index.astype('uint8'))
        w[nk[-2]:nk[-1], zneighbour:nslices] += 1 * np.diag(mask_index.astype('uint8'))
        for it_sl in range(zneighbour):
            w[nk[-2]:nk[-1], nslices + it_sl:2*nslices - zneighbour + it_sl] += 1 * np.diag(mask_index.astype('uint8'))
        d_C1[nk[-2]:nk[-1]] = zneighbour * mask_index.astype('uint8')


    phi = phi[..., :nk[-1]]
    obs_mask = obs_mask[..., :nk[-1]]
    w = w[:nk[-1]]
    d_inter = d_inter[:nk[-1]]
    d_Ref = d_Ref[:nk[-1]]
    d_C1 = d_C1[:nk[-1]]

    return phi, obs_mask, w, d_inter, d_Ref, d_C1, nk


# Read st3 graph
def init_st3(input_dir, image_shape, nslices, nneighbours=2, se=None, ref='MRI', c1='LFB', c2='HE'):

    nk = [0]

    w = np.zeros((nslices*3*(1+nneighbours), 3*nslices-1), dtype='int')
    obs_mask = np.zeros(image_shape + (nslices*3*(1+nneighbours),))

    d_Ref = np.zeros((nslices*3*(1+nneighbours),), dtype='int')
    d_C1 = np.zeros((nslices*3*(1+nneighbours),), dtype='int')
    d_C2 = np.zeros((nslices*3*(1+nneighbours),), dtype='int')
    d_inter = np.zeros((nslices*3*(1+nneighbours),), dtype='int')

    phi = np.zeros((2,) + image_shape + (3*nslices*(1+2*nneighbours),))

    # Intermodality
    for modality in range(2):
        if modality == 0:
            filename = c1
        else:
            filename = c2

        proxy = nib.load(join(input_dir, ref + '_' + filename + '.0N.velocity_field.nii.gz'))
        field = np.asarray(proxy.dataobj)
        phi[0, :, :, modality * nslices:(modality + 1) * nslices] = field[0]#resize(field[0], image_shape, anti_aliasing=True)
        phi[1, :, :, modality * nslices:(modality + 1) * nslices] = field[1]#resize(field[1], image_shape, anti_aliasing=True)

        # Masks
        proxy = nib.load(join(input_dir, ref + '_' + filename + '.0N.mask.nii.gz'))
        mask_mov = np.double(np.asarray(proxy.dataobj) > 0)
        mask_mov = mask_mov[..., :nslices]
        mask_index = np.sum(mask_mov, axis=(0, 1))

        proxy = nib.load(join(input_dir, ref + '.mask.nii.gz'))
        mask_ref = np.double(np.asarray(proxy.dataobj) > 0)
        mask_ref = mask_ref[..., :nslices]

        mask = mask_mov * mask_ref
        mask = resize(mask, image_shape, anti_aliasing=False, order=0)
        if se is not None:
            for it_shape in range(mask.shape[-1]):
                if mask_index[it_shape] > 0:
                    mask[..., it_shape] = np.double(binary_dilation(mask[..., it_shape], se))
                else:
                    mask[..., it_shape] = 0

        obs_mask[..., modality * nslices:(modality + 1) * nslices] = mask
        w[modality * nslices:(modality + 1) * nslices, modality * nslices:(modality + 1) * nslices] += 1 * np.eye(nslices, dtype=int)
        d_inter[modality * nslices:(modality + 1) * nslices] = 1

        nk.append((modality + 1) * nslices)

    filename = c1 + '_' + c2
    proxy = nib.load(join(input_dir, filename + '.0N.velocity_field.nii.gz'))
    field = np.asarray(proxy.dataobj)
    phi[0, :, :, 2 * nslices:3 * nslices] = field[0]#resize(field[0], image_shape, anti_aliasing=True)
    phi[1, :, :, 2 * nslices:3 * nslices] = field[1]#resize(field[1], image_shape, anti_aliasing=True)

    # Masks
    proxy = nib.load(join(input_dir, filename + '.0N.mask.nii.gz'))
    mask_mov = np.double(np.asarray(proxy.dataobj) > 0)
    mask_mov = mask_mov[..., :nslices]

    proxy = nib.load(join(input_dir, c1 + '.mask.nii.gz'))
    mask_ref = np.double(np.asarray(proxy.dataobj) > 0)
    mask_ref = mask_ref[..., :nslices]
    mask_index = np.sum(mask_ref, axis=(0, 1))

    mask = mask_mov * mask_ref
    mask = resize(mask, image_shape, anti_aliasing=False, order=0)
    if se is not None:
        for it_shape in range(mask.shape[-1]):
            if mask_index[it_shape] > 0:
                mask[..., it_shape] = np.double(binary_dilation(mask[..., it_shape], se))
            else:
                mask[..., it_shape] = 0

    obs_mask[..., 2 * nslices:3 * nslices] = mask
    w[2 * nslices:3 * nslices, :nslices] += -1 * np.eye(nslices, dtype=int)
    w[2 * nslices:3 * nslices, nslices:2 * nslices] += 1 * np.eye(nslices, dtype=int)
    d_inter[2 * nslices:3 * nslices] = 1
    nk.append(3 * nslices)

    # Intramodality
    for zneighbour in range(1,1+nneighbours):
        # REF
        filename = ref + '.' + str(zneighbour) + 'N'
        proxy = nib.load(join(input_dir, filename + '.velocity_field.nii.gz'))
        nslices_tmp = nslices - zneighbour
        nk.append(nk[-1] + nslices_tmp)

        field = np.asarray(proxy.dataobj)[..., :nslices_tmp]
        phi[0, :, :, nk[-2]:nk[-1]] = field[0]#resize(field[0], image_shape, anti_aliasing=True)
        phi[1, :, :, nk[-2]:nk[-1]] = field[1]#resize(field[1], image_shape, anti_aliasing=True)

        # Masks
        proxy = nib.load(join(input_dir, filename + '.mask.nii.gz'))
        mask_mov = np.double(np.asarray(proxy.dataobj) > 0)
        mask_mov = mask_mov[..., :nslices-zneighbour]

        proxy = nib.load(join(input_dir, ref + '.mask.nii.gz'))
        mask_ref = np.double(np.asarray(proxy.dataobj) > 0)
        mask_ref = mask_ref[..., :nslices-zneighbour]

        mask_index = (np.sum(mask_mov, axis=(0, 1)) > 0) & (np.sum(mask_ref, axis=(0, 1)) > 0)
        mask = mask_ref * mask_mov
        mask = resize(mask, image_shape, anti_aliasing=False, order=0)

        if se is not None:
            for it_shape in range(mask.shape[-1]):
                if mask_index[it_shape] > 0:
                    mask[..., it_shape] = np.double(binary_dilation(mask[..., it_shape], se))
                else:
                    mask[..., it_shape] = 0

        obs_mask[..., nk[-2]:nk[-1]] = mask
        for it_sl in range(zneighbour):
            w[nk[-2]:nk[-1], 2 * nslices + it_sl:3 * nslices - zneighbour + it_sl] += 1 * np.eye(nslices - zneighbour,
                                                                                               dtype=int)
        d_Ref[nk[-2]:nk[-1]] = zneighbour

        # C1
        filename = c1 + '.' + str(zneighbour) + 'N'
        proxy = nib.load(join(input_dir, filename + '.velocity_field.nii.gz'))
        nslices_tmp = nslices - zneighbour
        nk.append(nk[-1] + nslices_tmp)

        field = np.asarray(proxy.dataobj)[..., :nslices_tmp]
        phi[0, :, :, nk[-2]:nk[-1]] = field[0]#resize(field[0], image_shape, anti_aliasing=True)
        phi[1, :, :, nk[-2]:nk[-1]] = field[1]#resize(field[1], image_shape, anti_aliasing=True)

        # Masks
        proxy = nib.load(join(input_dir, filename + '.mask.nii.gz'))
        mask_mov = np.double(np.asarray(proxy.dataobj) > 0)
        mask_mov = mask_mov[..., :nslices-zneighbour]

        proxy = nib.load(join(input_dir,  c1 + '.mask.nii.gz'))
        mask_ref = np.double(np.asarray(proxy.dataobj) > 0)
        mask_ref = mask_ref[..., :nslices-zneighbour]

        mask_index = (np.sum(mask_mov, axis=(0, 1))>0) & (np.sum(mask_ref, axis=(0, 1))>0)
        mask = mask_ref * mask_mov
        mask = resize(mask, image_shape, anti_aliasing=False, order=0)
        if se is not None:
            for it_shape in range(mask.shape[-1]):
                if mask_index[it_shape] > 0:
                    mask[..., it_shape] = np.double(binary_dilation(mask[..., it_shape], se))
                else:
                    mask[..., it_shape] = 0

        obs_mask[..., nk[-2]:nk[-1]] = mask
        w[nk[-2]:nk[-1], :nslices-zneighbour] += -1 * np.eye(nslices - zneighbour, dtype=int)
        w[nk[-2]:nk[-1], zneighbour:nslices] += 1 * np.eye(nslices - zneighbour, dtype=int)
        for it_sl in range(zneighbour):
            w[nk[-2]:nk[-1], 2*nslices + it_sl :3*nslices-zneighbour + it_sl] += 1 * np.eye(nslices - zneighbour, dtype=int)
        d_C1[nk[-2]:nk[-1]] = zneighbour


        # C2
        filename = c2 + '.' + str(zneighbour) + 'N'
        proxy = nib.load(join(input_dir, filename + '.velocity_field.nii.gz'))
        nslices_tmp = nslices - zneighbour
        nk.append(nk[-1] + nslices_tmp)

        field = np.asarray(proxy.dataobj)[..., :nslices_tmp]
        phi[0, :, :, nk[-2]:nk[-1]] = field[0]#resize(field[0], image_shape, anti_aliasing=True)
        phi[1, :, :, nk[-2]:nk[-1]] = field[1]#resize(field[1], image_shape, anti_aliasing=True)

        # Masks
        proxy = nib.load(join(input_dir, filename + '.mask.nii.gz'))
        mask_mov = np.double(np.asarray(proxy.dataobj) > 0)
        mask_mov = mask_mov[..., :nslices-zneighbour]

        proxy = nib.load(join(input_dir, c2 + '.mask.nii.gz'))
        mask_ref = np.double(np.asarray(proxy.dataobj) > 0)
        mask_ref = mask_ref[..., :nslices-zneighbour]

        mask_index = (np.sum(mask_mov, axis=(0, 1)) > 0) & (np.sum(mask_ref, axis=(0, 1)) > 0)
        mask = mask_ref * mask_mov
        mask = resize(mask, image_shape, anti_aliasing=False, order=0)
        if se is not None:
            for it_shape in range(mask.shape[-1]):
                if mask_index[it_shape] > 0:
                    mask[..., it_shape] = np.double(binary_dilation(mask[..., it_shape], se))
                else:
                    mask[..., it_shape] = 0

        obs_mask[..., nk[-2]:nk[-1]] = mask
        w[nk[-2]:nk[-1], nslices:2 * nslices - zneighbour] += -1 * np.eye(nslices - zneighbour, dtype=int)
        w[nk[-2]:nk[-1], nslices + zneighbour:2 * nslices] += 1 * np.eye(nslices - zneighbour, dtype=int)
        for it_sl in range(zneighbour):
            w[nk[-2]:nk[-1], 2 * nslices + it_sl: 3 * nslices - zneighbour + it_sl] += 1 * np.eye(nslices - zneighbour, dtype=int)
        d_C2[nk[-2]:nk[-1]] = zneighbour

    phi = phi[..., :nk[-1]]
    obs_mask = obs_mask[..., :nk[-1]]
    w = w[:nk[-1]]
    d_inter = d_inter[:nk[-1]]
    d_Ref = d_Ref[:nk[-1]]
    d_C1 = d_C1[:nk[-1]]
    d_C2 = d_C2[:nk[-1]]

    return phi, obs_mask, w, d_inter, d_Ref, d_C1, d_C2, nk


# Gaussian: l2-loss
def st3_L2(phi, obs_mask, w, d_inter, d_Ref, d_C1, d_C2, nslices, niter=5):

    vars = np.zeros(4)
    image_shape = obs_mask.shape[:2]

    #Initialize transforms
    Tres = np.zeros(phi.shape[:3] + (3*nslices-1,))
    Tres[..., :nslices] = phi[..., :nslices]
    Tres[..., nslices:2*nslices] = phi[..., nslices:2*nslices]
    Tres[..., 2*nslices:] = phi[..., 3*nslices:4*nslices-1]

    for it_iter in range(niter):
        print('Iteration ' + str(it_iter) + '/' + str(niter))

        # Initialize error and number of pixels
        err = phi - np.dot(Tres, w.T)
        E = np.sum(np.sum(np.sum(err*err,axis=0), axis=0), axis=0)
        Npix = np.sum(np.sum(obs_mask, axis=0), axis=0)

        # Variances
        if it_iter == 0:
            vars[0] = 5
            vars[1] = 5
            vars[2] = 5
            vars[3] = 50

        else:
            index_Ref = [idx for idx in np.where(d_Ref>0)[0] if Npix[idx] > 0]
            index_C1 = [idx for idx in np.where(d_C1>0)[0] if Npix[idx] > 0]
            index_C2 = [idx for idx in np.where(d_C2>0)[0] if Npix[idx] > 0]
            index_inter = [idx for idx in np.where(d_inter>0)[0] if Npix[idx] > 0]

            vars[0] = np.sum(E[index_Ref] / (2*Npix[index_Ref]))
            vars[1] = np.sum(E[index_C1] / (2*Npix[index_C1]))
            vars[2] = np.sum(E[index_C2] / (2*Npix[index_C2]))
            vars[3] = np.sum(E[index_inter] / (2*Npix[index_inter]))


        varsK = vars[0]*d_Ref + vars[1]*d_C1 + vars[2]*d_C2 + vars[3]*d_inter

        cost = np.sum( Npix * np.log(2*np.pi*varsK) - 0.5*E/varsK )
        print('    Cost at this iteration: ' + str(np.round(cost,2)), flush=True)
        print('    Variances: ' + str(vars), flush=True)

        print('    Computing weights and updating the transforms')
        precision = 1e-15
        for it_control_row in range(image_shape[0]):

            if np.mod(it_control_row, 10) == 0:
                print('       Row ' + str(it_control_row) + '/' + str(image_shape[0]))

            for it_control_col in range(image_shape[1]):
                index_obs = np.where(obs_mask[it_control_row, it_control_col, :] == 1)[0]
                if index_obs.shape[0] == 0:
                    Tres[:, it_control_row, it_control_col] = 0
                else:
                    w_control = w[index_obs]
                    phi_control = phi[:,it_control_row, it_control_col,index_obs]
                    varsK_control = varsK[index_obs]

                    # Divide each observation by its variance
                    matrix_tmp = w_control.T / varsK_control

                    # matrix_tmp_2: it builds the generative process (y = X * a)
                    #    *in the diagonal* roughly counts the number of observations of each latent variable.
                    #                      it gives more weight in the current observation
                    #    *outside the diagonal* signed number of relations between latent variables to weight
                    #                           the different observations, building a window-like filter
                    #                           (different basis functions for each observation).
                    #    *varsK_control* > 1 decreases the weight of the osbservation and viceversa for < 1.
                    #    *precision* values add "noisy" observations at each latent dimension (high variance).
                    matrix_tmp_2 = np.dot(matrix_tmp, w_control) + precision*np.eye(3*nslices-1)

                    #invert the basis functions to "deconvolve" Xinv·y = a
                    lambda_control = np.dot(np.linalg.inv(matrix_tmp_2),matrix_tmp)

                    for it_tf in range(3*nslices-1):

                        Tres[0, it_control_row, it_control_col, it_tf] = np.dot(lambda_control[it_tf], phi_control[0].T)
                        Tres[1, it_control_row, it_control_col, it_tf] = np.dot(lambda_control[it_tf], phi_control[1].T)

    return Tres


# Laplacian: l1-loss
def st3_L1(phi, obs_mask, w, nslices):

    image_shape = obs_mask.shape[:2]
    Tres = np.zeros(phi.shape[:3] + (3*nslices-1,))
    for it_control_row in range(image_shape[0]):
        if np.mod(it_control_row, 10) == 0:
            print('    Row ' + str(it_control_row) + '/' + str(image_shape[0]))

        for it_control_col in range(image_shape[1]):
            index_obs = np.where(obs_mask[it_control_row, it_control_col, :] == 1)[0]
            if index_obs.shape[0] == 0:
                Tres[:,it_control_row, it_control_col] = 0
            else:
                w_control = w[index_obs]
                phi_control = phi[:, it_control_row, it_control_col, index_obs]
                n_control = len(index_obs)

                for it_dim in range(2):
                    model = gp.Model('LP')
                    model.setParam('OutputFlag', False)
                    model.setParam('Method', 1)

                    # Set the parameters
                    params = model.addMVar(shape=n_control + 3 * nslices - 1, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name='x')

                    # Set objective
                    c_lp = np.concatenate((np.ones((n_control,)), np.zeros((3 * nslices - 1,))), axis=0)
                    model.setObjective(c_lp @ params, gp.GRB.MINIMIZE)

                    # Set the inequality
                    A_lp = np.zeros((2 * n_control, n_control + 3 * nslices - 1))
                    A_lp[:n_control, :n_control] = -np.eye(n_control)
                    A_lp[:n_control, n_control:] = -w_control
                    A_lp[n_control:, :n_control] = -np.eye(n_control)
                    A_lp[n_control:, n_control:] = w_control
                    A_lp = sp.csr_matrix(A_lp)

                    reg = np.reshape(phi_control[it_dim], (n_control,))
                    b_lp = np.concatenate((-reg, reg), axis=0)

                    model.addConstr(A_lp @ params <= b_lp, name="c")

                    model.optimize()

                    Tres[it_dim, it_control_row, it_control_col] = params.X[n_control:]

    return Tres


# Gaussian: l2-loss
def st2_L2(phi, obs_mask, w, delta, d_Ref, d_C1, nslices, niter=5):

    vars = np.zeros(3)
    image_shape = obs_mask.shape[:2]

    #Initialize transforms
    Tres = np.zeros(phi.shape[:3] + (2*nslices-1,))
    Tres[..., :nslices] = phi[..., :nslices]
    Tres[..., nslices:] = phi[..., nslices:2*nslices]

    for it_iter in range(niter):
        print('Iteration ' + str(it_iter) + '/' + str(niter))

        # Initialize error and number of pixels
        err = phi - np.dot(Tres, w.T)
        E = np.sum(np.sum(np.sum(err*err,axis=0), axis=0), axis=0)
        Npix = np.sum(np.sum(obs_mask, axis=0), axis=0)

        # Variances
        if it_iter == 0:
            vars[0] = 5
            vars[1] = 5
            vars[2] = 5

        else:

            index_Ref = [idx for idx in np.where(d_Ref>0)[0] if Npix[idx] > 0]
            index_C1 = [idx for idx in np.where(d_C1>0)[0] if Npix[idx] > 0]
            index_delta = [idx for idx in np.where(delta>0)[0] if Npix[idx] > 0]

            vars[0] = np.sum(E[index_Ref] / (2*Npix[index_Ref]))
            vars[1] = np.sum(E[index_C1] / (2*Npix[index_C1]))
            vars[2] = np.sum(E[index_delta] / (2*Npix[index_delta]))


        varsK = vars[0]*d_Ref + vars[1]*d_C1 + vars[2]*delta
        cost = np.sum( Npix * np.log(2*np.pi*varsK) - 0.5*E/varsK )
        print('    Cost at this iteration: ' + str(np.round(cost,2)), flush=True)
        print('    Variances: ' + str(vars), flush=True)

        print('    Computing weights and updating the transforms')
        precision = 1e-15
        for it_control_row in range(image_shape[0]):

            if np.mod(it_control_row, 10) == 0:
                print('       Row ' + str(it_control_row) + '/' + str(image_shape[0]))

            for it_control_col in range(image_shape[1]):
                index_obs = np.where(obs_mask[it_control_row, it_control_col, :] == 1)[0]
                if index_obs.shape[0] == 0:
                    Tres[:, it_control_row, it_control_col] = 0
                else:
                    w_control = w[index_obs]
                    phi_control = phi[:,it_control_row, it_control_col,index_obs]
                    varsK_control = varsK[index_obs]


                    # print('Please, first try division instead of diagonal matrix!!!!!!!')
                    matrix_tmp = w_control.T / varsK_control
                    matrix_tmp_2 = np.dot(matrix_tmp, w_control) + precision*np.eye(2*nslices-1)
                    lambda_control = np.dot(np.linalg.inv(matrix_tmp_2),matrix_tmp)

                    for it_tf in range(2*nslices-1):

                        Tres[0, it_control_row, it_control_col, it_tf] = np.dot(lambda_control[it_tf], phi_control[0].T)
                        Tres[1, it_control_row, it_control_col, it_tf] = np.dot(lambda_control[it_tf], phi_control[1].T)

    return Tres


# Laplacian: l1-loss
def st2_L1(phi, obs_mask, w, nslices):

    image_shape = obs_mask.shape[:2]
    Tres = np.zeros(phi.shape[:3] + (2*nslices-1,))
    for it_control_row in range(image_shape[0]):
        if np.mod(it_control_row, 10) == 0:
            print('    Row ' + str(it_control_row) + '/' + str(image_shape[0]))

        for it_control_col in range(image_shape[1]):
            index_obs = np.where(obs_mask[it_control_row, it_control_col, :] == 1)[0]
            if index_obs.shape[0] == 0:
                Tres[:,it_control_row, it_control_col] = 0
            else:
                w_control = w[index_obs]
                phi_control = phi[:, it_control_row, it_control_col, index_obs]
                n_control = len(index_obs)

                for it_dim in range(2):
                    model = gp.Model('LP')
                    model.setParam('OutputFlag', False)
                    model.setParam('Method', 1)

                    # Set the parameters
                    params = model.addMVar(shape=n_control + 2 * nslices - 1, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                                           name='x')

                    # Set objective
                    c_lp = np.concatenate((np.ones((n_control,)), np.zeros((2 * nslices - 1,))), axis=0)
                    model.setObjective(c_lp @ params, gp.GRB.MINIMIZE)

                    # Set the inequality
                    A_lp = np.zeros((2 * n_control, n_control + 2 * nslices - 1))
                    A_lp[:n_control, :n_control] = -np.eye(n_control)
                    A_lp[:n_control, n_control:] = -w_control
                    A_lp[n_control:, :n_control] = -np.eye(n_control)
                    A_lp[n_control:, n_control:] = w_control
                    A_lp = sp.csr_matrix(A_lp)

                    reg = np.reshape(phi_control[it_dim], (n_control,))
                    b_lp = np.concatenate((-reg, reg), axis=0)

                    model.addConstr(A_lp @ params <= b_lp, name="c")

                    model.optimize()

                    Tres[it_dim, it_control_row, it_control_col] = params.X[n_control:]

    return Tres
