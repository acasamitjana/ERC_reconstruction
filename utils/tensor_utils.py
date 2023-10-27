import itertools

import numpy as np
import torch


#############
# Functions #
#############

def flatten(v):
    """
    flatten Tensor v

    Parameters:
        v: Tensor to be flattened

    Returns:
        flat Tensor
    """

    return v.reshape(-1)

def volshape_to_meshgrid(volshape, **kwargs):
    """
    compute Tensor meshgrid from a volume size

    Parameters:
        volshape: the volume size
        **args: "name" (optional)

    Returns:
        A list of Tensors

    See Also:
        tf.meshgrid, meshgrid, ndgrid, volshape_to_ndgrid
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [torch.arange(0, d) for d in volshape]
    r, c = torch.meshgrid(*linvec)
    return (r,c)

def interpn(vol, loc, interp_method='linear'):
    """
    N-D gridded interpolation in tensorflow

    vol can have more dimensions than loc[i], in which case loc[i] acts as a slice
    for the first dimensions

    Parameters:
        vol: volume with size vol_shape or [nb_features, *vol_shape]
        loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
            each tensor has to have the same size (but not nec. same size as vol)
            or a tensor of size [*new_vol_shape, D]
        interp_method: interpolation type 'linear' (default) or 'nearest'

    Returns:
        new interpolated volume of the same size as the entries in loc

    TODO:
        enable optional orig_grid - the original grid points.
        check out tf.contrib.resampler, only seems to work for 2D data
    """

    if isinstance(loc, (list, tuple)):
        loc = torch.stack(loc, dim=0)
    nb_dims = loc.shape[0]

    if len(vol.shape) not in [nb_dims, nb_dims + 1]:
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[1:])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = vol.expand((1,) + vol.shape)

    # flatten and float location Tensors
    loc = loc.type(torch.float32)

    volshape = vol.shape
    #
    # slices_2d = [loc[0], loc[1], vol[0]]
    # titles = ['loc', 'loc', 'vol']
    # slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True)

    # interpolate
    if interp_method == 'linear':
        loc0 = torch.floor(loc)

        # clip values
        max_loc = [d - 1 for d in vol.shape[1:]]
        clipped_loc = [torch.clamp(loc[d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [torch.clamp(loc0[d], 0, max_loc[d]) for d in range(nb_dims)]

        # get other end of point cube
        loc1 = [torch.clamp(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[f.type(torch.int32) for f in loc0lst], [f.type(torch.int32) for f in loc1]]

        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0]  # note reverse ordering since weights are inverse of diff.
        # go through all the cube corners, indexed by a ND binary vector
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0

        for c in cube_pts:
            # get nd values
            # note re: indices above volumes via https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's too
            #   expensive. Instead we fill the output with zero for the corresponding value. The CPU
            #   version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
            # indices = tf.stack(subs, axis=-1)
            # vol_val = tf.gather_nd(vol, indices)
            # faster way to gather than gather_nd, because the latter needs tf.stack which is slow :(
            idx = sub2ind(vol.shape[1:], subs)
            idx = idx.view([volshape[0],-1])

            vol_reshape = vol.reshape([volshape[0],-1])
            vol_val = torch.gather(vol_reshape, dim=1, index=idx).view(volshape[1:])
            # indices = torch.stack(subs, dim=0).type(torch.LongTensor)
            # print(indices[0].shape)
            # vol_val = vol[indices[0], indices[1]]

            # get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            # tf stacking is slow, we will use prod_n()
            # wlm = tf.stack(wts_lst, axis=0)
            # wt = tf.reduce_prod(wlm, axis=0)
            wt = prod_n(wts_lst)
            # wt = wt.expand((1,) + vol.shape)

            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val

    else:
        assert interp_method == 'nearest'
        roundloc = torch.round(loc).type('int32')

        # clip values
        max_loc = [(d - 1).type(torch.int32) for d in vol.shape]
        roundloc = [torch.clamp(roundloc[d], 0, max_loc[d]) for d in range(nb_dims)]

        # get values
        # tf stacking is slow. replace with gather
        # roundloc = tf.stack(roundloc, axis=-1)
        # interp_vol = tf.gather_nd(vol, roundloc)
        idx = sub2ind(vol.shape[1:], roundloc)
        interp_vol = torch.gather(vol.reshape([-1, vol.shape[-1]]), dim=0, index=idx)

    return interp_vol

def prod_n(lst):
    """
    Alternative to tf.stacking and prod, since tf.stacking can be slow
    """
    prod = lst[0].clone()
    for p in lst[1:]:
        prod *= p
    return prod

def sub2ind(siz, subs, **kwargs):
    """
    assumes column-order major
    """

    # subs is a list
    assert len(siz) == len(subs), \
        'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1]).astype('int32')
    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    ndx = ndx.type(torch.LongTensor)

    return ndx

##################
# Transformation #
##################

def affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij'):
    """
    transform an affine matrix to a dense location shift tensor in pytorch

    Algorithm:
        - get grid and shift grid to be centered at the center of the image (optionally)
        - apply affine matrix to each index.
        - subtract grid

    Parameters:
        affine_matrix: ND+1 x ND+1 or ND x ND+1 matrix (Tensor)
        volshape: 1xN Nd Tensor of the size of the volume.
        shift_center (optional)

    Returns:
        shift field (Tensor) of size *volshape x N

    This is based on (a.k.a. copied from) neuron code, so for more information and credit
    visit https://github.com/adalca/neuron/blob/master/neuron/utils.py

    """

    if affine_matrix.dtype != torch.float32:
        affine_matrix = affine_matrix.type(torch.float32)

    nb_dims = len(volshape)

    if len(affine_matrix.shape) == 1:
        if len(affine_matrix) != (nb_dims * (nb_dims + 1)):
            raise ValueError('transform is supposed a vector of len ndims * (ndims + 1).'
                             'Got len %d' % len(affine_matrix))

        affine_matrix = affine_matrix.view(*[nb_dims, nb_dims + 1])

    if not (affine_matrix.shape[0] in [nb_dims, nb_dims + 1] and affine_matrix.shape[1] == (nb_dims + 1)):
        raise Exception('RigidRegistration matrix shape should match'
                        '%d+1 x %d+1 or ' % (nb_dims, nb_dims) + \
                        '%d x %d+1.' % (nb_dims, nb_dims) + \
                        'Got: ' + str(volshape))

    # list of volume ndgrid
    # N-long list, each entry of shape volshape
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    mesh = [f.type(torch.float32) for f in mesh]

    if shift_center:
        mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(len(volshape))]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [flatten(f) for f in mesh]
    flat_mesh.append(torch.ones(flat_mesh[0].shape, dtype=torch.float32))
    mesh_matrix = torch.transpose(torch.stack(flat_mesh, dim=1), dim0=0, dim1=1)  # 4 x nb_voxels

    # compute locations
    loc_matrix = torch.matmul(affine_matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = torch.transpose(loc_matrix[:nb_dims, :], dim0=0, dim1=1)  # nb_voxels x N
    loc = loc_matrix.reshape(list(volshape) + [nb_dims])  # *volshape x N
    # loc = [loc[..., f] for f in range(nb_dims)]  # N-long list, each entry of shape volshape

    # get shifts and return
    return loc - torch.stack(mesh, dim=nb_dims)

