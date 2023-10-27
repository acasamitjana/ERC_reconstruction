import numpy as np
import csv
import pdb

from skimage.transform import rotate as imrotate, SimilarityTransform, warp
from utils.image_utils import align_with_identity_vox2ras0
from utils.deformation_utils import interpolate2D

def get_num_slices_dict(file):
    block_dict = {}
    with open(file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if row['BLOCK_ID'] in block_dict.keys():
                block_dict[row['BLOCK_ID']].append(row['SLICE_ID'])
            else:
                block_dict[row['BLOCK_ID']] = [row['SLICE_ID']]

    return block_dict


class Case(object):

    def get_vox2ras0(self, vox2ras0, block_id, blockshape, v2r_ref=np.eye(4)):
        return vox2ras0

    def change_orientation(self, slice_i, block_id, vox2ras0, v2r_ref=np.eye(4)):
        return slice_i


class P57(Case):

    def get_vox2ras0(self, vox2ras0, block_id, blockshape, v2r_ref=np.eye(4), nslices=1):


        # if block_id in ['B4.1', 'B3.1']:
        #     # flipud
        #     m_flipud = np.eye(4)
        #     m_flipud[0, 0] = -1
        #     m_flipud[0, 3] = blockshape[0] - 1
        #     new_vox2ras0 = vox2ras0 @ m_flipud
        #     return new_vox2ras0

        # elif block_id in ['B5.1']:
        #     # flipud
        #     m_flipud = np.eye(4)
        #     m_fliplr = np.eye(4)
        #     # m_fliplr[0, 0] = -1
        #     m_fliplr[1, 1] = -1
        #     # m_fliplr[0, 3] = blockshape[0] - 1
        #     m_fliplr[1, 3] = blockshape[1] - 1
        #     new_vox2ras0 = vox2ras0 @ m_fliplr
        #     return new_vox2ras0
        # elif block_id == 'P8.1':
        #     # transpose
        #     m_flipud = np.eye(4)
        #     m_flipud[0, 0] = -1
        #     m_rot = np.zeros((4, 4))
        #     m_rot[0, 1] = -1
        #     m_rot[1, 0] = 1
        #     m_rot[2, 2] = 1
        #     m_rot[3, 3] = 1
        #     m_trans = np.eye(4)
        #     m_trans[0, -1] = -blockshape[0] // 2
        #     m_trans[1, -1] = -blockshape[1] // 2
        #     m_trans_inv = np.eye(4)
        #     m_trans_inv[0, -1] = blockshape[0] // 2
        #     m_trans_inv[1, -1] = blockshape[1] // 2
        #     m_compensate = np.eye(4)
        #     m_compensate[0, -1] = -(blockshape[0] - blockshape[1]) // 2
        #     m_compensate[1, -1] = (blockshape[0] - blockshape[1]) // 2
        #     new_vox2ras0 = vox2ras0 @ m_compensate @ m_trans_inv @ m_flipud @ m_rot @ m_trans
        #     return new_vox2ras0
        # elif 'C' in block_id or block_id in ['B1.1', 'B2.1']:
        #     return vox2ras0
        # else:
        #     m_fliplr = np.eye(4)
        #     m_fliplr[1, 1] = -1
        #     m_fliplr[1, 3] = blockshape[1] - 1
        #     new_vox2ras0 = vox2ras0 @ m_fliplr
        #     return new_vox2ras0

        if 'C' in block_id:
            dim = 1
        else:
            dim = 0
        dim_histo = 1
        proj_z = np.dot(vox2ras0[:3, dim_histo], v2r_ref[:3, dim]) / (np.linalg.norm(vox2ras0[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim]))
        angle = np.arccos(proj_z)

        if proj_z < 0: angle = -angle
        if block_id in ['A1.1', 'A3.2', 'P3.1', 'P3.3', 'A3.1', 'P1.4', 'A1.4']: angle = -angle
        if block_id in ['A1.3', 'P5.1', 'P1.3', 'B3.1', 'B4.1']: angle = 0
        if block_id in ['B5.1']: angle += (180 - 55)/180*np.pi
        if block_id in ['C1.1', 'C2.1', 'C3.1', 'C4.1']: angle = -angle

        center = np.array((blockshape[1], blockshape[0])) / 2. - 0.5
        tform1 = SimilarityTransform(translation=center)
        tform2 = SimilarityTransform(rotation=angle)
        tform3 = SimilarityTransform(translation=-center)
        tform = tform3 + tform2 + tform1
        rows, cols = blockshape[0], blockshape[1]
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1

        # fit output image in new shape
        output_shape = np.around((out_rows, out_cols))

        m_center = np.eye(4)
        m_center[0, -1] = -blockshape[0] / 2. - 0.5
        m_center[1, -1] = -blockshape[1] / 2. - 0.5

        m_center_inv = np.eye(4)
        m_center_inv[0, -1] = blockshape[0] / 2. - 0.5
        m_center_inv[1, -1] = blockshape[1] / 2. - 0.5

        m_tx = np.eye(4)
        m_tx[0, -1] = minr
        m_tx[1, -1] = minc

        m_rotate = np.eye(4)
        m_rotate[0, 0] = np.cos(angle)
        m_rotate[0, 1] = np.sin(angle)
        m_rotate[1, 0] = -np.sin(angle)
        m_rotate[1, 1] = np.cos(angle)

        m_fliplr = np.eye(4)
        m_flipud = np.eye(4)
        if 'C' not in block_id and block_id not in ['B3.1', 'B4.1']:
            m_fliplr[1, 1] = -1
            m_fliplr[1, 3] = output_shape[1] - 1

        if block_id in ['C5.1', 'B1.1', 'B2.1', 'B3.1', 'B4.1', 'B5.1']:
            m_flipud[0, 0] = -1
            m_flipud[0, -1] = output_shape[0] - 1

        m_tz = np.eye(4)
        if block_id in ['P8.1', 'C5.1']:
            m_tz[2, 2] = -1
            m_tz[2, -1] = nslices - 1
        new_vox2ras0 = vox2ras0 @ m_tz @ m_center_inv @ m_rotate @ m_center @ m_tx @ m_fliplr @ m_flipud

        return new_vox2ras0

    def change_orientation(self, slice_i, block_id, vox2ras0=None, v2r_ref=np.eye(4), order=1):
        # if block_id == 'P8.1':
        #     if len(slice_i.shape) > 2:
        #         slice_out = np.zeros((slice_i.shape[1], slice_i.shape[0], slice_i.shape[2]), dtype=slice_i.dtype)
        #         for itc in range(slice_i.shape[2]):
        #             slice_out[..., itc] = slice_i[..., itc].T
        #         slice_i = slice_out
        #     else:
        #         slice_i = slice_i.T
        #     return slice_i

        # if block_id in ['B3.1', 'B4.1']:
        #     slice_i = np.flipud(slice_i)
        #     return slice_i

        # elif block_id in ['B5.1']:
        #     slice_i = np.fliplr(slice_i)
        #     return slice_i

        # elif 'C' in block_id or block_id in ['B1.1', 'B2.1']:
        #     return slice_i

        # else:
        #
        #     dim = 0
        #     dim_histo = 1
        #     proj_z = np.dot(vox2ras0[:3, dim_histo], v2r_ref[:3, dim]) / (
        #             np.linalg.norm(vox2ras0[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim]))
        #     angle = np.arccos(proj_z)
        #     print(proj_z)
        #     print(angle*180/np.pi)
        #
        #
        #     slice_i = np.fliplr(slice_i)
        init_dtype = slice_i.dtype
        slice_i = np.double(slice_i)
        data = []
        if 'C' in block_id:
            dim = 1
        else:
            dim = 0
        dim_histo = 1
        proj_z = np.dot(vox2ras0[:3, dim_histo], v2r_ref[:3, dim]) / (np.linalg.norm(vox2ras0[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim]))
        angle = np.arccos(proj_z)
        if proj_z < 0: angle = -angle
        if block_id in ['A1.1', 'A3.2', 'P3.1', 'P3.3', 'A3.1', 'P1.4', 'A1.4']: angle = -angle
        if block_id in ['A1.3', 'P5.1', 'P1.3', 'B3.1', 'B4.1']: angle = 0
        if block_id in ['B5.1']: angle += (180 - 55)/180*np.pi
        if block_id in ['C1.1', 'C2.1', 'C3.1', 'C4.1']: angle = -angle

        if len(slice_i.shape) == 3:
            for it_c in range(3):
                d = imrotate(slice_i[..., it_c], angle / np.pi * 180, resize=True, order=order)
                if 'C' not in block_id and block_id not in ['B3.1', 'B4.1']: d = np.fliplr(d)
                if block_id in ['C5.1']: d = np.flipud(d)
                d = d.astype(init_dtype)
                data.append(d)

            slice_i = np.stack([d for d in data], axis=-1)

        else:
            d = imrotate(slice_i, angle / np.pi * 180, resize=True, order=order)
            if 'C' not in block_id and block_id not in ['B3.1', 'B4.1']: d = np.fliplr(d)
            if block_id in ['C5.1']: d = np.flipud(d)

            slice_i = d.astype(init_dtype)

        return slice_i

class P41(Case):

    def get_vox2ras0(self, vox2ras0, block_id, blockshape, v2r_ref=np.eye(4), nslices=1):
        # if block_id in ['B1.1', 'B2.1', 'B3.1', 'B4.1']:
        #     # flipud
        #     m_flipud = np.eye(4)
        #     m_flipud[0, 0] = -1
        #     m_flipud[0, 3] = blockshape[0] - 1
        #
        #     m_fliplr = np.eye(4)
        #     m_fliplr[0, 0] = -1
        #     m_fliplr[1, 1] = -1
        #     m_fliplr[0, 3] = blockshape[0] - 1
        #     m_fliplr[1, 3] = blockshape[1] - 1
        #     new_vox2ras0 = vox2ras0 @ m_fliplr @ m_flipud
        #
        #     return new_vox2ras0

        # elif 'C' in block_id:# or block_id == 'P8.1':
        #     return vox2ras0

        if 'C' in block_id:
            dim = 1
        else:
            dim = 0

        dim_histo = 1
        proj_z = np.dot(vox2ras0[:3, dim_histo], v2r_ref[:3, dim]) / (np.linalg.norm(vox2ras0[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim]))
        angle = np.arccos(proj_z)

        if proj_z < 0 and block_id != 'P9.1': angle = -angle
        if block_id == 'P3.2': angle = angle - np.pi/2
        if block_id == 'P3.3': angle = -angle# - np.pi/4
        # if block_id == 'P4.1':  angle = -angle
        if block_id in ['P6.1', 'P6.2', 'B1.1', 'B2.1', 'B3.1', 'B4.1']: angle = 0
        if block_id in ['C1.1', 'C2.1', 'C3.1']: angle = -angle

        center = np.array((blockshape[1], blockshape[0])) / 2. - 0.5
        tform1 = SimilarityTransform(translation=center)
        tform2 = SimilarityTransform(rotation=angle)
        tform3 = SimilarityTransform(translation=-center)
        tform = tform3 + tform2 + tform1
        rows, cols = blockshape[0], blockshape[1]
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1

        # fit output image in new shape
        output_shape = np.around((out_rows, out_cols))

        m_center = np.eye(4)
        m_center[0, -1] = -blockshape[0]/2. - 0.5
        m_center[1, -1] = -blockshape[1]/2. - 0.5

        m_center_inv = np.eye(4)
        m_center_inv[0, -1] = blockshape[0]/2. - 0.5
        m_center_inv[1, -1] = blockshape[1]/2. - 0.5

        m_tx = np.eye(4)
        m_tx[0, -1] = minr
        m_tx[1, -1] = minc

        m_rotate = np.eye(4)
        m_rotate[0,0] = np.cos(angle)
        m_rotate[0,1] = np.sin(angle)
        m_rotate[1,0] = -np.sin(angle)
        m_rotate[1,1] = np.cos(angle)

        m_fliplr = np.eye(4)
        if 'C' not in block_id:
            m_fliplr[1, 1] = -1
            m_fliplr[1, -1] = output_shape[1] - 1

        m_flipud = np.eye(4)
        if block_id in ['P9.1', 'C4.1']:
            m_flipud[0, 0] = -1
            m_flipud[0, -1] = output_shape[0] - 1

        m_tz = np.eye(4)
        if block_id in ['P9.1', 'C4.1']:
            m_tz[2, 2] = -1
            m_tz[2, -1] = nslices - 1

        new_vox2ras0 = vox2ras0 @ m_tz @ m_center_inv @ m_rotate @ m_center @ m_tx @ m_fliplr @ m_flipud



        return new_vox2ras0

    def change_orientation(self, slice_i, block_id, vox2ras0, v2r_ref=np.eye(4), order=1):

        # if block_id in ['B1.1', 'B2.1', 'B3.1', 'B4.1']:
        #     slice_i = np.flipud(slice_i)
        #     slice_i = np.fliplr(slice_i)
        #     return slice_i

        # elif 'C' in block_id:# or block_id == 'P8.1':
        #     return slice_i

        init_dtype = slice_i.dtype
        slice_i = np.double(slice_i)
        data = []

        if 'C' in block_id:
            dim = 1
        else:
            dim = 0
        dim_histo = 1
        proj_z = np.dot(vox2ras0[:3, dim_histo], v2r_ref[:3, dim]) / (np.linalg.norm(vox2ras0[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim]))
        angle = np.arccos(proj_z)

        if proj_z < 0 and block_id != 'P9.1': angle = -angle
        if block_id == 'P3.2': angle = angle - np.pi/2
        if block_id == 'P3.3': angle = -angle# - np.pi/4
        if block_id in ['P4.1']: angle = -angle
        if block_id in ['P6.1', 'P6.2', 'B1.1', 'B2.1', 'B3.1', 'B4.1']: angle = 0
        if block_id in ['C1.1', 'C2.1', 'C3.1']: angle = -angle

        if len(slice_i.shape) == 3:
            for it_c in range(3):
                d = imrotate(slice_i[..., it_c], angle / np.pi * 180, resize=True, order=order)
                if 'C' not in block_id: d = np.fliplr(d)

                if block_id in ['P9.1', 'C4.1']: d = np.flipud(d)
                d = d.astype(init_dtype)
                data.append(d)

            slice_i = np.stack([d for d in data], axis=-1)

        else:
            d = imrotate(slice_i, angle / np.pi * 180, resize=True, order=order)
            if 'C' not in block_id: d = np.fliplr(d)
            if block_id in ['P9.1', 'C4.1']: d = np.flipud(d)

            slice_i = d.astype(init_dtype)


        return slice_i

    # def get_vox2ras0_new(self, vox2ras0, block_id, blockshape, v2r_ref=np.eye(4)):
    #     if 'B' in block_id:
    #         # flipud
    #         m_flipud = np.eye(4)
    #         m_flipud[0, 0] = -1
    #         m_flipud[0, 3] = blockshape[0] - 1
    #
    #         m_fliplr = np.eye(4)
    #         m_fliplr[0, 0] = -1
    #         m_fliplr[1, 1] = -1
    #         m_fliplr[0, 3] = blockshape[0] - 1
    #         m_fliplr[1, 3] = blockshape[1] - 1
    #         new_vox2ras0 = vox2ras0 @ m_fliplr @ m_flipud
    #
    #     elif 'C' in block_id or block_id == 'P8.1':
    #         new_vox2ras0 = vox2ras0
    #
    #     elif block_id == 'P3.1':
    #         data, v2r = align_with_identity_vox2ras0(np.zeros(blockshape + (1,)), vox2ras0)
    #         dim = 0
    #         angle = np.pi / 2 + np.arccos(np.dot(v2r[:3, dim], v2r_ref[:3, dim]) / ( np.linalg.norm(v2r[:3, dim]) * np.linalg.norm(v2r_ref[:3, dim])))
    #
    #         # Get new output shape
    #         corners = np.array([
    #             [0, 0, 0],
    #             [0, data.shape[0] - 1, 0],
    #             [data.shape[2] - 1, data.shape[0] - 1, 0],
    #             [data.shape[2] - 1, 0, 0]
    #         ])
    #
    #         m_center = np.eye(3)
    #         m_center[0, -1] = -data.shape[0] / 2. - 0.5
    #         m_center[1, -1] = -data.shape[1] / 2. - 0.5
    #
    #         m_rotate = np.eye(3)
    #         m_rotate[0, 0] = np.cos(-angle)
    #         m_rotate[0, 1] = -np.sin(-angle)
    #         m_rotate[1, 0] = np.sin(-angle)
    #         m_rotate[1, 1] = np.cos(-angle)
    #
    #         m_center_inv = np.eye(3)
    #         m_center_inv[0, -1] = data.shape[0] / 2. - 0.5
    #         m_center_inv[1, -1] = data.shape[1] / 2. - 0.5
    #
    #         corners_new = m_center_inv @ m_rotate @ m_center @ corners.T
    #         corners_new = corners_new.T
    #         minc = corners_new[:, 0].min()
    #         minr = corners_new[:, 1].min()
    #         tx = (minc, minr)
    #         maxc = corners_new[:, 0].max()
    #         maxr = corners_new[:, 1].max()
    #         out_rows = maxr - minr + 1
    #         out_cols = maxc - minc + 1
    #         output_shape = np.around((out_rows, out_cols))
    #
    #         m_center = np.eye(4)
    #         m_center[0, -1] = -blockshape[0] / 2. - 0.5
    #         m_center[2, -1] = -blockshape[1] / 2. - 0.5
    #
    #         m_center_inv = np.eye(4)
    #         m_center_inv[0, -1] = blockshape[0] / 2. - 0.5
    #         m_center_inv[2, -1] = blockshape[1] / 2. - 0.5
    #
    #         m_tx = np.eye(4)
    #         m_tx[0, -1] = tx[0]
    #         m_tx[2, -1] = tx[1]
    #
    #         m_rotate = np.eye(4)
    #         m_rotate[0,0] = np.cos(-angle)
    #         m_rotate[0,2] = np.sin(-angle)
    #         m_rotate[2,0] = -np.sin(-angle)
    #         m_rotate[2,2] = np.cos(-angle)
    #
    #         m_fliplr = np.eye(4)
    #         m_fliplr[1, 1] = -1
    #         m_fliplr[1, -1] = output_shape[1] - 1
    #
    #         new_vox2ras0 = v2r @ m_center_inv @ m_rotate @ m_center @ m_tx @ m_fliplr
    #
    #     else:
    #         m_fliplr = np.eye(4)
    #         m_fliplr[1, 1] = -1
    #         m_fliplr[1, 3] = blockshape[1] - 1
    #         new_vox2ras0 = vox2ras0 @ m_fliplr
    #
    #     return new_vox2ras0
    #
    # def change_orientation_new(self, slice_i, block_id, vox2ras0, v2r_ref=np.eye(4), order=1):
    #
    #     if 'B' in block_id:
    #         slice_i = np.flipud(slice_i)
    #         slice_i = np.fliplr(slice_i)
    #
    #     elif 'C' in block_id or block_id=='P8.1':
    #         pass
    #
    #     elif block_id == 'P3.1':
    #         init_dtype = slice_i.dtype
    #         slice_i = np.double(slice_i)
    #         data = []
    #         if len(slice_i.shape) == 3:
    #             for it_c in range(3):
    #                 d, v2r = align_with_identity_vox2ras0(slice_i[:,:,it_c,np.newaxis], vox2ras0)
    #                 # v2r = vox2ras0
    #                 dim = 0
    #                 # dim_histo = 0
    #                 # angle = np.arccos(np.dot(v2r[:3, dim_histo], v2r_ref[:3, dim]) / (np.linalg.norm(v2r[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim])))
    #                 angle = np.pi / 2 + np.arccos(np.dot(v2r[:3, dim], v2r_ref[:3, dim]) / (np.linalg.norm(v2r[:3, dim]) * np.linalg.norm(v2r_ref[:3, dim])))
    #                 d = imrotate(d[:, 0], angle / np.pi * 180, resize=True, order=order)
    #                 d = np.fliplr(d)
    #                 d = d.astype(init_dtype)
    #                 data.append(d)
    #
    #             slice_i = np.stack([d for d in data], axis=-1)
    #
    #         else:
    #             d, v2r = align_with_identity_vox2ras0(slice_i[..., np.newaxis], vox2ras0)
    #             dim = 0
    #             angle = np.pi / 2 + np.arccos(np.dot(v2r[:3, dim], v2r_ref[:3, dim]) /
    #                                           (np.linalg.norm(v2r[:3, dim]) * np.linalg.norm(v2r_ref[:3, dim])))
    #
    #             d = imrotate(d[:, 0], angle / np.pi * 180, resize=True, order=order)
    #             d = np.fliplr(d)
    #             slice_i = d.astype(init_dtype)
    #
    #     else:
    #         slice_i = np.fliplr(slice_i)
    #
    #     return slice_i

class P58(Case):

    def get_vox2ras0(self, vox2ras0, block_id, blockshape, v2r_ref=np.eye(4), nslices=1):

        if block_id in ['B1.1', 'B2.1', 'B3.1', 'B4.1', 'B5.1']:
            m_flipud = np.eye(4)
            m_flipud[0, 0] = -1
            m_flipud[0, -1] = blockshape[0] - 1
            new_vox2ras0 = vox2ras0 @  m_flipud
            return new_vox2ras0

        # elif block_id == 'C4.1':
        #     m_flipud = np.eye(4)
        #     m_fliplr = np.eye(4)
        #     m_fliplr[0, 0] = -1
        #     m_fliplr[1, 1] = -1
        #     m_fliplr[0, 3] = blockshape[0] - 1
        #     m_fliplr[1, 3] = blockshape[1] - 1
        #     new_vox2ras0 = vox2ras0 @ m_fliplr @ m_flipud
        #
        #     return new_vox2ras0
        # elif block_id in ['C1.1', 'C2.1', 'C3.1', 'C5.1']:
        #     return vox2ras0

        if 'C' in block_id:
            dim = 1
        else:
            dim = 0
        dim_histo = 1

        proj_z = np.dot(vox2ras0[:3, dim_histo], v2r_ref[:3, dim]) / (np.linalg.norm(vox2ras0[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim]))
        angle = np.arccos(proj_z)

        if proj_z < 0 and block_id != 'P9.1': angle = -angle
        if block_id in ['C2.1', 'C3.1']: angle = -angle

        center = np.array((blockshape[1], blockshape[0])) / 2. - 0.5
        tform1 = SimilarityTransform(translation=center)
        tform2 = SimilarityTransform(rotation=angle)
        tform3 = SimilarityTransform(translation=-center)
        tform = tform3 + tform2 + tform1
        rows, cols = blockshape[0], blockshape[1]
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1

        # fit output image in new shape
        output_shape = np.around((out_rows, out_cols))

        m_center = np.eye(4)
        m_center[0, -1] = -blockshape[0]/2. - 0.5
        m_center[1, -1] = -blockshape[1]/2. - 0.5

        m_center_inv = np.eye(4)
        m_center_inv[0, -1] = blockshape[0]/2. - 0.5
        m_center_inv[1, -1] = blockshape[1]/2. - 0.5

        m_tx = np.eye(4)
        m_tx[0, -1] = minr
        m_tx[1, -1] = minc

        m_rotate = np.eye(4)
        m_rotate[0,0] = np.cos(angle)
        m_rotate[0,1] = np.sin(angle)
        m_rotate[1,0] = -np.sin(angle)
        m_rotate[1,1] = np.cos(angle)


        m_flipud = np.eye(4)
        m_fliplr = np.eye(4)
        if block_id not in ['B1.1', 'B2.1', 'B3.1', 'B4.1', 'C1.1', 'C2.1', 'C3.1', 'C4.1']:
            m_fliplr[1, 1] = -1
            m_fliplr[1, -1] = output_shape[1] - 1

        if block_id in ['P9.1', 'C4.1', 'B1.1', 'B2.1', 'B3.1', 'B4.1', 'B5.1']:
            m_flipud[0, 0] = -1
            m_flipud[0, -1] = output_shape[0] - 1

        m_tz = np.eye(4)
        if block_id in ['P9.1', 'C4.1']:
            m_tz[2, 2] = -1
            m_tz[2, -1] = nslices - 1

        new_vox2ras0 = vox2ras0 @ m_tz @ m_center_inv @ m_rotate @ m_center @ m_tx @ m_fliplr @ m_flipud

        return new_vox2ras0

    def change_orientation(self, slice_i, block_id, vox2ras0, v2r_ref=np.eye(4), order=1):

        init_dtype = slice_i.dtype
        slice_i = np.double(slice_i)

        # if block_id == 'C4.1':
        #     slice_i = np.fliplr(slice_i)
        #     return slice_i
        # elif block_id in ['C1.1', 'C2.1', 'C3.1', 'C5.1']:
        #     return slice_i

        if 'C' in block_id:
            dim = 1
        else:
            dim = 0
        dim_histo = 1

        proj_z = np.dot(vox2ras0[:3, dim_histo], v2r_ref[:3, dim]) / (np.linalg.norm(vox2ras0[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim]))
        angle = np.arccos(proj_z)
        if proj_z < 0 and block_id != 'P9.1': angle = -angle
        if block_id in ['B1.1', 'B2.1', 'B3.1', 'B4.1']: angle = 0
        if block_id in ['C2.1', 'C3.1']: angle = -angle

        data = []
        if len(slice_i.shape) == 3:
            for it_c in range(3):
                d = imrotate(slice_i[..., it_c], angle / np.pi * 180, resize=True, order=order)
                if block_id not in ['B1.1', 'B2.1', 'B3.1', 'B4.1', 'C1.1', 'C2.1', 'C3.1', 'C4.1']: d = np.fliplr(d)
                if block_id in ['P9.1', 'C4.1', 'B1.1', 'B2.1', 'B3.1', 'B4.1']: d = np.flipud(d)
                d = d.astype(init_dtype)
                data.append(d)

            slice_i = np.stack([d for d in data], axis=-1)

        else:
            d = imrotate(slice_i, angle / np.pi * 180, resize=True, order=order)
            if block_id not in ['B1.1', 'B2.1', 'B3.1', 'B4.1', 'C1.1', 'C2.1', 'C3.1', 'C4.1']: d = np.fliplr(d)
            if block_id in ['P9.1', 'C4.1', 'B1.1', 'B2.1', 'B3.1', 'B4.1']: d = np.flipud(d)
            slice_i = d.astype(init_dtype)

        return slice_i

class P85(Case):
    def get_vox2ras0(self, vox2ras0, block_id, blockshape, v2r_ref=np.eye(4), nslices=1):
        # if ('C' in block_id and block_id != 'C5.1'):
        #     m_fliplr = np.eye(4)
        #     m_fliplr[0, 0] = -1
        #     m_fliplr[1, 1] = -1
        #     m_fliplr[0, 3] = blockshape[0] - 1
        #     m_fliplr[1, 3] = blockshape[1] - 1
        #     new_vox2ras0 = vox2ras0 @ m_fliplr
        #
        #     return new_vox2ras0

        if 'C' in block_id:
            dim = 1
        else:
            dim = 0
        dim_histo = 1
        proj_z = np.dot(vox2ras0[:3, dim_histo], v2r_ref[:3, dim]) / (
                    np.linalg.norm(vox2ras0[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim]))

        # if 'B' in block_id: proj_z = -np.abs(proj_z)
        angle = np.arccos(proj_z)
        # if block_id == 'C5.1':  angle = angle + 45 * np.pi / 180
        if block_id in ['B1.1','B2.1','B3.1']: angle -= 7.5/180*np.pi
        # elif block_id == 'B4.1': angle += np.pi #-angle
        elif 'B' not in block_id and block_id not in ['P8.1', 'A2.2', 'A4.1', 'P3.2', 'P5.1', 'P5.2', 'P6.1', 'P6.2', 'C1.1']: angle = -angle
        # elif 'C' not in block_id and 'B' not in block_id: angle = -angle
        # if block_id in ['P1.1', 'P2.1', 'P1.4', 'A3.1', 'A1.4', 'A1.1', 'A3.2', 'A1.3']: angle = -angle

        center = np.array((blockshape[1], blockshape[0])) / 2. - 0.5
        tform1 = SimilarityTransform(translation=center)
        tform2 = SimilarityTransform(rotation=angle)
        tform3 = SimilarityTransform(translation=-center)
        tform = tform3 + tform2 + tform1
        rows, cols = blockshape[0], blockshape[1]
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1

        # fit output image in new shape
        output_shape = np.around((out_rows, out_cols))

        m_center = np.eye(4)
        m_center[0, -1] = -blockshape[0]/2. - 0.5
        m_center[1, -1] = -blockshape[1]/2. - 0.5

        m_center_inv = np.eye(4)
        m_center_inv[0, -1] = blockshape[0]/2. - 0.5
        m_center_inv[1, -1] = blockshape[1]/2. - 0.5

        m_tx = np.eye(4)
        m_tx[0, -1] = minr
        m_tx[1, -1] = minc

        m_rotate = np.eye(4)
        m_rotate[0,0] = np.cos(angle)
        m_rotate[0,1] = np.sin(angle)
        m_rotate[1,0] = -np.sin(angle)
        m_rotate[1,1] = np.cos(angle)

        m_flipud = np.eye(4)
        m_fliplr = np.eye(4)
        if block_id not in ['P8.1', 'C1.1', 'C2.1', 'C3.1', 'C4.1', 'C5.1']:
            m_fliplr[1, 1] = -1
            m_fliplr[1, -1] = output_shape[1] - 1

        if block_id in ['C1.1', 'C2.1', 'C3.1', 'C4.1']:
            m_flipud[0, 0] = -1
            m_flipud[0, -1] = output_shape[0] - 1

        m_tz = np.eye(4)
        if block_id in ['P8.1', 'C5.1']:
            m_tz[2, 2] = -1
            m_tz[2, -1] = nslices - 1

        new_vox2ras0 = vox2ras0 @ m_tz  @ m_center_inv @ m_rotate @ m_center @ m_tx @ m_fliplr @ m_flipud

        return new_vox2ras0

    def change_orientation(self, slice_i, block_id, vox2ras0, v2r_ref=np.eye(4), order=1):

        init_dtype = slice_i.dtype
        slice_i = np.double(slice_i)

        # if ('C' in block_id and block_id != 'C5.1'):
        #     slice_i = np.fliplr(slice_i)
        #     return slice_i

        if 'C' in block_id:
            dim = 1
        else:
            dim = 0
        dim_histo = 1
        proj_z = np.dot(vox2ras0[:3, dim_histo], v2r_ref[:3, dim]) / (np.linalg.norm(vox2ras0[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim]))

        # if 'B' in block_id: proj_z = -np.abs(proj_z)
        angle = np.arccos(proj_z)
        # if block_id == 'C5.1': angle = angle + 45 * np.pi / 180
        if block_id in ['B1.1','B2.1','B3.1']: angle -= 7.5/180*np.pi
        # elif block_id == 'B4.1': angle += np.pi
        elif 'B' not in block_id and block_id not in ['P8.1', 'A2.2', 'A4.1', 'P3.2', 'P5.1', 'P5.2', 'P6.1', 'P6.2', 'C1.1']: angle = -angle
        # if block_id in ['P1.1', 'P1.2', ' P1.3', 'P2.1', 'P1.4', 'A3.1', 'A1.4', 'A1.1', 'A3.2', 'A1.3', 'A2.3']: angle = -angle


        data = []
        if len(slice_i.shape) == 3:
            for it_c in range(3):
                d = imrotate(slice_i[..., it_c], angle / np.pi * 180, resize=True, order=order)
                if block_id not in ['P8.1','C1.1', 'C2.1', 'C3.1', 'C4.1', 'C5.1']: d = np.fliplr(d)
                if block_id in ['C1.1', 'C2.1', 'C3.1', 'C4.1']: d = np.flipud(d)
                d = d.astype(init_dtype)
                data.append(d)

            slice_i = np.stack([d for d in data], axis=-1)

        else:
            d = imrotate(slice_i, angle / np.pi * 180, resize=True, order=order)
            if block_id not in ['P8.1','C1.1', 'C2.1', 'C3.1', 'C4.1', 'C5.1']: d = np.fliplr(d)
            if block_id in ['C1.1', 'C2.1', 'C3.1', 'C4.1']: d = np.flipud(d)
            slice_i = d.astype(init_dtype)

        return slice_i

class EX9(Case):
    def get_vox2ras0(self, vox2ras0, block_id, blockshape, v2r_ref=np.eye(4), nslices=1):
        # if ('C' in block_id and block_id != 'C5.1'):
        #     m_fliplr = np.eye(4)
        #     m_fliplr[0, 0] = -1
        #     m_fliplr[1, 1] = -1
        #     m_fliplr[0, 3] = blockshape[0] - 1
        #     m_fliplr[1, 3] = blockshape[1] - 1
        #     new_vox2ras0 = vox2ras0 @ m_fliplr
        #
        #     return new_vox2ras0
        #
        #
        # # elif 'B' in block_id:
        # #
        # #     m_flipud = np.eye(4)
        # #     m_flipud[0, 0] = -1
        # #     m_flipud[0, -1] = blockshape[0] - 1
        # #
        # #     m_fliplr = np.eye(4)
        # #     m_fliplr[0, 0] = -1
        # #     m_fliplr[1, 1] = -1
        # #     m_fliplr[0, 3] = blockshape[0] - 1
        # #     m_fliplr[1, 3] = blockshape[1] - 1
        # #     new_vox2ras0 = vox2ras0 @ m_fliplr @ m_flipud
        # #
        # #     dim = 0
        # #     dim_histo = 1
        # #     proj_z = np.dot(vox2ras0[:3, dim_histo], v2r_ref[:3, dim]) / (
        # #             np.linalg.norm(vox2ras0[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim]))
        # #     angle = np.arccos(proj_z)
        # #
        # #     return new_vox2ras0
        #
        # elif block_id == 'C5.1':
        #     m_tz = np.eye(4)
        #     m_tz[2, 2] = -1
        #     m_tz[2, -1] = nslices - 1
        #     return vox2ras0 @ m_tz

        if 'C' in block_id:
            dim = 1
        else:
            dim = 0
        dim_histo = 1
        proj_z = np.dot(vox2ras0[:3, dim_histo], v2r_ref[:3, dim]) / (
                np.linalg.norm(vox2ras0[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim]))
        angle = np.arccos(proj_z)
        if block_id in ['B3.1']: angle = 0
        elif block_id in ['P7.1', 'P7.2', 'C1.1', 'C2.1',  'C4.1', 'C5.1', 'P5.2']: pass
        else: angle = -angle

        center = np.array((blockshape[1], blockshape[0])) / 2. - 0.5
        tform1 = SimilarityTransform(translation=center)
        tform2 = SimilarityTransform(rotation=angle)
        tform3 = SimilarityTransform(translation=-center)
        tform = tform3 + tform2 + tform1
        rows, cols = blockshape[0], blockshape[1]
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1

        # fit output image in new shape
        output_shape = np.around((out_rows, out_cols))

        m_center = np.eye(4)
        m_center[0, -1] = -blockshape[0]/2. - 0.5
        m_center[1, -1] = -blockshape[1]/2. - 0.5

        m_center_inv = np.eye(4)
        m_center_inv[0, -1] = blockshape[0]/2. - 0.5
        m_center_inv[1, -1] = blockshape[1]/2. - 0.5

        m_tx = np.eye(4)
        m_tx[0, -1] = minr
        m_tx[1, -1] = minc

        m_rotate = np.eye(4)
        m_rotate[0,0] = np.cos(angle)
        m_rotate[0,1] = np.sin(angle)
        m_rotate[1,0] = -np.sin(angle)
        m_rotate[1,1] = np.cos(angle)


        m_flipud = np.eye(4)
        m_fliplr = np.eye(4)
        if block_id not in ['C1.1', 'C2.1', 'C3.1', 'C4.1', 'C5.1']:
            m_fliplr[1, 1] = -1
            m_fliplr[1, -1] = output_shape[1] - 1
        if block_id in ['A5.2',  'P9.1', 'C1.1', 'C2.1', 'C3.1', 'C4.1']:
            m_flipud[0, 0] = -1
            m_flipud[0, -1] = output_shape[0] - 1

        m_tz = np.eye(4)
        if block_id in ['P9.1', 'C5.1']:
            m_tz[2, 2] = -1
            m_tz[2, -1] = nslices - 1

        new_vox2ras0 = vox2ras0 @ m_tz  @ m_center_inv @ m_rotate @ m_center @ m_tx @ m_fliplr @ m_flipud

        return new_vox2ras0

    def change_orientation(self, slice_i, block_id, vox2ras0, v2r_ref=np.eye(4), order=1):

        init_dtype = slice_i.dtype
        slice_i = np.double(slice_i)

        # if ('C' in block_id and block_id != 'C5.1'):
        #     slice_i = np.fliplr(slice_i)
        #     return slice_i
        #
        #
        # # elif 'B' in block_id:
        # #     slice_i = np.fliplr(slice_i)
        # #     slice_i = np.flipud(slice_i)
        # #     return slice_i
        #
        # elif block_id == 'C5.1': return slice_i


        if 'C' in block_id:
            dim = 1
        else:
            dim = 0
        dim_histo = 1
        proj_z = np.dot(vox2ras0[:3, dim_histo], v2r_ref[:3, dim]) / (
                    np.linalg.norm(vox2ras0[:3, dim_histo]) * np.linalg.norm(v2r_ref[:3, dim]))
        angle = np.arccos(proj_z)
        if block_id in ['B3.1']: angle = 0
        elif block_id in ['P7.1', 'P7.2', 'C1.1', 'C2.1', 'C4.1', 'C5.1', 'P5.2']: pass
        else:  angle = -angle
        # if proj_z < 0 and block_id not in ['A3.3', 'P5.2']: angle = -angle
        # elif block_id in ['P7.1', 'P7.2', 'P6.1', 'P6.2','B3.1']: angle = 0
        # else:  angle = -angle

        # if block_id in ['A2.3', 'B1.1', 'B2.1', 'B4.1', 'P5.3', 'P2.1', 'P1.1', 'P3.1', 'A1.3', 'P4.1', 'A1.1', 'A2.1', 'A3.1', 'A2.2', 'P3.4']:

        data = []
        if len(slice_i.shape) == 3:
            for it_c in range(3):
                d = imrotate(slice_i[..., it_c], angle / np.pi * 180, resize=True, order=order)
                if block_id not in ['C1.1', 'C2.1', 'C3.1', 'C4.1', 'C5.1']: d = np.fliplr(d)
                if block_id in ['P9.1', 'A5.2', 'C1.1', 'C2.1', 'C3.1', 'C4.1']: d = np.flipud(d)
                d = d.astype(init_dtype)
                data.append(d)

            slice_i = np.stack([d for d in data], axis=-1)

        else:
            d = imrotate(slice_i, angle / np.pi * 180, resize=True, order=order)
            if block_id not in ['C1.1', 'C2.1', 'C3.1', 'C4.1', 'C5.1']: d = np.fliplr(d)
            if block_id in ['P9.1', 'A5.2', 'C1.1', 'C2.1', 'C3.1', 'C4.1']: d = np.flipud(d)
            slice_i = d.astype(init_dtype)

        return slice_i



DICT_CLASSES = {
    'P57-16': P57,
    'P41-16': P41,
    'P58-16': P58,
    'P85-18': P85,
    'EX9-19': EX9,
}
