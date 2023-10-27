import numpy as np
import torch

def predict_sbr(data_dict, model, device, da_model=None):
    '''
    This class is used to check the results from SbR training.
    :param data_dict:
    :param model:
    :param device:
    :param da_model:
    :return:
    '''

    output_results = {}
    with torch.no_grad():
        ref_image, flo_image = data_dict['x_ref'].to(device), data_dict['x_flo'].to(device)
        ref_mask, flo_mask = data_dict['x_ref_mask_init'].to(device), data_dict['x_flo_mask_init'].to(device)
        ref_labels, flo_labels = data_dict['x_ref_labels'].to(device), data_dict['x_flo_labels'].to(device)
        nonlinear_field = [nlf.to(device) for nlf in data_dict['nonlinear']]
        affine_field = [aff.to(device) for aff in data_dict['affine']]

        if da_model is not None:
            flo_image = da_model.transform(flo_image, affine_field[0], nonlinear_field[0])
            flo_mask = da_model.transform(flo_mask, affine_field[0], nonlinear_field[0])
            flo_labels = da_model.transform(flo_labels, affine_field[0], nonlinear_field[0])

        flo_image_fake = model['G_M'](flo_image)
        flo_image_fake = flo_image_fake * flo_mask
        r_fake, f, v = model['R_M'](flo_image_fake, ref_image)
        f_rev = model['R_M'].get_flow_field(-v)

        r = model['R_M'].predict(flo_image, f, svf=False)
        r_mask = model['R_M'].predict(flo_mask, f, svf=False, mode='nearest')
        r_labels = model['R_M'].predict(flo_labels, f, svf=False, mode='nearest')

        r_flo = model['R_M'].predict(ref_image, f_rev, svf=False)
        r_flo_mask = model['R_M'].predict(ref_mask, f_rev, svf=False, mode='nearest')

        output_results['data_M'] = ref_image[:,0].cpu().detach().numpy()
        output_results['data_H'] = flo_image[:,0].cpu().detach().numpy()
        output_results['gen_M'] = flo_image_fake[:,0].cpu().detach().numpy()
        output_results['reg_data_H'] = r[:,0].cpu().detach().numpy()
        output_results['reg_gen_M'] = r_fake[:,0].cpu().detach().numpy()
        output_results['reg_data_M'] = r_flo[:,0].cpu().detach().numpy()

        output_results['mask_M'] = ref_mask[:,0].cpu().detach().numpy()
        output_results['mask_H'] = flo_mask[:,0].cpu().detach().numpy()
        output_results['reg_mask_H'] = r_mask[:,0].cpu().detach().numpy()
        output_results['reg_mask_M'] = r_flo_mask[:,0].cpu().detach().numpy()
        output_results['reg_mask_H'] = np.argmax(r_labels.cpu().detach().numpy(), axis=1)

        output_results['flow'] = f.cpu().detach().numpy()

    return output_results


def predict_registration(data_dict, model, device, da_model=None):
    '''
    This class is used to check the results from standard training.
    :param data_dict:
    :param model:
    :param device:
    :param da_model:
    :return:
    '''
    output_results = {}
    with torch.no_grad():
        ref_image, flo_image = data_dict['x_ref'].to(device), data_dict['x_flo'].to(device)
        ref_mask, flo_mask = data_dict['x_ref_mask'].to(device), data_dict['x_flo_mask'].to(device)
        ref_labels, flo_labels = data_dict['x_ref_labels'].to(device), data_dict['x_flo_labels'].to(device)
        nonlinear_field = [nlf.to(device) for nlf in data_dict['nonlinear']]
        affine_field = [aff.to(device) for aff in data_dict['affine']]

        if da_model is not None:
            flo_image = da_model.transform(flo_image, affine_field[0], nonlinear_field[0])
            flo_mask = da_model.transform(flo_mask, affine_field[0], nonlinear_field[0])
            flo_labels = da_model.transform(flo_labels, affine_field[0], nonlinear_field[0])


        r, f, v = model(flo_image, ref_image)
        f_rev = model.get_flow_field(-v)

        r_mask = model.predict(flo_mask, f, svf=False)
        r_labels = model.predict(flo_labels, f, svf=False)

        r_ref = model.predict(ref_image, f_rev, svf=False)
        r_ref_mask = model.predict(ref_mask, f_rev, svf=False)
        output_results['flow'] = f.cpu().detach().numpy()

        output_results['ref_image'] = ref_image[:,0].cpu().detach().numpy()
        output_results['flo_image'] = flo_image[:,0].cpu().detach().numpy()
        output_results['reg_flo_image'] = r[:,0].cpu().detach().numpy()
        output_results['reg_ref_image'] = r_ref[:,0].cpu().detach().numpy()

        output_results['ref_mask'] = ref_mask[:,0].cpu().detach().numpy()
        output_results['flo_mask'] = flo_mask[:,0].cpu().detach().numpy()
        output_results['reg_flo_mask'] = r_mask[:,0].cpu().detach().numpy()
        output_results['reg_ref_mask'] = r_ref_mask[:,0].cpu().detach().numpy()
        output_results['reg_ref_labels'] = np.argmax(r_labels.cpu().detach().numpy(), axis=1)

    return output_results

torch_dtype = torch.float
class LinearTest(object):

    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device


    def predict(self, model_dict, iteration, **kwargs):

        for bid in self.dataset.data_loader.subject_dict.keys():
            model_dict[bid].train()
            model_dict[bid].zero_grad()

        accum_image = torch.zeros((1, 1) + self.dataset.data_loader.vol_shape, dtype=torch_dtype, device=self.device)
        accum_mask = torch.zeros((1, 1) + self.dataset.data_loader.vol_shape, dtype=torch_dtype, device=self.device)

        header_list = []
        for it_bid, bid in enumerate(self.dataset.data_loader.subject_dict.keys()):
            image, mask, header = self.dataset[bid]
            image, mask, header = image.to(self.device), mask.to(self.device), header.to(self.device)

            model = model_dict[bid]
            affine, new_header = model(header, image.shape[2:])

            image = image.type(torch_dtype)
            mask = mask.type(torch_dtype)
            affine = affine.type(torch_dtype)

            reg_image = model_dict['warper'](image, affine, shape=image.shape[2:])
            reg_mask = model_dict['warper'](mask, affine, shape=image.shape[2:])

            accum_mask += reg_mask
            accum_image += reg_image * reg_mask

            header_list.append(new_header)

        print('')

    def update_headers(self, model_dict, **kwargs):

        for bid in self.dataset.data_loader.subject_dict.keys():
            model_dict[bid].eval()

        for it_bid, bid in enumerate(self.dataset.data_loader.subject_dict.keys()):
            image, mask, header = self.dataset[bid]

            image, mask, header = image.to(self.device), mask.to(self.device), header.to(self.device)

            model = model_dict[bid]
            affine, new_header = model(header, image.shape[2:])

            self.dataset.data_loader.subject_dict[bid]._affine = new_header.cpu().detach().numpy()

    def update_headers_images(self, model_dict):

        for it_bid, bid in enumerate(self.dataset.data_loader.subject_dict.keys()):
            image, mask, header = self.dataset[bid]
            image, mask, header = image.to(self.device), mask.to(self.device), header.to(self.device)

            model = model_dict[bid]
            affine, new_header, fields = model(header.to(self.device), image.shape)

            fields = fields.type(torch_dtype)
            nonlin_image = model.warp(image, fields)
            nonlin_mask = model.warp(mask, fields)

            image_block = np.squeeze(nonlin_image.to('cpu').detach().numpy())
            mask_block = np.squeeze(nonlin_mask.to('cpu').detach().numpy())

            self.dataset.data_loader.subject_dict[bid]._affine = new_header.cpu().detach().numpy()

            self.dataset.images_dict[bid] = image_block
            self.dataset.mask_dict[bid] = mask_block

