import sys, os
from pathlib import Path
root_dir = Path(__file__).parent.absolute().parent.absolute()
sys.path.append(str(root_dir))

from utils.utils import *
from bm4d import bm4d, BM4DProfile, BM4DStages, BM4DProfile2D, BM4DProfileComplex, BM4DProfileBM3DComplex
import random

def cast(value, num_type):
    return num_type(value) if value is not None else None


class PTYCHO:
    def __init__(self, y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None, num_iter=100, recon_win=None, joint_recon=False):
        # initialization 
        self.dtype_cmplx = np.complex64
        self.dtype_real = np.float64

        self.y_meas = cast(y_meas, self.dtype_real)
        self.patch_bounds = patch_bounds
        self.recon_win = np.ones_like(init_obj) if recon_win is None else recon_win
        self.joint_recon = joint_recon 

        self.cur_image = cast(init_obj, self.dtype_cmplx)  # obj_est
        self.img_shape = self.cur_image.shape
        self.cur_patches = self.img2patch(self.cur_image)  # obj_mat
        self.patch_shape = self.y_meas.shape

        self.ref_obj = cast(ref_obj, self.dtype_cmplx)       # ref_obj
        self.ref_probe = cast(ref_probe, self.dtype_cmplx)   # ref_probe

        self.obj_nrmse = []
        self.probe_nrmse = []

        self.cur_probe = cast(init_probe, self.dtype_cmplx) if init_probe is not None else self.ref_probe  # probe_est
        
        # recon
        self.num_iter = num_iter
        non_zero_idx = np.nonzero(self.recon_win)
        self.blk_idx = [np.amin(non_zero_idx[0]), np.amax(non_zero_idx[0]) + 1,
                        np.amin(non_zero_idx[1]), np.amax(non_zero_idx[1]) + 1]

    def plot_img(self, img):
        plt.subplot(121)
        plt.imshow(np.real(img), cmap='gray')
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(np.imag(img), cmap='gray')
        plt.colorbar()
        plt.show()
        plt.clf()
         
    def epie_recon(self, obj_step_sz=0.1, probe_step_sz=0.1, save_dir=None):
        self.check_fpath(save_dir)
        start_time = time.time()
        seq = np.arange(0, len(self.y_meas), 1).tolist()
        est_image = np.copy(self.cur_image)
        est_probe = np.copy(self.cur_probe)
        crds = self.patch_bounds
        approach = 'ePIE'
        print('{} reconstruction starts ...'.format(approach))
        for i in range(self.num_iter):
            random.shuffle(seq)
            for j in seq:
                crd0, crd1, crd2, crd3 = crds[j, 0], crds[j, 1], crds[j, 2], crds[j, 3]
                projected_img = np.copy(est_image[crd0:crd1, crd2:crd3])
                frm = projected_img * est_probe
                freq = compute_ft(frm)
                freq_update = (self.y_meas[j] * freq / np.abs(freq)).astype(self.dtype_cmplx)
                delta_frm = compute_ift(freq_update) - frm
                est_image[crd0:crd1, crd2:crd3] += obj_step_sz * np.conj(est_probe) * delta_frm / (np.amax(np.abs(est_probe)) ** 2)
                if self.joint_recon:
                    est_probe += probe_step_sz * np.conj(projected_img) * delta_frm / (np.amax(np.abs(projected_img)) ** 2)

            # phase normalization and scale image to minimize the intensity difference
            if self.ref_obj is not None:
                est_image_adj = phase_norm(est_image * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
                cur_obj_nrmse = compute_nrmse(est_image_adj * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
                self.obj_nrmse.append(cur_obj_nrmse)
                print(i, cur_obj_nrmse)
            else:
                est_image_adj = est_image

            # phase normalization and scale image to minimize the intensity difference
            if self.ref_probe is not None:
                est_probe_adj = phase_norm(est_probe, self.ref_probe)
                cur_probe_nrmse = compute_nrmse(est_probe_adj, self.ref_probe)
                self.probe_nrmse.append(cur_probe_nrmse)
            else:
                est_probe_adj = est_probe

            if (i+1) % 10 == 0:
                print('Finished {:d} of {:d} iterations.'.format(i+1, self.num_iter))

        # calculate time consumption
        elapsed_time = time.time() - start_time
        print('Time consumption of {}:'.format(approach), elapsed_time)

        # save recon results
        save_tiff(est_image, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
        save_array(self.obj_nrmse, save_dir + 'obj_nrmse_' + str(self.obj_nrmse[-1]))
        if self.joint_recon:
            save_tiff(est_probe, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
            save_array(probe_nrmse, save_dir + 'probe_nrmse_' + str(self.probe_nrmse[-1]))

        # return recon results
        keys = ['obj_revy', 'obj_err', 'probe_revy', 'probe_err']
        vals = [est_image_adj, self.obj_nrmse, est_probe_adj, self.probe_nrmse]
        output = dict(zip(keys, vals))

        return output

    def img2patch(self, img, output_patches=None):
        """
        Function to extract image patches from full-sized image.
        Args:
            img: the full-sized image.
            output_patches: an array of size self.y_meas.shape to hold the output patches
        Returns:
            projected image patches.
        """
        if output_patches is None:
            output_patches = np.zeros(self.y_meas.shape, dtype=self.dtype_cmplx)
        coords = self.patch_bounds
        # TODO: convert to a more efficient implementation - perhaps jax or cython
        for j in range(len(output_patches)):
            output_patches[j, :, :] = img[coords[j, 0]:coords[j, 1], coords[j, 2]:coords[j, 3]]

        return output_patches

    def patch2img(self, patches, output_image=None):
        if output_image is None:
            output_image = np.zeros(self.img_shape, dtype=self.dtype_cmplx)
        coords = self.patch_bounds
        for j in range(len(patches)):
            output_image[coords[j, 0]:coords[j, 1], coords[j, 2]:coords[j, 3]] += patches[j]

        return output_image



    def xbar(self, patches, normalize=True, patch_wgt=None, image_wgt=None):
        if patch_wgt is None:
            patch_wgt = np.ones(self.patch_shape, dtype=self.dtype_real)
        if image_wgt is None:
            image_wgt = self.patch2img(patches)
        
        # initialization
        output_image = np.zeros(self.img_shape, dtype=self.dtype_cmplx)
        weighted_patches = (patch_wgt * patches).astype(self.dtype_cmplx)
        image_wgt[image_wgt < 1e-3] = 1e-3

        # Back projection from patches to image
        crds = self.patch_bounds
        for j in range(len(patches)):
            output_image[crds[j, 0]:crds[j, 1], crds[j, 2]:crds[j, 3]] += weighted_patches[j]
        
        # Normalization 
        if normalize:
            output_image = output_image / image_wgt
        
        return output_image.astype(self.dtype_cmplx)

    def dbar(self, probe_mat, patches, image_exp):
        img_wgt = np.abs(patches, dtype=self.dtype_real) ** image_exp
        output = np.sum(probe_mat * img_wgt, 0) / np.sum(img_wgt, 0)

        return output.astype(self.dtype_cmplx)

    def operator_F(self, cur_est, joint_est, data_fit_param):
        # FT{D*v_j}
        freq = compute_ft(cur_est * joint_est)
        
        # y \times FT{D*v_j} / |FT{D*v_j}|
        freq_update = (self.y_meas * freq / np.abs(freq)).astype(self.dtype_cmplx)
        
        # IFT{y \times FT{D*v_j} / |FT{D*v_j}| }
        freq_ift = compute_ift(freq_update)
        
        # take weighted average of current estimate and projected data-fitting point
        output = (1 - data_fit_param) * cur_est + \
                 data_fit_param * divide_cmplx_numbers(freq_ift, joint_est).astype(self.dtype_cmplx)

        return output

    def operator_G(self, patches, probe_est, use_reg=False, bm3d_psd=0.1, patch_wgt=None, image_wgt=None):
        xbar_image = self.xbar(patches, patch_wgt=patch_wgt, image_wgt=image_wgt)

        if use_reg:
            # restrict to valid region for denoiser
            block_idx = self.blk_idx
            temp_img = xbar_image[block_idx[0]: block_idx[1], block_idx[2]: block_idx[3]]

            denoised_temp_img = bm4d(temp_img, bm3d_psd, profile=BM4DProfileBM3DComplex())[:, :, 0]
            xbar_image[block_idx[0]: block_idx[1], block_idx[2]: block_idx[3]] = denoised_temp_img

        # extract patches out of image
        output_patches = self.img2patch(xbar_image)

        return xbar_image, output_patches

    def check_fpath(self, fpath):
        if fpath is not None:
            os.makedirs(fpath, exist_ok=True)


    def pmace_recon(self, data_fit_param=0.5, rho=0.5, probe_exp=1.5, image_exp=0.5, use_reg=False, sigma=0.1, save_dir=None):
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        # initializatioin
        updated_patches = np.copy(self.cur_patches)
        new_probe = [self.cur_probe] * len(self.y_meas)
        patch_wgt = np.abs(new_probe, dtype=self.dtype_real) ** probe_exp
        image_wgt = self.patch2img(patch_wgt)
        approach = 'reg-PMACE' if use_reg else 'PMACE'
        print('{} reconstruction starts ...'.format(approach))

        start_time = time.time()
        
        for i in range(self.num_iter):
            # w <- F(v)
            cur_patches = self.operator_F(updated_patches, new_probe, data_fit_param)

            # z <- G(2w - v)
            new_image, new_patches = self.operator_G(2 * cur_patches - updated_patches, new_probe, use_reg=use_reg, bm3d_psd=sigma, patch_wgt=patch_wgt, image_wgt=image_wgt)

            # v <- v + 2 \rho (z - w)
            updated_patches += 2 * rho * (new_patches - cur_patches)

            # obtain current estimate of complex image
            est_image = new_image if use_reg else self.xbar(updated_patches, patch_wgt=patch_wgt, image_wgt=image_wgt)
            
            # phase normalization and scale image to minimize the intensity difference
            if self.ref_obj is not None:
                est_image_adj = phase_norm(est_image * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
                cur_obj_nrmse = compute_nrmse(est_image_adj * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
                self.obj_nrmse.append(cur_obj_nrmse)
                print(i, cur_obj_nrmse)
            else:
                est_image_adj = est_image

            # joint reconstruction
            if self.joint_recon:
                # w <- F(v)
                cur_probe = self.operator_F(updated_probe, new_patches, data_fit_param)
                # z <- G(2w - v)
                new_probe = self.dbar((2 * cur_probe - new_probe), new_patches, image_exp)
                # v <- v + 2 \rho (z - w)
                updated_probe += 2 * rho * (new_probe - cur_probe)
                # obtain current estimate of complex probe
                est_probe = self.dbar(updated_probe, new_patches, image_exp)
                # update patch weight and image weight
                patch_wgt = np.abs(new_probe, dtype=self.dtype_real) ** probe_exp
                image_wgt = self.patch2img(patch_wgt)

                # phase normalization and scale image to minimize the intensity difference
                if self.ref_probe is not None:
                    est_probe_adj = phase_norm(est_probe, self.ref_probe)
                    cur_probe_nrmse = compute_nrmse(est_probe_adj, self.ref_probe)
                    self.probe_nrmse.append(cur_probe_nrmse)
                else:
                    est_probe_adj = est_probe

            if (i+1) % 10 == 0:
                print('Finished {:d} of {:d} iterations.'.format(i+1, self.num_iter))

        print(est_image_adj)

        # calculate time consumption
        elapsed_time = time.time() - start_time
        print('Time consumption of {}:'.format(approach), elapsed_time)

        # save recon results
        save_tiff(est_image, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
        save_array(self.obj_nrmse, save_dir + 'obj_nrmse_' + str(self.obj_nrmse[-1]))
        if self.joint_recon:
            save_tiff(est_probe, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
            save_array(probe_nrmse, save_dir + 'probe_nrmse_' + str(self.probe_nrmse[-1]))

        # return recon results
        keys = ['obj_revy', 'obj_err', 'probe_revy', 'probe_err']
        vals = [est_image_adj, self.obj_nrmse, est_probe_adj, self.probe_nrmse]
        output = dict(zip(keys, vals))
    
        return output



