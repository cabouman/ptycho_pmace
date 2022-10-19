import sys, os
from pathlib import Path
root_dir = Path(__file__).parent.absolute().parent.absolute()
sys.path.append(str(root_dir))
from utils.utils import *
from bm4d import bm4d, BM4DProfile, BM4DStages, BM4DProfile2D, BM4DProfileComplex, BM4DProfileBM3DComplex
import random


class PMACE:
    def __init__(self, y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None, 
                 recon_win=None, save_dir=None, probe_exp=1.5, image_exp=0.5):
        """
        Class to perform ptychographic recosntructions.
        Args:
            y_meas: pre-process diffraction patter (intensity data).
            patch_bounds: scan coordinates of projections.
            init_obj: formulated initial guess of complex object.
            init_probe: formulated initial guess of complex probe.
            ref_obj: complex reference image.
            ref_probe: complex reference image.
            recon_win: pre-defined/assigned window for comparing reconstruction results.
            save_dir: directory to save reconstruction results.
            probe_exp: exponent of probe weighting in consensus calculation of image estimate.
            image_exp: exponent of image weighting in consensus calculation of probe estimate.
        """
        # initialization 
        self.dtype_cmplx = np.complex64
        self.dtype_real = np.float64

        self.y_meas = self.cast(y_meas, self.dtype_real)
        self.patch_bounds = patch_bounds
        self.recon_win = np.ones_like(init_obj) if recon_win is None else recon_win
        self.save_dir = self.check_fpath(save_dir)
        
        self.ref_obj = self.cast(ref_obj, self.dtype_cmplx)       # ref_obj
        self.ref_probe = self.cast(ref_probe, self.dtype_cmplx)   # ref_probe
   
        self.cur_image = self.cast(init_obj, self.dtype_cmplx)    # obj_est
        self.img_shape = self.cur_image.shape
        self.cur_patches = self.img2patch(self.cur_image)         # obj_mat
        self.patch_shape = self.y_meas.shape
        self.cur_probe = self.cast(init_probe, self.dtype_cmplx) if init_probe is not None else self.ref_probe  # probe_est
        self.cur_probe_mat = [self.cur_probe] * len(self.y_meas)
        
        self.probe_exp = self.cast(probe_exp, self.dtype_real)
        self.image_exp = self.cast(image_exp, self.dtype_real)
        
        # spatially-varying weights in consensus calculations
        self.xbar_patch_wgt = np.abs(self.cur_probe_mat, dtype=self.dtype_real) ** self.probe_exp
        self.xbar_image_wgt = self.patch2img(self.xbar_patch_wgt)
        pmace_tol = np.amax([1e-3, np.amax(np.abs(self.xbar_image_wgt)) * 1e-6])
        self.xbar_image_wgt[np.abs(self.xbar_image_wgt) < pmace_tol] = pmace_tol
        
        self.dbar_probe_arr_wgt = np.abs(self.cur_patches, dtype=self.dtype_real) ** self.image_exp
        self.dbar_probe_wgt = np.sum(self.dbar_probe_arr_wgt, 0)

        non_zero_idx = np.nonzero(np.abs(recon_win))
        crd0, crd1 = np.max([0, np.amin(non_zero_idx[0])]), np.min([np.amax(non_zero_idx[0])+1, self.img_shape[0]])
        crd2, crd3 = np.max([0, np.amin(non_zero_idx[1])]), np.min([np.amax(non_zero_idx[1])+1, self.img_shape[1]])
        self.blk_idx = [crd0, crd1, crd2, crd3]
        
        self.obj_nrmse = []
        self.probe_nrmse = []
        self.meas_nrmse = []
        # self.images = []
        # self.images.append(self.cur_image[self.blk_idx[0]: self.blk_idx[1], self.blk_idx[2]: self.blk_idx[3]])

    def cast(self, value, num_type):
        """
        Chnage current data type to target data type.
        """
        return num_type(value) if value is not None else None

    def reset(self):
        """
        Reset attributes.
        """
        self.obj_nrmse = []
        self.probe_nrmse = []
        self.meas_nrmse = []

    def check_fpath(self, fpath):
        """
        Create directory if it does not exist.
        Args:
            fpath: fire directory.
        """
        if fpath is not None:
            os.makedirs(fpath, exist_ok=True)

        return fpath

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
        """
        Function to project image patches back to full-sized image.
        Args:
            patches: image patches.
        Returns:
            full-sized image.
        """
        if output_image is None:
            output_image = np.zeros(self.img_shape, dtype=self.dtype_cmplx)
        
        coords = self.patch_bounds
        for j in range(len(patches)):
            output_image[coords[j, 0]:coords[j, 1], coords[j, 2]:coords[j, 3]] += patches[j]

        return output_image.astype(self.dtype_cmplx)

    def xbar(self, patches, normalize=True):
        """
        Function to obtain consensus result from image patches.
        Args:
            patches: image patches.
            normalize: option to do normalization.
        Returns:
            full-sized image.
        """
        # back projection from weighted patches to image
        output_image = self.patch2img((self.xbar_patch_wgt * patches).astype(self.dtype_cmplx))
        
        # normalization 
        if normalize:
            output_image = output_image / self.xbar_image_wgt
        
        return output_image.astype(self.dtype_cmplx)

    def dbar(self, probe_mat):
        """
        Function to obtain consensus result from probe mat.
        Args:
            probe_mat: spacing-varying probes.
        Returns:
            complex probe.
        """
        output = np.sum(probe_mat * self.dbar_probe_arr_wgt, 0) / self.dbar_probe_wgt
    
        return output.astype(self.dtype_cmplx)

    def operator_F(self, cur_est, joint_est, data_fit_param):
        """
        Fit data using stack of weighted proximal map functions.
        Args:
            cur_est: current estimate of image patches / complex probe.
            joint_est: current estimate of complex probe / image patches.
            data_fit_param: averaging weight in PMACE updates. Param near 1 gives closer fit to data.
        Returns:
            new estimate of projected images / complex probe.
        """
        # FT{D*v_j}
        f = compute_ft(cur_est * joint_est)
        
        # IFT{y \times FT{D*v_j} / |FT{D*v_j}| }
        inv_f = compute_ift( self.y_meas * np.exp(1j * np.angle(f)) )
        
        # take weighted average of current estimate and projected data-fitting point
        output = (1 - data_fit_param) * cur_est + data_fit_param * divide_cmplx_numbers(inv_f, joint_est)

        return output.astype(self.dtype_cmplx)

    def operator_G(self, patches, use_reg=False, bm3d_psd=0.01):
        """
        Consensus operator which takes spatially weighted average of input image patches and 
        reallocates the results.
        Args:
            patches: current estimate of image patches.
            use_reg: add serial regularization.
            bm3d_psd: psd of complex bm3d software.
        Returns:
            new estimate of projected images.
        """
        # take weighted average of input image patches
        xbar_image = self.xbar(patches, normalize=True)

        # apply complex bm3d
        if use_reg:
            # restrict to valid region for denoiser
            crds = self.blk_idx
            tmp_img = xbar_image[crds[0]: crds[1], crds[2]: crds[3]]

            denoised_tmp_img = bm4d(tmp_img, bm3d_psd, profile=BM4DProfileBM3DComplex())[:, :, 0]
            xbar_image[crds[0]: crds[1], crds[2]: crds[3]] = denoised_tmp_img

        # extract patches out of image
        output_patches = self.img2patch(xbar_image)

        return xbar_image, output_patches

    def recon(self, num_iter=100, joint_recon=False, obj_data_fit_param=0.5, probe_data_fit_param=0.5,
              rho=0.5, use_reg=False, sigma=0.01):
        """
        Args:
            num_iter: number of iterations.
            joint_recon: option to recover complex probe for blind ptychography.
            obj_data_fit_param: averaging weight in object updates. Param near 1 gives closer fit to data.
            probe_data_fit_param: averaging weight in probe updates.
            rho: Mann averaging parameter.
            use_reg: option of applying denoiser to PMACE.
            sigma: denoising parameter.
        Returns:
            Reconstructed complex images, and errors between reconstructions and reference images. 
        """
        # initialization
        approach = 'reg-PMACE' if use_reg else 'PMACE'
        updated_patches = self.cur_patches
        new_probe = self.cur_probe

        # reconstruction
        start_time = time.time()
        print('{} reconstruction starts ...'.format(approach))
        for i in range(num_iter):
            # w <- F(v)
            cur_patches = self.operator_F(updated_patches, new_probe, obj_data_fit_param)
            # z <- G(2w - v)
            new_image, new_patches = self.operator_G(2 * cur_patches - updated_patches, use_reg=use_reg, bm3d_psd=sigma)
            # v <- v + 2 \rho (z - w)
            updated_patches = updated_patches + 2 * rho * (new_patches - cur_patches)
            # obtain current estimate of complex image
            est_image = new_image if use_reg else self.xbar(updated_patches)

            # joint reconstruction
            if joint_recon:
                # calculate probe weights
                self.dbar_probe_arr_wgt = np.abs(new_patches, dtype=self.dtype_real) ** self.image_exp
                self.dbar_probe_wgt = np.sum(self.dbar_probe_arr_wgt, 0)

                # w <- F(v)
                cur_probe = self.operator_F(updated_probe, new_patches, probe_data_fit_param)
                # z <- G(2w - v)
                #new_probe = np.sum((2*cur_probe - updated_probe) * self.probe_arr_wgt, 0) / self.probe_wgt
                new_probe = self.dbar(2 * cur_probe - updated_probe)
                # v <- v + 2 \rho (z - w)
                updated_probe = updated_probe + 2 * rho * (new_probe - cur_probe)
                # obtain current estimate of complex probe
                est_probe = self.dbar(self.pmace_probe, new_patches, self.image_exp)

                # update patch weight and image weight
                self.xbar_patch_wgt = np.abs([new_probe] * len(self.y_meas), dtype=self.dtype_real) ** self.probe_exp
                self.xbar_image_wgt = self.patch2img(self.xbar_patch_wgt)
                pmace_tol = np.amax([1e-3, np.amax(np.abs(self.xbar_image_wgt)) * 1e-6])
                self.xbar_image_wgt[np.abs(self.xbar_image_wgt) < pmace_tol] = pmace_tol

            # phase normalization and scale image to minimize the intensity difference
            if self.ref_obj is not None:
                est_image_adj = phase_norm(est_image * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
                cur_obj_nrmse = compute_nrmse(est_image_adj * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
                self.obj_nrmse.append(cur_obj_nrmse)
            else:
                est_image_adj = est_image
            
            # phase normalization and scale image to minimize the intensity difference
            if joint_recon:
                if self.ref_probe is not None:
                    est_probe_adj = phase_norm(est_probe, self.ref_probe)
                    cur_probe_nrmse = compute_nrmse(est_probe_adj, self.ref_probe)
                    self.probe_nrmse.append(cur_probe_nrmse)
                else:
                    est_probe_adj = est_probe
            else:
                est_probe = self.cur_probe
                est_probe_adj = est_probe

            # calculate error in measurement domain
            est_patch = self.img2patch(est_image)
            est_meas = np.abs(compute_ft(est_probe * est_patch))
            self.meas_nrmse.append(compute_nrmse(est_meas, self.y_meas))

            # append reconstructed images
            #self.images.append(est_image_adj[self.blk_idx[0]: self.blk_idx[1], self.blk_idx[2]: self.blk_idx[3]])
            if (i+1) % 10 == 0:
                print('Finished {:d} of {:d} iterations.'.format(i+1, num_iter))

        # calculate time consumption
        print('Time consumption:', time.time() - start_time)

        # # create gif image
        # gen_gif(self.images, fps=2, save_dir=self.save_dir)

        # save recon results
        if self.save_dir is not None:
            save_tiff(est_image, self.save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
            if self.obj_nrmse:
                save_array(self.obj_nrmse, self.save_dir + 'obj_nrmse_' + str(self.obj_nrmse[-1]))
            if self.meas_nrmse:
                save_array(self.meas_nrmse, self.save_dir + 'meas_nrmse_' + str(self.meas_nrmse[-1]))
            if joint_recon:
                save_tiff(est_probe, self.save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
                if self.probe_nrmse:
                    save_array(self.probe_nrmse, self.save_dir + 'probe_nrmse_' + str(self.probe_nrmse[-1]))

        # return recon results
        keys = ['object', 'err_obj', 'probe', 'err_probe', 'err_meas']
        vals = [est_image_adj, self.obj_nrmse, est_probe_adj, self.probe_nrmse, self.meas_nrmse]
        output = dict(zip(keys, vals))

        return output


