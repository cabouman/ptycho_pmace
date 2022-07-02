from utils.utils import *
from bm4d import bm4d, BM4DProfile, BM4DStages, BM4DProfile2D, BM4DProfileComplex, BM4DProfileBM3DComplex


def cast(value, num_type):
    return num_type(value) if value is not None else None


class PMACE:
    def __init__(self, y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None,
                 probe_exp=1.5, obj_exp=0.25, recon_win=None, save_dir=None):
        """ 
        Class to perform PMACE reconstruction on ptychographic data.
        Args:
            y_meas: pre-processed diffraction pattern (intensity data).
            patch_bounds: scan coordinates of projections.
            init_obj: formulated initial guess of complex object.
            init_probe: formulated initial guess of complex probe.
            ref_obj: complex reference image.
            ref_probe: complex reference image.
            probe_exp: exponent of probe weighting in consensus calculation of probe estimate.
            obj_exp: exponent of image weighting in consensus calculation of probe estimate.
            recon_win: pre-defined/assigned window for comparing reconstruction results.
            save_dir: directory to save reconstruction results.
        """
        # check directory
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # initialization
        self.dtype = np.complex64
        self.dtype_real = np.float64

        if recon_win is None:
            recon_win = np.ones_like(init_obj)
        self.recon_win = recon_win
        self.y_meas = cast(y_meas, self.dtype_real)

        self.cur_image = cast(init_obj, self.dtype)  # obj_est
        self.img_shape = self.cur_image.shape
        self.patch_bounds = patch_bounds
        self.cur_patches = self.img2patch(self.cur_image)  # obj_mat

        self.init_probe = cast(init_probe, self.dtype)
        self.ref_probe = cast(ref_probe, self.dtype)
        self.probe_exp = cast(probe_exp, self.dtype_real)

        self.ref_obj = cast(ref_obj, self.dtype)
        self.probe_est = self.init_probe if init_probe is not None else self.ref_probe
        self.xbar_patch_wt = np.abs(self.probe_est, dtype=self.dtype_real) ** self.probe_exp
        # TODO:  allow for updates to self.xbar_patch_wt in the case of probe estimation

        self.xbar_image_wt = self.xbar(np.ones(y_meas.shape, self.dtype_real), normalize=False)
        probe_coverage = np.abs(self.xbar_image_wt)
        tol = np.amax([1e-3, np.amax(probe_coverage) * 1e-6])
        self.xbar_image_wt[probe_coverage < tol] = tol  # Avoid division by zero in pixels not covered by probe

        # determine the region for applying denoiser
        non_zero_idx = np.nonzero(self.recon_win)
        self.blk_idx = [np.amin(non_zero_idx[0]), np.amax(non_zero_idx[0]) + 1, np.amin(non_zero_idx[1]),
                        np.amax(non_zero_idx[1]) + 1]  # TODO: Change blk_idx to avoid going outside valid region

        self.obj_nrmse_ls = []
        self.probe_nrmse_ls = []
        self.dp_nrmse_ls = []

    def xbar(self, patches, output_image=None, normalize=True):
        """
        Combine a tensor of patches by first weighting each patch, then adding into a new image at a specified location,
        then normalizing to get a pixel-wise average.
        Args:
            patches: a tensor of shape num_patches x height x width
            output_image: the averaged patches.  This can be created locally or passed in to reuse memory.
            normalize: boolean to determine if the patch sum is normalized to be an average or left unnormalized.

        Returns:
            The patch sum, normalized or not as specified.
        """
        if output_image is None:
            output_image = np.zeros(self.img_shape, dtype=patches.dtype)
        else:  # Clear the output image
            output_image = 0 * output_image
        weighted_patches = (self.xbar_patch_wt * patches).astype(patches.dtype)
        coords = self.patch_bounds
        # backprojection
        for j in range(len(patches)):
            output_image[coords[j, 0]:coords[j, 1], coords[j, 2]:coords[j, 3]] += weighted_patches[j]
        # normalization
        if normalize:
            output_image = (output_image / self.xbar_image_wt).astype(patches.dtype)
        return output_image.astype(patches.dtype)

    def img2patch(self, img, output_patches=None):
        """
        Function to extract image patches from full-sized image.
        Args:
            img: the full-sized image.
            output_patches: an array of size self.y_meas.shape to hold the output patches
        Returns:
            projected image patches.
        """
        num_agts, m, n = self.y_meas.shape
        if output_patches is None:
            output_patches = np.zeros(self.y_meas.shape, dtype=self.dtype)
        coords = self.patch_bounds
        # TODO: convert to a more efficient implementation - perhaps jax or cython
        for j in range(num_agts):
            output_patches[j, :, :] = img[coords[j, 0]:coords[j, 1], coords[j, 2]:coords[j, 3]]

        return output_patches
    
    def operator_F(self, patches, beta_data_fit, probe_est=None):
        """
        Args:
            patches: current estimate of image patches
            beta_data_fit: averaging weight.  beta_patch near 1 gives closer fit to data.
            probe_est: current estimate of probe function

        Returns:
            new estimate of projected images
        """
        if probe_est is None:
            probe_est = self.probe_est
        # FT{D*v_j}
        freq = compute_ft(probe_est * patches)
        # y \times FT{D*v_j} / |FT{D*v_j}|
        freq_update = (self.y_meas * freq / np.abs(freq)).astype(self.dtype)
        # IFT{y \times FT{D*v_j} / |FT{D*v_j}| }
        freq_ift = compute_ift(freq_update)
        # take weighted average of current estimate and projected data-fitting point
        output = (1 - beta_data_fit) * patches + \
                 beta_data_fit * divide_cmplx_numbers(freq_ift, probe_est).astype(self.dtype)

        return output

    def operator_G(self, patches, output_image=None, output_patches=None, use_reg=False, bm3d_psd=0.1):
        xbar_image = self.xbar(patches, output_image)

        if use_reg:
            # restrict to valid region for denoiser
            block_idx = self.blk_idx
            temp_img = xbar_image[block_idx[0]: block_idx[1], block_idx[2]: block_idx[3]]

            denoised_temp_img = bm4d(temp_img, bm3d_psd, profile=BM4DProfileBM3DComplex())[:, :, 0]
            xbar_image[block_idx[0]: block_idx[1], block_idx[2]: block_idx[3]] = denoised_temp_img

        # extract patches out of image
        output_patches = self.img2patch(xbar_image, output_patches)

        return xbar_image, output_patches

    def recon(self, num_iter=100, rho=0.5, use_reg=False, sigma=0.1, joint_recon=False, beta_patch=0.5):
        """
        Args:
            num_iter: number of iterations.
            rho: Mann averaging parameter.
            use_reg: add serial regularization.
            sigma: denoising parameter in prior model.
            joint_recon: option to recover complex probe for blind ptychography.
            beta_patch: averaging weight for data-fitting.  (1 - beta) * v_j + beta * (Fourier projection)

        Returns:

        """
        # PMACE reconstruction
        probe_est = self.probe_est
        probe_mat = probe_est  # TODO: For space-varying probe, use [probe_est] * len(y_meas)

        approach = 'reg-PMACE' if use_reg else 'PMACE'
        print('{} recon starts ...'.format(approach))
        start_time = time.time()
        cur_patches = self.cur_patches
        updated_patches = np.copy(cur_patches)
        new_image = np.zeros(self.img_shape, dtype=self.dtype)
        new_patches = np.zeros(self.cur_patches.shape, dtype=self.dtype)

        for i in range(num_iter):
            # w <- F(v)
            cur_patches = self.operator_F(updated_patches, beta_patch)

            # z <- G(2w - v)
            new_image, new_patches = self.operator_G(2 * cur_patches - updated_patches, new_image, new_patches,
                                                     use_reg=use_reg)

            # v <- v + 2 \rho (z - w)
            updated_patches += 2 * rho * (new_patches - cur_patches)

            # obtain current estimate of complex images
            if not use_reg:
                est_image = self.xbar(updated_patches)
            else:
                est_image = new_image

        #     # compute the NRMSE between forward propagated reconstruction result and recorded measurements
        #     dp_est = np.abs(compute_ft(probe_est * img2patch(np.copy(obj_est), project_coords, dp.shape)))
        #     dp_nrmse_val = compute_nrmse(dp_est, dp)
        #     dp_nrmse_ls.append(dp_nrmse_val)
        #
            # phase normalization and scale image to minimize the intensity difference
            if self.ref_obj is not None:
                est_image_adj = phase_norm(est_image * self.recon_win, self.ref_obj * self.recon_win)
                obj_nrmse_val = compute_nrmse(est_image_adj * self.recon_win, self.ref_obj * self.recon_win, self.recon_win)
                self.obj_nrmse_ls.append(obj_nrmse_val)
            else:
                est_image_adj = est_image

            if (i+1) % 10 == 0:
                print('Finished {:d} of {:d} iterations.'.format(i+1, num_iter))
        #
        #     if ref_probe is not None:
        #         probe_revy = phase_norm(np.copy(probe_est), ref_probe)
        #         probe_nrmse_val = compute_nrmse(probe_revy, ref_probe)
        #         probe_nrmse_ls.append(probe_nrmse_val)
        #     else:
        #         probe_revy = probe_est
        #
            # calculate time consumption
        elapsed_time = time.time() - start_time
        print('Time consumption of {}:'.format(approach), elapsed_time)

        # save recon results
        save_tiff(est_image, self.save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
        save_array(self.obj_nrmse_ls, self.save_dir + 'obj_nrmse_' + str(self.obj_nrmse_ls[-1]))
        # save_array(dp_nrmse_ls, self.save_dir + 'diffr_nrmse')

        # return recon results
        keys = ['obj_revy', 'obj_err']
        vals = [est_image_adj, self.obj_nrmse_ls]
        output = dict(zip(keys, vals))
    
        return output
