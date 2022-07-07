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
    def __init__(self, y_meas, patch_bounds, 
                 init_obj, init_probe=None, ref_obj=None, ref_probe=None, 
                 num_iter=100, recon_win=None, joint_recon=False, save_dir=None,
                 pmace_probe_exp=1.5, pmace_image_exp=0.5):
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
            joint_recon: option to recover complex probe for blind ptychography.
            save_dir: directory to save reconstruction results.
            pmace_probe_exp: exponent of probe weighting in consensus calculation of image estimate.
            pmace_image_exp: exponent of image weighting in consensus calculation of probe estimate.
        """
        # initialization 
        self.dtype_cmplx = np.complex64
        self.dtype_real = np.float64

        self.y_meas = cast(y_meas, self.dtype_real)
        self.patch_bounds = patch_bounds
        self.recon_win = np.ones_like(init_obj) if recon_win is None else recon_win
        self.joint_recon = joint_recon 
        self.save_dir = save_dir
    
        self.cur_image = cast(init_obj, self.dtype_cmplx)  # obj_est
        self.img_shape = self.cur_image.shape
        self.cur_patches = self.img2patch(self.cur_image)  # obj_mat
        self.patch_shape = self.y_meas.shape

        self.ref_obj = cast(ref_obj, self.dtype_cmplx)       # ref_obj
        self.ref_probe = cast(ref_probe, self.dtype_cmplx)   # ref_probe

        self.obj_nrmse = []
        self.probe_nrmse = []

        self.cur_probe = cast(init_probe, self.dtype_cmplx) if init_probe is not None else self.ref_probe  # probe_est
        self.cur_probe_mat = [self.cur_probe] * len(self.y_meas)
        
        # recon
        self.num_iter = num_iter
        non_zero_idx = np.nonzero(self.recon_win)
        self.blk_idx = [np.amin(non_zero_idx[0]), np.amax(non_zero_idx[0]) + 1,
                        np.amin(non_zero_idx[1]), np.amax(non_zero_idx[1]) + 1]
        # epie recon
        self.seq = np.arange(0, len(self.y_meas), 1).tolist()

        # wf recon
        self.wf_const_val = np.amax(self.patch2img(np.abs(self.cur_probe_mat, dtype=self.dtype_real) ** 2))

        # sharp(+) recon
        self.sharp_img_wgt = self.patch2img(np.abs(self.cur_probe_mat, dtype=self.dtype_real) ** 2)
        sharp_tol = np.amax([1e-3, np.amax(np.abs(self.sharp_img_wgt)) * 1e-6])
        self.sharp_img_wgt[np.abs(self.sharp_img_wgt) < sharp_tol] = sharp_tol

        # pmace recon
        self.probe_exp = cast(pmace_probe_exp, self.dtype_real)
        self.image_exp = cast(pmace_image_exp, self.dtype_real)
        self.pmace_patches = np.copy(self.cur_patches)
        #self.pmace_probe = np.copy(self.cur_probe)
        self.pmace_patch_wgt = np.abs(self.cur_probe_mat, dtype=self.dtype_real) ** self.probe_exp
        self.pmace_image_wgt = self.patch2img(self.pmace_patch_wgt)
        pmace_tol = np.amax([1e-3, np.amax(np.abs(self.pmace_image_wgt)) * 1e-6])
        self.pmace_image_wgt[np.abs(self.pmace_image_wgt) < pmace_tol] = pmace_tol


    def plot_img(self, img):
        plt.subplot(121)
        plt.imshow(np.real(img), cmap='gray')
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(np.imag(img), cmap='gray')
        plt.colorbar()
        plt.show()
        plt.clf()

    def recon(self, recon_approach='PMACE', num_iter=100, 
              epie_obj_ss=1, epie_probe_ss=1, accel_wf=True, 
              sharp_relax_param=0.75, sharp_plus_relax_param=0.8, 
              pmace_data_fit_param=0.5, pmace_rho=0.5, 
              pmace_use_reg=False, pmace_sigma=0.01):
        """
        Args:
            recon_approach: option to select reconstruction approaches,
            num_iter: number of iterations.
            save_dir: directory to save reconstruction results.
            epie_obj_ss: step size of object update using ePIE.
            epie_probe_ss: step size of probe udpate using ePIE.
            accel_wf: option to accelerate WF using Nesterov's acceleration.
            sharp_relax_param: relaxation parameter in SHARP reconstruction.
            sharp_plus_relax_param: relaxation parameter in SHARP+ reconstruction.
            pmace_data_fit_param: averaging weight in PMACE updates. Param near 1 gives closer fit to data.
            pmace_rho: Mann averaging parameter.
            pmace_use_reg: option of applying denoiser to PMACE.
            pmace_sigma: denoising parameter (psd of complex bm3d software).
        Returns:
            Reconstructed complex images, and errors between reconstructions and reference images. 
        """
        # initialization
        est_image, old_est_image = np.copy(self.cur_image), np.copy(self.cur_image)
        est_probe, old_est_probe = np.copy(self.cur_probe), np.copy(self.cur_probe)
        pmace_args = dict(data_fit_param=pmace_data_fit_param, rho=pmace_rho, use_reg=pmace_use_reg, sigma=pmace_sigma)
        save_dir = self.save_dir + '{}/'.format(recon_approach) 
        self.check_fpath(save_dir)

        # reconstruction
        start_time = time.time()
        print('{} reconstruction starts ...'.format(recon_approach))
        for i in range(self.num_iter):
            # ePIE reconstruction
            if recon_approach == 'ePIE':
                est_image, est_probe = self.epie_iter(input_image=est_image, input_probe=est_probe, obj_step_sz=epie_obj_ss)
            # WF/AWF reconstruction
            if recon_approach == 'WF' or 'AWF':
                # AWF reconstruction
                if accel_wf:
                    beta = (i + 2) / (i + 4)
                    cur_est_image = est_image + beta * (est_image - old_est_image)
                    cur_est_probe = est_probe + beta * (est_probe - old_est_probe)
                    old_est_image, old_est_probe = est_image, est_probe
                    est_image, est_probe = self.wf_iter(input_image=cur_est_image, input_probe=cur_est_probe)
                # WF recosntruction
                else:
                    est_image, est_probe = self.wf_iter(input_image=est_image, input_probe=est_probe)

            # SHARP/SHARP+ reconstruction
            if recon_approach == 'SHARP':
                est_image, est_probe = self.sharp_iter(beta=sharp_relax_param, input_image=est_image, input_probe=est_probe)
            if recon_approach == 'SHARP+':
                est_image, est_probe = self.sharp_plus_iter(beta=sharp_plus_relax_param, input_image=est_image, input_probe=est_probe)

            if recon_approach == 'PMACE' or 'reg-PMACE':
                est_image, est_probe = self.pmace_iter(input_image=est_image, input_probe=est_probe, **pmace_args)

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
        print('Time consumption of {}:'.format(recon_approach), time.time() - start_time)

        # save recon results
        save_tiff(est_image, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
        save_array(self.obj_nrmse, save_dir  + 'obj_nrmse_' + str(self.obj_nrmse[-1]))
        if self.joint_recon:
            save_tiff(est_probe, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
            save_array(probe_nrmse, save_dir + 'probe_nrmse_' + str(self.probe_nrmse[-1]))

        # return recon results
        keys = ['obj_revy', 'obj_err', 'probe_revy', 'probe_err']
        vals = [est_image_adj, self.obj_nrmse, est_probe_adj, self.probe_nrmse]
        output = dict(zip(keys, vals))

        return output



    def epie_iter(self, input_image=None, input_probe=None, obj_step_sz=0.1, probe_step_sz=0.1):
        """
        Use ePIE to update the current estimate of complex object image and/or complex probe.
        Args:
            input_image: current estimate of complex image.
            input_probe: current estimate of complex probe.
            obj_step_sz: step size of image update.
            probe_step_sz: step size of probe update.
        Returns:
            output_image: new estimate of complex image.
            output_probe: new estimate of complex probe.
        """
        # initialization 
        input_image = self.cur_image if input_image is None else input_image
        input_probe = self.cur_probe if input_probe is None else input_probe
        output_image, output_probe = np.copy(input_image), np.copy(input_probe)
       
        # shuffle the scan points and perform ePIE to revise estimates
        random.shuffle(self.seq)
        for idx in self.seq:
            # revise estimates on each scan point
            crd0, crd1, crd2, crd3 = self.patch_bounds[idx, 0], self.patch_bounds[idx, 1], self.patch_bounds[idx, 2], self.patch_bounds[idx, 3]
            projected_img = np.copy(input_image[crd0:crd1, crd2:crd3])
            frm = projected_img * input_probe
            freq = compute_ft(frm)
            freq_update = (self.y_meas[idx] * freq / np.abs(freq)).astype(self.dtype_cmplx)
            delta_frm = compute_ift(freq_update) - frm
            # obtain new estimate of complex image
            output_image[crd0:crd1, crd2:crd3] += obj_step_sz * np.conj(input_probe) * delta_frm / (np.amax(np.abs(input_probe)) ** 2)
            # obtain new estaimte of complex probe
            if self.joint_recon:
                output_probe += probe_step_sz * np.conj(projected_img) * delta_frm / (np.amax(np.abs(projected_img)) ** 2)

        return output_image, output_probe

    def wf_iter(self, input_image=None, input_probe=None):
        """
        Use WF/AWF to update the current estimate of complex object image and/or complex probe.
        Args:
            input_image: current estimate of complex image.
            input_probe: current estimate of complex probe.
        Returns:
            output_image: new estimate of complex image.
            output_probe: new estimate of complex probe.
        """
        # initialization 
        input_image = self.cur_image if input_image is None else input_image
        input_probe = self.cur_probe if input_probe is None else input_probe
        output_image, output_probe = np.copy(input_image), np.copy(input_probe)

        # revise estiamte of complex image
        projected_patches = self.img2patch(input_image)
        # Ax = FT{D*P_j*v}
        freq = compute_ft(projected_patches * input_probe)
        # Ax - y * Ax / |Ax| = - FT{D * P_j * v} - y * FT{D * P_j * v} / |FT{D * P_j * v}|
        freq_update = freq - self.y_meas * freq / np.abs(freq)
        # A^(H){```} = P^t D^* IFFT{```}
        frm = compute_ift(freq_update)
        # step_sz = {max eigenvalue of semi positive deifnite matrix}^(-1) = {1/\lambda_max(A^(H)A)=1/(alpha*sum_j |D_j|^2)}^(-1)
        output_image = input_image - self.patch2img(frm * np.conj(input_probe)) / self.wf_const_val

        # revise estimate of complex probe
        if self.joint_recon:
            projected_patches = self.img2patch(output_image)
            # calculate step size of probe udpate
            mu = 1 / np.amax(np.sum(np.abs(projected_patches, dtype=self.dtype_real) ** 2, 0))
            # Bd = FT{X_j*d}
            freq = compute_ft(projected_patches * input_probe)
            # Bd - y * Bd / |Bd|
            freq_update = freq - self.y_meas * freq / np.abs(freq)
            # B^(H){```} = (X_j)^(H) IFFT{```}
            probe_mat = mu * (input_probe - frm * np.conj(projected_patches)) 
            # obtain new estimate of probe
            output_probe = np.average(probe_mat, axis=0)
            # update step size of image update
            self.wf_const_val = np.amax(self.patch2img(np.abs(output_probe, dtype=self.dtype_real) ** 2))

        return output_image, output_probe

    def sharp_iter(self, beta, input_image=None, input_probe=None):
        """
        Use SHARP to update the current estimate of complex object image and/or complex probe.
        Args:
            beta: relaxation parameter.
            input_image: current estimate of complex image.
            input_probe: current estimate of complex probe.
        Returns:
            output_image: new estimate of complex image.
            output_probe: new estimate of complex probe.
        """
        # initialization 
        input_image = self.cur_image if input_image is None else input_image
        input_probe = self.cur_probe if input_probe is None else input_probe
        output_image, output_probe = np.copy(input_image), np.copy(input_probe)
        est_frm = self.img2patch(input_image) * input_probe
        
        # projections
        frm_proj_f = self.projector_fourier(est_frm)
        frm_proj_s = self.projector_space(est_frm, est_probe, img_wgt)
        frm_update = 2 * beta * self.projector_space(frm_proj_f, est_probe, img_wgt) + (1 - 2 * beta) * frm_proj_f + beta * (frm_proj_s - est_frm)
        est_frm = frm_update

        # obtain current estimate of complex image
        output_image = self.patch2img(est_frm * np.conj(est_probe)) / self.sharp_img_wgt
        
        # obtain current estimate of complex probe
        if self.joint_recon:
            tmp_n = np.average(self.img2patch(np.conj(output_image) * est_frm), axis=0)
            tmp_d = np.average(self.img2patch(np.abs(output_image) ** 2), axis=0)
            output_probe = np.divide(tmp_n, tmp_d, where=(tmp_d != 0))
            # update image weight
            self.sharp_img_wgt = self.patch2img(np.abs([output_probe] * len(self.y_meas), dtype=self.dtype_real) ** 2)
            sharp_tol = np.amax([1e-3, np.amax(np.abs(self.sharp_img_wgt)) * 1e-6])
            self.sharp_img_wgt[np.abs(self.sharp_img_wgt) < sharp_tol] = sharp_tol

        return output_image, output_probe

    def sharp_plus_iter(self, beta, input_image=None, input_probe=None):
        """
        Use SHARP+ to update the current estimate of complex object image and/or complex probe.
        Args:
            beta: relaxation parameter.
            input_image: current estimate of complex image.
            input_probe: current estimate of complex probe.
        Returns:
            output_image: new estimate of complex image.
            output_probe: new estimate of complex probe.
        """
        # initialization 
        input_image = self.cur_image if input_image is None else input_image
        input_probe = self.cur_probe if input_probe is None else input_probe
        output_image, output_probe = np.copy(input_image), np.copy(input_probe)
        est_frm = self.img2patch(input_image) * input_probe

        # projections
        frm_proj_f = self.projector_fourier(est_frm)
        frm_proj_s = self.projector_space(est_frm, est_probe, img_wgt)
        frm_update = 2 * beta * self.projector_space(frm_proj_f, est_probe, img_wgt) + (1 - 2 * beta) * frm_proj_f - beta * (frm_proj_s - est_frm)
        est_frm = frm_update

        # obtain current estimate of complex image
        output_image = self.patch2img(est_frm * np.conj(est_probe)) / self.sharp_img_wgt

        # obtain current estimate of complex probe
        if self.joint_recon:
            tmp_n = np.average(self.img2patch(np.conj(output_image) * est_frm), axis=0)
            tmp_d = np.average(self.img2patch(np.abs(output_image) ** 2), axis=0)
            output_probe = np.divide(tmp_n, tmp_d, where=(tmp_d != 0))
            # update image weight
            self.sharp_img_wgt = self.patch2img(np.abs([output_probe] * len(self.y_meas), dtype=self.dtype_real) ** 2)
            sharp_tol = np.amax([1e-3, np.amax(np.abs(self.sharp_img_wgt)) * 1e-6])
            self.sharp_img_wgt[np.abs(self.sharp_img_wgt) < sharp_tol] = sharp_tol

        return output_image, output_probe

    def pmace_iter(self, input_image=None, input_probe=None, data_fit_param=0.5, rho=0.5, use_reg=False, sigma=0.01):
        """
        Use PMACE to update the current estimate of complex object image and/or complex probe.
        Args:
            input_image: current estimate of complex image.
            input_probe: current estimate of complex probe.
            data_fit_param: averaging weight in PMACE updates. Param near 1 gives closer fit to data.
            rho: Mann averaging parameter.
            use_reg: option of applying denoiser to PMACE.
            sigma: denoising parameter (psd of complex bm3d software).

        Returns:
            output_image: new estimate of complex image.
            output_probe: new estimate of complex probe.
        """
        # initialization
        input_image = self.cur_image if input_image is None else input_image
        input_probe = self.cur_probe if input_probe is None else input_probe
        output_image, output_probe = np.copy(input_image), np.copy(input_probe)
        
        # w <- F(v)
        cur_patches = self.operator_F(self.pmace_patches, input_probe, data_fit_param)
        # z <- G(2w - v)
        new_image, new_patches = self.operator_G(2 * cur_patches - self.pmace_patches, input_probe, use_reg=use_reg, bm3d_psd=sigma)
        # v <- v + 2 \rho (z - w)
        self.pmace_patches = self.pmace_patches + 2 * rho * (new_patches - cur_patches)
        # obtain current estimate of complex image
        output_image = new_image if use_reg else self.xbar(self.pmace_patches)

        # joint reconstruction
        if self.joint_recon:
            # w <- F(v)
            cur_probe = self.operator_F(self.pmace_probe, new_patches, data_fit_param)
            # z <- G(2w - v)
            new_probe = self.dbar((2 * cur_probe - self.pmace_probe), new_patches, image_exp)
            # v <- v + 2 \rho (z - w)
            self.pmace_probe = self.pmace_probe + 2 * rho * (new_probe - cur_probe)
            # obtain current estimate of complex probe
            output_probe = self.dbar(self.pmace_probe, new_patches, image_exp)
            # update patch weight and image weight
            self.pmace_patch_wgt = np.abs([output_probe] * len(self.y_meas), dtype=self.dtype_real) ** self.probe_exp
            self.pmace_image_wgt = self.patch2img(self.pmace_patch_wgt)
            pmace_tol = np.amax([1e-3, np.amax(np.abs(self.pmace_image_wgt)) * 1e-6])
            self.pmace_image_wgt[np.abs(self.pmace_image_wgt) < pmace_tol] = pmace_tol

        return output_image, output_probe




#    def epie_recon(self, obj_step_sz=0.1, probe_step_sz=0.1, save_dir=None):
#        self.check_fpath(save_dir)
#        start_time = time.time()
#        seq = np.arange(0, len(self.y_meas), 1).tolist()
#        est_image = np.copy(self.cur_image)
#        est_probe = np.copy(self.cur_probe)
#        crds = self.patch_bounds
#        approach = 'ePIE'
#        print('{} reconstruction starts ...'.format(approach))
#        for i in range(self.num_iter):
#            random.shuffle(seq)
#            for j in seq:
#                crd0, crd1, crd2, crd3 = crds[j, 0], crds[j, 1], crds[j, 2], crds[j, 3]
#                projected_img = np.copy(est_image[crd0:crd1, crd2:crd3])
#                frm = projected_img * est_probe
#                freq = compute_ft(frm)
#                freq_update = (self.y_meas[j] * freq / np.abs(freq)).astype(self.dtype_cmplx)
#                delta_frm = compute_ift(freq_update) - frm
#                est_image[crd0:crd1, crd2:crd3] += obj_step_sz * np.conj(est_probe) * delta_frm / (np.amax(np.abs(est_probe)) ** 2)
#                if self.joint_recon:
#                    est_probe += probe_step_sz * np.conj(projected_img) * delta_frm / (np.amax(np.abs(projected_img)) ** 2)
#
#            # phase normalization and scale image to minimize the intensity difference
#            if self.ref_obj is not None:
#                est_image_adj = phase_norm(est_image * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
#                cur_obj_nrmse = compute_nrmse(est_image_adj * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
#                self.obj_nrmse.append(cur_obj_nrmse)
#                print(i, cur_obj_nrmse)
#            else:
#                est_image_adj = est_image
#
#            # phase normalization and scale image to minimize the intensity difference
#            if self.ref_probe is not None:
#                est_probe_adj = phase_norm(est_probe, self.ref_probe)
#                cur_probe_nrmse = compute_nrmse(est_probe_adj, self.ref_probe)
#                self.probe_nrmse.append(cur_probe_nrmse)
#            else:
#                est_probe_adj = est_probe
#
#            if (i+1) % 10 == 0:
#                print('Finished {:d} of {:d} iterations.'.format(i+1, self.num_iter))
#
#        # calculate time consumption
#        elapsed_time = time.time() - start_time
#        print('Time consumption of {}:'.format(approach), elapsed_time)
#
#        # save recon results
#        save_tiff(est_image, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
#        save_array(self.obj_nrmse, save_dir + 'obj_nrmse_' + str(self.obj_nrmse[-1]))
#        if self.joint_recon:
#            save_tiff(est_probe, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
#            save_array(probe_nrmse, save_dir + 'probe_nrmse_' + str(self.probe_nrmse[-1]))
#
#        # return recon results
#        keys = ['obj_revy', 'obj_err', 'probe_revy', 'probe_err']
#        vals = [est_image_adj, self.obj_nrmse, est_probe_adj, self.probe_nrmse]
#        output = dict(zip(keys, vals))
#
#        return output


    def projector_fourier(self, frm):
        freq = compute_ft(frm)
        freq_update = (self.y_meas * freq / np.abs(freq)).astype(self.dtype_cmplx)
        output = compute_ift(freq_update)
        return output

    def projector_space(self, frm, probe, img_wgt):
        img_wgt[img_wgt < 1e-3] = 1e-3
        tmp_img = self.patch2img(frm * np.conj(probe)) / img_wgt
        output = self.img2patch(tmp_img) * probe
        return output

    def reflected_resolvent_fourier(self, frm):
        return 2 * self.projector_fourier(frm) - frm

    def reflected_resolvent_space(self, frm, probe_est, img_wgt):
        return 2 * self.projector_space(frm, probe_est, img_wgt) - frm

#    def sharp_recon(self, relax_param=0.75, save_dir=None):
#        # check directory
#        self.check_fpath(save_dir)
#        # initialization
#        beta = relax_param
#        #est_image = np.copy(self.cur_image)
#        est_probe = np.copy(self.cur_probe)
#        est_frm = self.cur_patches * est_probe
#        probe_mat = [est_probe] * len(self.y_meas)
#        img_wgt = self.patch2img(np.abs(probe_mat) ** 2)
#        img_wgt[img_wgt < 1e-3] = 1e-3
#        est_probe_adj = est_probe
#
#        # SHARP reconstruction
#        print('SHARP reconstruction starts ...')
#        start_time = time.time()
#        for i in range(self.num_iter):
#            # projection
#            frm_proj_f = self.projector_fourier(est_frm)
#            frm_proj_s = self.projector_space(est_frm, est_probe, img_wgt)
#            frm_update = 2 * beta * self.projector_space(frm_proj_f, est_probe, img_wgt) + (1 - 2 * beta) * frm_proj_f + beta * (frm_proj_s - est_frm)
#            est_frm = frm_update
#        
#            # obtain current estimate of complex image
#            est_image = self.patch2img(est_frm * np.conj(est_probe)) / img_wgt
#            
#            # phase normalization and scale image to minimize the intensity difference
#            if self.ref_obj is not None:
#                est_image_adj = phase_norm(est_image * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
#                cur_obj_nrmse = compute_nrmse(est_image_adj * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
#                self.obj_nrmse.append(cur_obj_nrmse)
#                print(i, cur_obj_nrmse)
#            else:
#                est_image_adj = est_image
#
#            # joint reconstruction
#            if self.joint_recon:
#                tmp_n = np.average(self.img2patch(np.conj(est_image) * est_frm), axis=0)
#                tmp_d = np.average(self.img2patch(np.abs(est_image) ** 2), axis=0)
#                est_probe = np.divide(tmp_n, tmp_d, where=(tmp_d != 0))
#                # update image weight
#                probe_mat = [est_probe] * len(self.y_meas)
#                img_wgt = self.patch2img(np.abs(probe_mat) ** 2)
#                img_wgt[img_wgt < 1e-3] = 1e-3
#
#                # phase normalization and scale image to minimize the intensity difference
#                if self.ref_probe is not None:
#                    est_probe_adj = phase_norm(est_probe, self.ref_probe)
#                    cur_probe_nrmse = compute_nrmse(est_probe_adj, self.ref_probe)
#                    self.probe_nrmse.append(cur_probe_nrmse)
#                else:
#                    est_probe_adj = est_probe
#
#            if (i+1) % 10 == 0:
#                print('Finished {:d} of {:d} iterations.'.format(i+1, self.num_iter))
#
#        # calculate time consumption
#        print('Time consumption of SHARP:', time.time() - start_time)
#
#        # save recon results
#        save_tiff(est_image, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
#        save_array(self.obj_nrmse, save_dir + 'obj_nrmse_' + str(self.obj_nrmse[-1]))
#        if self.joint_recon:
#            save_tiff(est_probe, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
#            save_array(probe_nrmse, save_dir + 'probe_nrmse_' + str(self.probe_nrmse[-1]))
#
#        # return recon results
#        keys = ['obj_revy', 'obj_err', 'probe_revy', 'probe_err']
#        vals = [est_image_adj, self.obj_nrmse, est_probe_adj, self.probe_nrmse]
#        output = dict(zip(keys, vals))
#    
#        return output
        
#   def sharp_plus_recon(self, relax_param=0.75, save_dir=None):
#       # check directory
#       self.check_fpath(save_dir)
#       # initialization
#       beta = relax_param
#       #est_image = np.copy(self.cur_image)
#       est_probe = np.copy(self.cur_probe)
#       est_frm = self.cur_patches * est_probe
#       probe_mat = [est_probe] * len(self.y_meas)
#       img_wgt = self.patch2img(np.abs(probe_mat) ** 2)
#       img_wgt[img_wgt < 1e-3] = 1e-3
#       est_probe_adj = est_probe
#
#        # SHARP reconstruction
#        print('SHARP reconstruction starts ...')
#        start_time = time.time()
#        for i in range(self.num_iter):
#           # projection
#            frm_proj_f = self.projector_fourier(est_frm)
#            frm_proj_s = self.projector_space(est_frm, est_probe, img_wgt)
#            frm_update = 2 * beta * self.projector_space(frm_proj_f, est_probe, img_wgt) + (1 - 2 * beta) * frm_proj_f - beta * (frm_proj_s - est_frm)
#            est_frm = frm_update
#
#            # obtain current estimate of complex image
#            est_image = self.patch2img(est_frm * np.conj(est_probe)) / img_wgt
#
#            # phase normalization and scale image to minimize the intensity difference
#            if self.ref_obj is not None:
#                est_image_adj = phase_norm(est_image * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
#                cur_obj_nrmse = compute_nrmse(est_image_adj * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
#                self.obj_nrmse.append(cur_obj_nrmse)
#                print(i, cur_obj_nrmse)
#            else:
#                est_image_adj = est_image
#            # joint reconstruction
#            if self.joint_recon:
#                tmp_n = np.average(self.img2patch(np.conj(est_image) * est_frm), axis=0)
#                tmp_d = np.average(self.img2patch(np.abs(est_image) ** 2), axis=0)
#                est_probe = np.divide(tmp_n, tmp_d, where=(tmp_d != 0))
#                # update image weight
#                probe_mat = [est_probe] * len(self.y_meas)
#                img_wgt = self.patch2img(np.abs(probe_mat) ** 2)
#                img_wgt[img_wgt < 1e-3] = 1e-3
#
#                # phase normalization and scale image to minimize the intensity difference
#                if self.ref_probe is not None:
#                    est_probe_adj = phase_norm(est_probe, self.ref_probe)
#                    cur_probe_nrmse = compute_nrmse(est_probe_adj, self.ref_probe)
#                    self.probe_nrmse.append(cur_probe_nrmse)
#                else:
#                    est_probe_adj = est_probe
#
#            if (i+1) % 10 == 0:
#                print('Finished {:d} of {:d} iterations.'.format(i+1, self.num_iter))
#
#        # calculate time consumption
#        print('Time consumption of SHARP:', time.time() - start_time)
#
#        # save recon results
#        save_tiff(est_image, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
#        save_array(self.obj_nrmse, save_dir + 'obj_nrmse_' + str(self.obj_nrmse[-1]))
#        if self.joint_recon:
#            save_tiff(est_probe, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
#            save_array(probe_nrmse, save_dir + 'probe_nrmse_' + str(self.probe_nrmse[-1]))
#
#        # return recon results
#        keys = ['obj_revy', 'obj_err', 'probe_revy', 'probe_err']
#        vals = [est_image_adj, self.obj_nrmse, est_probe_adj, self.probe_nrmse]
#        output = dict(zip(keys, vals))
#
#        return output


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
        #if patch_wgt is None:
        #    patch_wgt = np.ones(self.patch_shape, dtype=self.dtype_real)
        #if image_wgt is None:
        #    image_wgt = self.patch2img(patches)
        
        # initialization
        output_image = np.zeros(self.img_shape, dtype=self.dtype_cmplx)
        weighted_patches = (self.pmace_patch_wgt * patches).astype(self.dtype_cmplx)

        # Back projection from patches to image
        crds = self.patch_bounds
        for j in range(len(patches)):
            output_image[crds[j, 0]:crds[j, 1], crds[j, 2]:crds[j, 3]] += weighted_patches[j]
        
        # Normalization 
        if normalize:
            output_image = output_image / self.pmace_image_wgt
        
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


#    def pmace_recon(self, data_fit_param=0.5, rho=0.5, probe_exp=1.5, image_exp=0.5, use_reg=False, sigma=0.1, save_dir=None):
#        if save_dir is not None:
#            os.makedirs(save_dir, exist_ok=True)
#        # initializatioin
#        updated_patches = np.copy(self.cur_patches)
#        new_probe = [self.cur_probe] * len(self.y_meas)
#        patch_wgt = np.abs(new_probe, dtype=self.dtype_real) ** probe_exp
#        image_wgt = self.patch2img(patch_wgt)
#        approach = 'reg-PMACE' if use_reg else 'PMACE'
#        print('{} reconstruction starts ...'.format(approach))
#
#        start_time = time.time()
#        
#        for i in range(self.num_iter):
#            # w <- F(v)
#            cur_patches = self.operator_F(updated_patches, new_probe, data_fit_param)
#
#            # z <- G(2w - v)
#            new_image, new_patches = self.operator_G(2 * cur_patches - updated_patches, new_probe, use_reg=use_reg, bm3d_psd=sigma, patch_wgt=patch_wgt, image_wgt=image_wgt)
#
#            # v <- v + 2 \rho (z - w)
#            updated_patches += 2 * rho * (new_patches - cur_patches)
#
#            # obtain current estimate of complex image
#            est_image = new_image if use_reg else self.xbar(updated_patches, patch_wgt=patch_wgt, image_wgt=image_wgt)
#            
#            # phase normalization and scale image to minimize the intensity difference
#            if self.ref_obj is not None:
#                est_image_adj = phase_norm(est_image * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
#                cur_obj_nrmse = compute_nrmse(est_image_adj * self.recon_win, self.ref_obj * self.recon_win, cstr=self.recon_win)
#                self.obj_nrmse.append(cur_obj_nrmse)
#                print(i, cur_obj_nrmse)
#            else:
#                est_image_adj = est_image
#
#            # joint reconstruction
#            if self.joint_recon:
#                # w <- F(v)
#                cur_probe = self.operator_F(updated_probe, new_patches, data_fit_param)
#                # z <- G(2w - v)
#                new_probe = self.dbar((2 * cur_probe - new_probe), new_patches, image_exp)
#                # v <- v + 2 \rho (z - w)
#                updated_probe += 2 * rho * (new_probe - cur_probe)
#                # obtain current estimate of complex probe
#                est_probe = self.dbar(updated_probe, new_patches, image_exp)
#                # update patch weight and image weight
#                patch_wgt = np.abs(new_probe, dtype=self.dtype_real) ** probe_exp
#                image_wgt = self.patch2img(patch_wgt)
#
#                # phase normalization and scale image to minimize the intensity difference
#                if self.ref_probe is not None:
#                    est_probe_adj = phase_norm(est_probe, self.ref_probe)
#                    cur_probe_nrmse = compute_nrmse(est_probe_adj, self.ref_probe)
#                    self.probe_nrmse.append(cur_probe_nrmse)
#                else:
#                    est_probe_adj = est_probe
#
#            if (i+1) % 10 == 0:
#                print('Finished {:d} of {:d} iterations.'.format(i+1, self.num_iter))
#
#        print(est_image_adj)
#
#        # calculate time consumption
#        elapsed_time = time.time() - start_time
#        print('Time consumption of {}:'.format(approach), elapsed_time)
#
#        # save recon results
#        save_tiff(est_image, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
#        save_array(self.obj_nrmse, save_dir + 'obj_nrmse_' + str(self.obj_nrmse[-1]))
#        if self.joint_recon:
#            save_tiff(est_probe, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
#            save_array(probe_nrmse, save_dir + 'probe_nrmse_' + str(self.probe_nrmse[-1]))
#
#        # return recon results
#        keys = ['obj_revy', 'obj_err', 'probe_revy', 'probe_err']
#        vals = [est_image_adj, self.obj_nrmse, est_probe_adj, self.probe_nrmse]
#        output = dict(zip(keys, vals))
#    
#        return output



