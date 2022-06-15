from ptycho_pmace.utils.utils import *
from ptycho_pmace.utils.prior import *


def weighted_prox_map(current_est, joint_est, dp, param):
    """
    The proximal map function considering probe weights.
    :param current_est: current estimate of projected images or probe function.
    :param joint_est: current estimate of probe function or projected images.
    :param diffraction: pre-processed phase-less measurements or diffraction patterns.
    :param param: noise-to-signal ratio.
    :return: new estimate of projected images or probe function.
    """
    # FT{D*P_j*v}
    freq = compute_ft(current_est * joint_est)
    # y \times FT{D*P_j*v} / |FT{D*P_j*v}|
    freq_update = divide_cmplx_numbers(dp * np.copy(freq), np.abs(freq))
    # IFT{y \times FT{D*P_j*v} / |FT{D*P_j*v}| }
    freq_ift = compute_ift(freq_update)
    # take weighted average of current estimate and closest data-fitting point
    output = (param * current_est + divide_cmplx_numbers(freq_ift, joint_est)) / (1 + param)

    return output


def weighted_consen_operator(projected_patch, coords, norm, img_sz, add_reg=True,
                             reg_wgt=0.6, noise_std=10/255, prior_model='bm3d', block_idx=None):
    """
    The consensus operator G \left ( x \right ) = \begin{bmatrix}
                                                      \bar{x_{0}}\\
                                                      \vdots \\
                                                      \bar{x_{J-1}}
                                                   \end{bmatrix}
    where \Tilde{x_{j}} = P_{j} ((1 - \mu) \bar{x_{j}} + \mu H(\bar{x_{j}})), and
    \bar{x_{j}} = (\Lambda ^{-1} \sum_{j=1}^{J-1} P_{j}^{t} |D|^{\kappa} x_{j}.
    :param projected_patch: projected image patches.
    :param coords: coordinates of projections.
    :param norm: \Lambda ^{-1} which controls weights of pixels by the contribution to redundancy and probe weights.
    :param img_sz: size of complex image to be reconstructed.
    :param add_reg:
    :param reg_wgt: regularization weight.
    :param noise_std: standard deviation of noise, which is denoising parameter in bm3d.
    :param prior_model: choice of denoiser.
    :param block_idx: defines the region of image to be denoised.
    :return: new estimate of image patches.
    """
    # obtain complex image by processing the patches
    cmplx_img = patch2img(projected_patch, coords, img_sz, norm)

    if add_reg:
        # make a copy of the complex image as the input of denoiser
        reg_img = np.copy(cmplx_img)
        if block_idx is None:
            block_idx = [0, cmplx_img.shape[0], 0, cmplx_img.shape[1]]
        temp_img = cmplx_img[block_idx[0]: block_idx[1], block_idx[2]: block_idx[3]]

        denoised_temp_img = denoise_cmplx_bm3d(temp_img, psd=noise_std)
        reg_img[block_idx[0]: block_idx[1], block_idx[2]: block_idx[3]] = denoised_temp_img

        # use regularization weights to control the regularization
        cmplx_img = (1 - reg_wgt) * cmplx_img + reg_wgt * reg_img

    # extract patches out of image
    cmplx_patch = img2patch(cmplx_img, coords, projected_patch.shape)

    return cmplx_img, cmplx_patch


def pmace_recon(dp, project_coords, init_obj, init_probe=None, obj_ref=None, probe_ref=None,
                num_iter=100, obj_pm=1, probe_pm=1, rho=0.5, probe_exp=1.25, obj_exp=0.25,
                add_reg=True, reg_wgt=0.6, noise_std=10 / 255, prior='bm3d',
                joint_recon=False, cstr_win=None, save_dir=None):
    """
    Function to perform PMACE reconstruction on ptychographic data.
    :param dp: pre-processed diffraction pattern (intensity data).
    :param project_coords: scan coordinates of projections.
    :param init_obj: formulated initial guess of complex object.
    :param init_probe: formulated initial guess of complex probe.
    :param obj_ref: complex reference image.
    :param probe_ref: complex reference image.
    :param num_iter: number of iterations.
    :param obj_pm: noise-to-signal ratio in the data.
    :param probe_pm: noise-to-signal ratio in the data.
    :param rho: Mann averaging parameter.
    :param probe_exp: exponent of probe weighting in consensus calculation of probe estimate..
    :param obj_exp: exponent of image weighting in consensus calculation of probe estimate.
    :param add_reg: add serial regularization.
    :param reg_wgt: regularization weight.
    :param noise_std: denoising parameter required in prior model.
    :param prior: prior model.
    :param joint_recon: option to recover complex probe for blind ptychography.
    :param cstr_win: pre-defined/assigned window for comparing reconstruction results.
    :param save_dir: directory to save reconstruction results.
    :return: reconstructed images and error metrics.
    """
    # check directory
    approach = 'reg-PMACE' if add_reg else 'PMACE'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # initialization
    if cstr_win is None:
        cstr_win = np.ones_like(init_obj)

    obj_est = np.asarray(init_obj, dtype=np.complex128)
    obj_mat = img2patch(obj_est, project_coords, dp.shape)
    updated_obj_mat = np.copy(obj_mat)

    probe_est = np.asarray(init_probe, dtype=np.complex128) if joint_recon else np.copy(probe_ref)
    probe_mat = [probe_est] * len(dp)
    updated_probe_mat = np.copy(probe_mat)
    probe_concensus_output = np.copy(probe_mat)

    # determine the area for applying denoiser
    denoising_blk = patch2img(np.ones(dp.shape), project_coords, init_obj.shape)
    non_zero_idx = np.nonzero(denoising_blk)
    blk_idx = [np.amin(non_zero_idx[0]), np.amax(non_zero_idx[0]) + 1, np.amin(non_zero_idx[1]),
               np.amax(non_zero_idx[1]) + 1]

    obj_nrmse_ls = []
    probe_nrmse_ls = []
    dp_nrmse_ls = []

    # PMACE reconstruction
    print('{} recon starts ...'.format(approach))
    start_time = time.time()
    for i in range(num_iter):
        # w <- F(v; w)
        obj_mat = weighted_prox_map(updated_obj_mat, probe_concensus_output, dp, obj_pm)
        # z <- G(2w - v)
        obj_wgt = patch2img(np.ones(dp.shape, dtype=np.complex128) * (np.abs(probe_concensus_output) ** probe_exp),
                                     project_coords, init_obj.shape)
        obj_est, obj_concensus_output = weighted_consen_operator((2 * obj_mat - updated_obj_mat) * (np.abs(probe_concensus_output) ** probe_exp),
                                                        project_coords, obj_wgt, init_obj.shape, add_reg=add_reg, reg_wgt=reg_wgt,
                                                        noise_std=noise_std, prior_model=prior, block_idx=blk_idx)
        # v <- v + 2 \rho (z - w)
        updated_obj_mat += 2 * rho * (obj_concensus_output - obj_mat)
        # obtain current estimate of complex images
        if not add_reg:
            obj_est = patch2img(updated_obj_mat * (np.abs(probe_concensus_output) ** probe_exp), project_coords,
                                init_obj.shape, obj_wgt)

        if joint_recon:
            # w <- F(v; w)
            probe_mat = weighted_prox_map(updated_probe_mat, obj_concensus_output, dp,
                                          probe_pm)  # image weighted proximal map
            # z <- G(2w - v)
            # probe_concensus_output = np.average((2 * probe_mat - probe_mat_update), axis=0)
            probe_concensus_output = np.sum(
                (2 * probe_mat - updated_probe_mat) * (np.abs(obj_concensus_output) ** obj_exp), 0) / np.sum(
                (np.abs(obj_concensus_output) ** obj_exp), 0)
            # v <- v + 2 \rho (z - w)
            updated_probe_mat += 2 * rho * (probe_concensus_output - probe_mat)
            # obtain current estimate of complex images
            # probe_est = np.average(probe_mat_update, axis=0)
            probe_est = np.sum(updated_probe_mat * (np.abs(obj_concensus_output) ** obj_exp), 0) / np.sum(
                (np.abs(obj_concensus_output) ** obj_exp), 0)

        # compute the NRMSE between forward propagated reconstruction result and recorded measurements
        dp_est = np.abs(compute_ft(probe_est * img2patch(np.copy(obj_est), project_coords, dp.shape)))
        dp_nrmse_val = compute_nrmse(dp_est, dp)
        dp_nrmse_ls.append(dp_nrmse_val)

        # phase normalization and scale image to minimize the intensity difference
        if obj_ref is not None:
            obj_revy = phase_norm(np.copy(obj_est) * cstr_win, obj_ref * cstr_win)
            obj_nrmse_val = compute_nrmse(obj_revy * cstr_win, obj_ref * cstr_win, cstr_win)
            obj_nrmse_ls.append(obj_nrmse_val)
        else:
            obj_revy = obj_est

        if probe_ref is not None:
            probe_revy = phase_norm(np.copy(probe_est), probe_ref)
            probe_nrmse_val = compute_nrmse(probe_revy, probe_ref)
            probe_nrmse_ls.append(probe_nrmse_val)
        else:
            probe_revy = probe_est

        # calculate time consumption
    elapsed_time = time.time() - start_time
    print('Time consumption of {}:'.format(approach), elapsed_time)

    # save recon results
    save_tiff(obj_est, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
    save_array(obj_nrmse_ls, save_dir + 'obj_nrmse')
    save_array(dp_nrmse_ls, save_dir + 'diffr_nrmse')
    if joint_recon:
        save_tiff(probe_est, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
        save_array(probe_nrmse_ls, save_dir + 'probe_nrmse')

    # return recon results
    keys = ['obj_revy', 'probe_revy', 'obj_err', 'probe_err', 'diffr_err']
    vals = [obj_revy, probe_revy, obj_nrmse_ls, probe_nrmse_ls, dp_nrmse_ls]
    output = dict(zip(keys, vals))

    return output
