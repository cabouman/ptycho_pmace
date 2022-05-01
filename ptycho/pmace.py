from ptycho_pmace.utils.utils import *
from ptycho_pmace.utils.prior import *


def weighted_prox_map(current_est, joint_est, diffraction, param):
    """
    The proximal map function considering probe weights.
    :param current_est: current estimate of projected images or probe function.
    :param joint_est: current estimate of probe function or projected images.
    :param diffraction: pre-processed phaseless measurements or diffraction patterns.
    :param param: noise-to-signal ratio.
    :return: new estimate of projected images or probe function.
    """
    # FT{D*P_j*v}
    freq = compute_ft(current_est * joint_est)
    # y \times FT{D*P_j*v} / |FT{D*P_j*v}|
    freq_update = divide_cmplx_numbers(diffraction * np.copy(freq), np.abs(freq))
    # IFT{y \times FT{D*P_j*v} / |FT{D*P_j*v}| }
    freq_ift = compute_ift(freq_update)
    # take weighted average of current estimate and closest data-fitting point
    output = (param * current_est + divide_cmplx_numbers(freq_ift, joint_est)) / (1 + param)

    return output


def weighted_consen_operator(projected_patch, coords, norm, img_sz):
    """
    The consensus operator G \left ( x \right ) = \begin{bmatrix}
                                                      \bar{x_{0}}\\
                                                      \vdots \\
                                                      \bar{x_{J-1}}
                                                   \end{bmatrix}
    where \bar{x_{j}} = P_{j} \Lambda ^{-1} \sum_{j=1}^{J-1} P_{j}^{t} |D|^{\kappa} x_{j}.
    :param projected_patch: projected image patches.
    :param coords: coordinates of projections.
    :param norm: \Lambda ^{-1} which controls weights of pixels by the contribution to redundancy and probe weights.
    :param img_sz: size of complex image to be reconstructed.
    :return: new estimate of image patches.
    """
    # put projected patches back to full-sized image with probe-weighting
    cmplx_img = patch2img(projected_patch, coords, img_sz, norm)
    # extract patches out of image
    output = img2patch(cmplx_img, coords, projected_patch.shape)

    return output


def pmace_recon(init_guess, diffr_data, coords, obj_ref, probe_ref,
                num_iter=100, obj_nsr_pm=1, rho=0.5, probe_exp=1.25, cstr_win=None, save_dir=None):
    """
    PMACE reconstruction to perform single estimate on complex transmittance of complex
    object assuming known complex probe function. The Mann iteration is given by:
        Initialization
        While not converged do {
            w <- F(v; w)
            z <- G(2w - v)
            v <- v + 2 \rho (z - w)
        }
        return x <- v
    :param init_guess: formulated initial guess of complex transmittance of complex object image.
    :param diffr_data: pre-processed diffraction patterns (phase-less measurements).
    :param coords: scan coordinates of projections.
    :param obj_ref: ground truth image or reference image.
    :param probe_ref: known or estimated complex probe function.
    :param num_iter: number of iterations.
    :param obj_nsr_pm: noise-to-signal ration parameter in probe-weighted proximal function.
    :param rho: Mann averaging parameter.
    :param probe_exp: probe exponent.
    :param cstr_win: pre-defined cover/window for comparing reconstruction results.
    :param save_dir: save reconstruction results to the given directory.
    :return: reconstructed complex transmittance of complex object and error metrics.
    """
    # check directories
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    if cstr_win is None:
        cstr_win = np.ones_like(init_guess)
        
    # initialization
    num_agts, m, n = diffr_data.shape
    probe = probe_ref
    consen_norm = patch2img(np.ones(diffr_data.shape, dtype=np.complex128) * (np.abs(probe) ** probe_exp), coords, init_guess.shape)
    x_mat = img2patch(init_guess, coords, [num_agts, m, n])
    v_mat = np.copy(x_mat)
    obj_nrmse_ls = []
    diffr_nrmse_ls = []
    time_ls = []
    start_time = time.time()

    # PMACE reconstruction
    print('PMACE starts ...')
    for i in range(num_iter):
        # w <- F(v; w)
        x_mat = weighted_prox_map(v_mat, probe, diffr_data, obj_nsr_pm)

        # z <- G(2w - v)
        x_cons = weighted_consen_operator((2 * x_mat - v_mat) * (np.abs(probe)**probe_exp), coords, consen_norm, init_guess.shape)

        # v <- v + 2 \rho (z - w)
        v_mat += 2 * rho * (x_cons - x_mat)

        # phase normalization and obtain the new estimate of complex object of current iteration
        obj_est = patch2img(v_mat * (np.abs(probe)**probe_exp), coords, init_guess.shape, consen_norm)

        # calculate time consumption
        elapsed_time = time.time() - start_time
        time_ls.append(elapsed_time)

        # compute the nrmse error between propagted reconstruction result and recorded measurements
        diffraction_est = np.abs(compute_ft(probe * img2patch(np.copy(obj_est), coords, diffr_data.shape)))
        diffr_nrmse_val = compute_nrmse(diffraction_est, diffr_data, np.ones(diffr_data.shape))
        diffr_nrmse_ls.append(diffr_nrmse_val)

        # phase normalization and compute the nrmse error between reconstructed image and reference image
        obj_revy = phase_norm(np.copy(obj_est) * cstr_win, np.copy(obj_ref) * cstr_win)
        obj_nrmse_val = compute_nrmse(obj_revy * cstr_win, obj_ref * cstr_win, cstr_win)
        obj_nrmse_ls.append(obj_nrmse_val)
        print('iter =', i, 'img_error =', obj_nrmse_val, 'diffr_err =', diffr_nrmse_val)

    # save recon results
    save_tiff(obj_est, save_dir + 'iter_{}.tiff'.format(i + 1))
    save_array(obj_nrmse_ls, save_dir + 'obj_nrmse')
    save_array(diffr_nrmse_ls, save_dir + 'diffr_nrmse')

    # return the result
    keys = ['obj_revy', 'obj_err', 'diffr_err']
    vals = [obj_revy, obj_nrmse_ls, diffr_nrmse_ls]
    output = dict(zip(keys, vals))

    return output


def reg_consen_operator(projected_patch, coords, norm, img_sz,
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
    :param reg_wgt: regularization weight.
    :param noise_std: standard deviation of noise, which is denoising parameter in bm3d.
    :param prior_model: choice of denoiser.
    :param block_idx: defines the region of image to be denoised.
    :return: new estimate of image patches.
    """

    # obtain complex image by processing the patches
    cmplx_img = patch2img(projected_patch, coords, img_sz, norm)
    # make a copy of the complex image as the input of denoiser
    reg_img = np.copy(cmplx_img)
    if block_idx is None:
        block_idx = [0, cmplx_img.shape[0], 0, cmplx_img.shape[1]]
    temp_img = cmplx_img[block_idx[0]: block_idx[1], block_idx[2]: block_idx[3]]

    # # apply denoiser (prior model)
    # if model_name in ['dncnn_gray_blind', 'dncnn_15', 'dncnn_25', 'dncnn_50']:
    #     denoised_temp_img = denoise_dncnn(temp_img, model_name=model_name)
    # else:
    #     denoised_temp_img = denoise_cmplx_bm3d(temp_img, psd=noise_std)
    denoised_temp_img = denoise_cmplx_bm3d(temp_img, psd=noise_std)
    reg_img[block_idx[0]: block_idx[1], block_idx[2]: block_idx[3]] = denoised_temp_img

    # use regularization weights to control the regularization
    cmplx_img = (1 - reg_wgt) * cmplx_img + reg_wgt * reg_img
    # extract patches out of image
    output = img2patch(cmplx_img, coords, projected_patch.shape)

    return output


def reg_pmace_recon(init_guess, diffr_data, coords, obj_ref, probe_ref, num_iter=100, obj_nsr_pm=0.5,
                    rho=0.5, probe_exp=1.25, reg_wgt=0.6, noise_std=10/255, prior_model='bm3d',
                    cstr_win=None, save_dir=None):
    """
    This function add regularization to PMACE formulation.
    :param init_guess: formulated initial guess of complex transmittance of complex object image.
    :param diffr_data: pre-processed diffraction patterns (phase-less measurements).
    :param coords: scan coordinates of projections.
    :param obj_ref: ground truth image or reference image.
    :param probe_ref: known or estimated complex probe function.
    :param num_iter: number of iterations.
    :param obj_nsr_pm: noise-to-signal ration parameter in probe-weighted proximal function.
    :param rho: Mann averaging parameter.
    :param probe_exp: probe exponent.
    :param reg_wgt: regularization weight.
    :param noise_std: denoising parameter of denoiser.
    :param prior_model: default prior model is complex bm3d software.
    :param cstr_win: pre-defined cover/window for comparing reconstruction results.
    :param save_dir: save reconstruction results to the given directory.
    :return: reconstructed complex transmittance of complex object and error metrics.
    """
    # check directories
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    if cstr_win is None:
        cstr_win = np.ones_like(init_guess)

    # initialization
    num_agts, m, n = diffr_data.shape
    probe = np.copy(probe_ref)
    consen_norm = patch2img([np.abs(probe_ref) ** probe_exp] * num_agts, coords, init_guess.shape)
    denoising_blk = patch2img(np.ones(diffr_data.shape), coords, init_guess.shape)
    non_zero_idx = np.nonzero(denoising_blk)
    blk_idx = [np.amin(non_zero_idx[0]), np.amax(non_zero_idx[0])+1, np.amin(non_zero_idx[1]), np.amax(non_zero_idx[1])+1]

    x_mat = img2patch(init_guess, coords, [num_agts, m, n])
    v_mat = np.copy(x_mat)
    obj_nrmse_ls = []
    diffr_nrmse_ls = []
    time_ls = []
    start_time = time.time()

    # PMACE reconstruction
    print('reg-PMACE starts ...')
    for i in range(num_iter):
        # w <- F(v; w)
        x_mat = weighted_prox_map(v_mat, probe, diffr_data, obj_nsr_pm)

        # z <- G(2w - v)
        x_cons = reg_consen_operator((2 * x_mat - v_mat) * (np.abs(probe) ** probe_exp), coords, consen_norm, init_guess.shape,
                                     reg_wgt=reg_wgt, noise_std=noise_std, prior_model=prior_model, block_idx=blk_idx)
        # v <- v + 2 \rho (z - w)
        v_mat += 2 * rho * (x_cons - x_mat)

        # phase normalization and obtain the new estimate of complex object of current iteration
        obj_est = patch2img(v_mat * (np.abs(probe) ** probe_exp), coords, init_guess.shape, consen_norm)

        # calculate time consumption
        elapsed_time = time.time() - start_time
        time_ls.append(elapsed_time)

        # compute the nrmse error between propagted reconstruction result and recorded measurements
        diffraction_est = np.abs(compute_ft(probe * img2patch(np.copy(obj_est), coords, diffr_data.shape)))
        diffr_nrmse_val = compute_nrmse(diffraction_est, diffr_data, np.ones(diffr_data.shape))
        diffr_nrmse_ls.append(diffr_nrmse_val)

        # phase normalization and compute the nrmse error between reconstructed image and reference image
        obj_revy = phase_norm(np.copy(obj_est) * cstr_win, np.copy(obj_ref) * cstr_win)
        obj_nrmse_val = compute_nrmse(obj_revy * cstr_win, obj_ref * cstr_win, cstr_win)
        obj_nrmse_ls.append(obj_nrmse_val)
        print('iter =', i, 'img_error =', obj_nrmse_val, 'diffr_err =', diffr_nrmse_val)

    # save recon results
    save_tiff(obj_est, save_dir + 'iter_{}.tiff'.format(i + 1))
    save_array(obj_nrmse_ls, save_dir + 'obj_nrmse')
    save_array(diffr_nrmse_ls, save_dir + 'diffr_nrmse')

    # return the result
    keys = ['obj_revy', 'obj_err', 'diffr_err']
    vals = [obj_revy, obj_nrmse_ls, diffr_nrmse_ls]
    output = dict(zip(keys, vals))

    return output


def pmace_joint_recon(init_obj, init_probe, diffr, projection_coords, obj_ref, probe_ref,
                      num_iter=100, obj_param=0.5, probe_param=1, rho=0.5, probe_exp=1.25, obj_exp=0.25,
                      cstr_win=None, save_dir=None):
    """
    Function to perform ptychographyic with PMACE formulation. The Mann iteration is given by:
        Initialization
        While not converged do {
            w <- F(v; w)
            z <- G(2w - v)
            v <- v + 2 \rho (z - w)
        }
        return x <- v
    :param init_obj: initialized complex object.
    :param init_probe: initialized complex probe.
    :param diffr: pre-processed phaseless measurements / diffraction patterns.
    :param projection_coords: coordinates of scan positions.
    :param obj_ref: ground truth complex image.
    :param probe_ref: ground truth complex probe.
    :param num_iter: number of iterations.
    :param obj_param: noise-to-signal ratio in the data.
    :param probe_param: noise-to-signal ratio in probe.
    :param rho: Mann averaging parameter.
    :param probe_exp: exponent of probe weighting.
    :param cstr_win: reconstruction window.
    :param save_dir: path for saving reconstructed images.
    :return: reconstructed complex image and probe and nrmse.
    """

    # check directories
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    if cstr_win is None:
        cstr_win = np.ones_like(init_obj)

    obj_est = np.asarray(init_obj, dtype=np.complex128)
    obj_mat = img2patch(obj_est, projection_coords, diffr.shape)
    obj_mat_update = np.copy(obj_mat)

    probe_est = np.asarray(init_probe, dtype=np.complex128)
    # probe_mat = np.ones(diffr.shape, dtype=np.complex128) * probe_est
    probe_mat = [probe_est] * len(diffr)
    probe_mat_update = np.copy(probe_mat)
    probe_concensus_output = np.copy(probe_mat)

    obj_nrmse_ls = []
    probe_nrmse_ls = []
    diffr_nrmse_ls = []

    start_time = time.time()
    # PMACE reconstruction
    print('PMACE joint recon starts ...')
    for i in range(num_iter):
        # w <- F(v; w)
        obj_mat = weighted_prox_map(obj_mat_update, probe_concensus_output, diffr, obj_param)
        # z <- G(2w - v)
        obj_wgt = patch2img(np.ones(diffr.shape, dtype=np.complex128) * (np.abs(probe_concensus_output) ** probe_exp), projection_coords, init_obj.shape)
        obj_concensus_output = weighted_consen_operator((2 * obj_mat - obj_mat_update) * (np.abs(probe_concensus_output) ** probe_exp), projection_coords, obj_wgt, init_obj.shape)
        # v <- v + 2 \rho (z - w)
        obj_mat_update += 2 * rho * (obj_concensus_output - obj_mat)
        # obtain current estimate of complex images
        obj_est = patch2img(obj_mat_update * (np.abs(probe_concensus_output) ** probe_exp), projection_coords, init_obj.shape, obj_wgt)

        # w <- F(v; w)
        probe_mat = weighted_prox_map(probe_mat_update, obj_concensus_output, diffr, probe_param)  # image weighted proximal map
        # z <- G(2w - v)
        # probe_concensus_output = np.average((2 * probe_mat - probe_mat_update), axis=0)
        probe_concensus_output = np.sum((2 * probe_mat - probe_mat_update) * (np.abs(obj_concensus_output) ** obj_exp), 0) / np.sum((np.abs(obj_concensus_output) ** obj_exp), 0)
        # v <- v + 2 \rho (z - w)
        probe_mat_update += 2 * rho * (probe_concensus_output - probe_mat)
        # obtain current estimate of complex images
        # probe_est = np.average(probe_mat_update, axis=0)
        probe_est = np.sum(probe_mat_update * (np.abs(obj_concensus_output) ** obj_exp), 0) / np.sum((np.abs(obj_concensus_output) ** obj_exp), 0)

        # compute the nrmse error between propagted reconstruction result and recorded measurements
        diffraction_est = np.abs(compute_ft(probe_est * img2patch(np.copy(obj_est), projection_coords, diffr.shape)))
        diffr_nrmse_val = compute_nrmse(diffraction_est, diffr)
        diffr_nrmse_ls.append(diffr_nrmse_val)

        # phase normalization and scale image to minimize the intensity difference
        if obj_ref is not None:
            obj_revy = phase_norm(np.copy(obj_est) * cstr_win, obj_ref * cstr_win)
            img_nrmse_val = compute_nrmse(obj_revy * cstr_win, obj_ref * cstr_win, cstr_win)
            obj_nrmse_ls.append(img_nrmse_val)
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
    print('Time consumption of PMACE:', elapsed_time)

    # save recon results
    save_tiff(obj_est, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
    save_tiff(probe_est, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
    save_array(obj_nrmse_ls, save_dir + 'obj_nrmse')
    save_array(probe_nrmse_ls, save_dir + 'probe_nrmse')
    save_array(diffr_nrmse_ls, save_dir + 'diffr_nrmse')

    # return result
    keys = ['obj_revy', 'probe_revy', 'obj_err', 'probe_err', 'diffr_err']
    vals = [obj_revy, probe_revy, obj_nrmse_ls, probe_nrmse_ls, diffr_nrmse_ls]
    output = dict(zip(keys, vals))

    return output


def reg_pmace_joint_recon(init_obj, init_probe, diffr, projection_coords, obj_ref, probe_ref,
                          num_iter=100, obj_param=0.5, probe_param=1, rho=0.5, probe_exp=1.25, obj_exp=0.25,
                          reg_wgt=0.1, noise_std=0.1, prior_model='bm3d', cstr_win=None, save_dir=None):
    """
    Function to perform ptychographyic with PMACE formulation + serial regularization.
    :param init_obj: initialized complex object.
    :param init_probe: initialized complex probe.
    :param diffr: pre-processed phaseless measurements / diffraction patterns.
    :param projection_coords: coordinates of scan positions.
    :param obj_ref: ground truth complex image.
    :param probe_ref: ground truth complex probe.
    :param num_iter: number of iterations.
    :param obj_param: noise-to-signal ratio in the data.
    :param probe_param: noise-to-signal ratio in probe.
    :param rho: Mann averaging parameter.
    :param probe_exp: exponent of probe weighting.
    :param obj_exp: exponent of image weighting in consensus calculation of probe estimate.
    :param reg_wgt: regularization weight.
    :param noise_std: denoising parameter.
    :param prior_model: denoising model.
    :param cstr_win: reconstruction window.
    :param save_dir: path for saving reconstructed images.
    :return: reconstructed complex image and probe and nrmse.
    """
    # check directories
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    if cstr_win is None:
        cstr_win = np.ones_like(init_obj)

    obj_est = np.asarray(init_obj, dtype=np.complex128)
    obj_mat = img2patch(obj_est, projection_coords, diffr.shape)
    obj_mat_update = np.copy(obj_mat)

    probe_est = np.asarray(init_probe, dtype=np.complex128)
    # probe_mat = np.ones(diffr.shape, dtype=np.complex128) * probe_est
    probe_mat = [probe_est] * len(diffr)
    probe_mat_update = np.copy(probe_mat)
    probe_concensus_output = np.copy(probe_mat)

    # determine the area for applying denoiser
    denoising_blk = patch2img(np.ones(diffr.shape), projection_coords, init_obj.shape)
    non_zero_idx = np.nonzero(denoising_blk)
    blk_idx = [np.amin(non_zero_idx[0]), np.amax(non_zero_idx[0])+1, np.amin(non_zero_idx[1]), np.amax(non_zero_idx[1])+1]

    obj_nrmse_ls = []
    probe_nrmse_ls = []
    diffr_nrmse_ls = []

    start_time = time.time()
    # PMACE reconstruction
    print('PMACE + serial regularization joint recon starts ...')
    for i in range(num_iter):
        # w <- F(v; w)
        obj_mat = weighted_prox_map(obj_mat_update, probe_concensus_output, diffr, obj_param)
        # z <- G(2w - v)
        obj_wgt = patch2img(np.ones(diffr.shape, dtype=np.complex128) * (np.abs(probe_concensus_output) ** probe_exp), projection_coords, init_obj.shape)
        obj_concensus_output = reg_consen_operator((2 * obj_mat - obj_mat_update) * (np.abs(probe_concensus_output) ** probe_exp),
                                                   projection_coords, obj_wgt, init_obj.shape, reg_wgt=reg_wgt,
                                                   noise_std=noise_std, prior_model=prior_model, block_idx=blk_idx)
        # v <- v + 2 \rho (z - w)
        obj_mat_update += 2 * rho * (obj_concensus_output - obj_mat)
        # obtain current estimate of complex images
        obj_est = patch2img(obj_mat_update * (np.abs(probe_concensus_output) ** probe_exp), projection_coords, init_obj.shape, obj_wgt)

        # w <- F(v; w)
        probe_mat = weighted_prox_map(probe_mat_update, obj_concensus_output, diffr, probe_param)  # image weighted proximal map
        # z <- G(2w - v)
        # probe_concensus_output = np.average((2 * probe_mat - probe_mat_update), axis=0)
        probe_concensus_output = np.sum((2 * probe_mat - probe_mat_update) * (np.abs(obj_concensus_output) ** obj_exp), 0) / np.sum((np.abs(obj_concensus_output) ** obj_exp), 0)
        # v <- v + 2 \rho (z - w)
        probe_mat_update += 2 * rho * (probe_concensus_output - probe_mat)
        # obtain current estimate of complex images
        # probe_est = np.average(probe_mat_update, axis=0)
        probe_est = np.sum(probe_mat_update * (np.abs(obj_concensus_output) ** obj_exp), 0) / np.sum((np.abs(obj_concensus_output) ** obj_exp), 0)

        # compute the nrmse error between propagted reconstruction result and recorded measurements
        diffraction_est = np.abs(compute_ft(probe_est * img2patch(np.copy(obj_est), projection_coords, diffr.shape)))
        diffr_nrmse_val = compute_nrmse(diffraction_est, diffr)
        diffr_nrmse_ls.append(diffr_nrmse_val)

        # phase normalization and scale image to minimize the intensity difference
        if obj_ref is not None:
            obj_revy = phase_norm(np.copy(obj_est) * cstr_win, obj_ref * cstr_win)
            img_nrmse_val = compute_nrmse(obj_revy * cstr_win, obj_ref * cstr_win, cstr_win)
            obj_nrmse_ls.append(img_nrmse_val)
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
    print('Time consumption of reg-PMACE:', elapsed_time)

    # save recon results
    save_tiff(obj_est, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
    save_tiff(probe_est, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
    save_array(obj_nrmse_ls, save_dir + 'obj_nrmse')
    save_array(probe_nrmse_ls, save_dir + 'probe_nrmse')
    save_array(diffr_nrmse_ls, save_dir + 'diffr_nrmse')

    # return result
    keys = ['obj_revy', 'probe_revy', 'obj_err', 'probe_err', 'diffr_err']
    vals = [obj_revy, probe_revy, obj_nrmse_ls, probe_nrmse_ls, diffr_nrmse_ls]
    output = dict(zip(keys, vals))

    return output

