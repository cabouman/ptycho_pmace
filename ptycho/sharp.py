from ptycho_pmace.utils.utils import *


def fourier_projector(frame_data, diffr_data):
    """
    This phase projector projects the frames onto the Fourier magnitude constraint.
                   P_a z = [P_a1 z_1, P_a2 z_2, ..., P_aJ z_J]
    where
    P_aj z_j = F^* {y_j * F z_j / |F z_j|}.
    :param frame_data: image patches multiplied by beam profile function at each scan position, i.e. D*P_j*x.
    :param diffr_data: pre-processed diffraction patterns (recorded phase-less measurements).
    :return: revised estimates of frames.
    """
    # FT{D*P_j*v}]
    spectrum = compute_ft(frame_data)
    # y \times FT{D*P_j*v}] / |FT{D*P_j*v}]|
    spectrum_update = divide_cmplx_numbers(diffr_data * np.copy(spectrum), np.abs(spectrum))
    # spectrum_update = diffr_data * np.copy(spectrum) / np.abs(spectrum)
    # Take inverse FT
    output = compute_ift(spectrum_update)

    return output


def space_projector(frame_data, probe, coords, norm, img_sz):
    """
    The image projector matches the object with object domain constraint.
                           P_Q = Q (Q^* Q)^(-1) Q^*
    Or equivalently
                            P_Q = Q Lambda^(-1) Q^*
    where
    Lambda = \sum_j P_j^t D^* D P_j.
    :param frame_data: the extracted frames z_j = D P_j x.
    :param probe: the beam profile function.
    :param coords: coordinates of projections.
    :param norm: \sum P_j^t D^* D P_j.
    :param img_sz: the shape of full-size image.
    :return: revised estimates of frames.
    """
    wgt_img = patch2img(frame_data * np.conj(probe), coords, img_sz, norm)
    output = img2patch(wgt_img, coords, frame_data.shape) * probe

    return output


def fourier_relected_resolvent(frames, diffr):

    output = 2 * fourier_projector(frames, diffr) - frames

    return output


def space_relected_resolvent(frames, probe, coords, norm, img_sz):

    output = 2 * space_projector(frames, probe, coords, norm, img_sz) - frames

    return output


def sharp_recon(dp, project_coords, init_obj, init_probe=None, obj_ref=None, probe_ref=None,
                num_iter=100, relax_pm=0.6, joint_recon=False, cstr_win=None, save_dir=None):
    """
    Function to perform SHARP reconstruction on ptychographic data. SHARP is introduced in:
    https://doi.org/10.1107/S1600576716008074.
    :param dp: pre-processed diffraction pattern (intensity data).
    :param project_coords: scan coordinates of projections.
    :param init_obj: formulated initial guess of complex object.
    :param init_probe: formulated initial guess of complex probe.
    :param obj_ref: complex reference image.
    :param probe_ref: complex reference image.
    :param num_iter: number of iterations.
    :param relax_pm: relaxation parameter.
    :param joint_recon: option to recover complex probe for blind ptychography.
    :param cstr_win: pre-defined/assigned window for comparing reconstruction results.
    :param save_dir: directory to save reconstruction results.
    :return: reconstructed images and error metrics.
    """
    # check directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # initialization
    if cstr_win is None:
        cstr_win = np.ones_like(init_obj)

    obj_est = np.copy(init_obj)
    projected_img_mat = img2patch(init_obj, project_coords, dp.shape)
    probe_est = np.copy(init_probe) if joint_recon else np.copy(probe_ref)
    frm_est = np.copy(projected_img_mat) * probe_est
    current_frm_est = np.copy(frm_est)

    obj_nrmse_ls = []
    probe_nrmse_ls = []
    dp_nrmse_ls = []

    # SHARP reconstruction
    print('SHARP recon starts ...')
    start_time = time.time()
    for i in range(num_iter):
        # calculate spatially-varying weights
        probe_mat = [probe_est] * len(dp)
        img_wgt = patch2img(np.abs(probe_mat) ** 2, project_coords, init_obj.shape)
        # take projections
        proj_frm_fourier = fourier_projector(current_frm_est, dp)
        proj_frm_space = space_projector(current_frm_est, probe_est, project_coords, img_wgt, init_obj.shape)
        # SHARP updates
        frm_est = 2 * relax_pm * space_projector(proj_frm_fourier, probe_est, project_coords, img_wgt, init_obj.shape) + \
                  (1 - 2 * relax_pm) * proj_frm_fourier + relax_pm * (proj_frm_space - current_frm_est)
        # update current frame data estimate
        current_frm_est = np.copy(frm_est)

        # obtain the current estimate of complex object
        obj_est = patch2img(frm_est * np.conj(probe_est), project_coords, init_obj.shape, img_wgt)
        # obtain the current estimate of complex probe
        if joint_recon:
            temp_num = np.average(img2patch(np.conj(obj_est), project_coords, dp.shape) * frm_est, axis=0)
            temp_denom = np.average(img2patch(np.abs(obj_est) ** 2, project_coords, dp.shape), axis=0)
            probe_est = divide_cmplx_numbers(temp_num, temp_denom)

        # # dynamic strategy for updating beta
        # beta = beta + (1 - beta) * (1 - np.exp(-(i/7)**3))

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
    print('Time consumption of SHARP:', elapsed_time)

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


def sharp_plus_recon(dp, project_coords, init_obj, init_probe=None, obj_ref=None, probe_ref=None,
                     num_iter=100, relax_pm=0.6, joint_recon=False, cstr_win=None, save_dir=None):
    """
    Function to perform SHARP+ reconstruction on ptychographic data. SHARP+ is introduced in:
    https://arxiv.org/abs/2111.14240.
    :param dp: pre-processed diffraction pattern (intensity data).
    :param project_coords: scan coordinates of projections.
    :param init_obj: formulated initial guess of complex object.
    :param init_probe: formulated initial guess of complex probe.
    :param obj_ref: complex reference image.
    :param probe_ref: complex reference image.
    :param num_iter: number of iterations.
    :param relax_pm: relaxation parameter.
    :param joint_recon: option to recover complex probe for blind ptychography.
    :param cstr_win: pre-defined/assigned window for comparing reconstruction results.
    :param save_dir: directory to save reconstruction results.
    :return: reconstructed images and error metrics.
    """
    # check directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # initialization
    if cstr_win is None:
        cstr_win = np.ones_like(init_obj)

    obj_est = np.copy(init_obj)
    projected_img_mat = img2patch(init_obj, project_coords, dp.shape)
    probe_est = np.copy(init_probe) if joint_recon else np.copy(probe_ref)
    frm_est = np.copy(projected_img_mat) * probe_est
    current_frm_est = np.copy(frm_est)

    obj_nrmse_ls = []
    probe_nrmse_ls = []
    dp_nrmse_ls = []

    # SHARP reconstruction
    print('SHARP recon starts ...')
    start_time = time.time()
    for i in range(num_iter):
        # calculate spatially-varying weights
        probe_mat = [probe_est] * len(dp)
        img_wgt = patch2img(np.abs(probe_mat) ** 2, project_coords, init_obj.shape)
        # take projections
        proj_frm_fourier = fourier_projector(current_frm_est, dp)
        proj_frm_space = space_projector(current_frm_est, probe_est, project_coords, img_wgt, init_obj.shape)
        # SHARP updates
        frm_est = 2 * relax_pm * space_projector(proj_frm_fourier, probe_est, project_coords, img_wgt, init_obj.shape) + \
                  (1 - 2 * relax_pm) * proj_frm_fourier - relax_pm * (proj_frm_space - current_frm_est)
        # update current frame data estimate
        current_frm_est = np.copy(frm_est)

        # obtain the current estimate of complex object
        obj_est = patch2img(frm_est * np.conj(probe_est), project_coords, init_obj.shape, img_wgt)
        # obtain the current estimate of complex probe
        if joint_recon:
            temp_num = np.average(img2patch(np.conj(obj_est), project_coords, dp.shape) * frm_est, axis=0)
            temp_denom = np.average(img2patch(np.abs(obj_est) ** 2, project_coords, dp.shape), axis=0)
            probe_est = divide_cmplx_numbers(temp_num, temp_denom)

        # # dynamic strategy for updating beta
        # beta = beta + (1 - beta) * (1 - np.exp(-(i/7)**3))

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
    print('Time consumption of SHARP+:', elapsed_time)

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
