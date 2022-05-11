from ptycho_pmace.utils.utils import *
from ptycho_pmace.utils.nrmse import *


def wf_obj_func(current_est, probe_ref, diffr_data, coords, discretized_sys_mat, param=1):
    """
    Function to determine the gradient descent updates of complex object estimate.
    :param current_est: the current estimate of complex object.
    :param probe_ref: known complex probe.
    :param diffr_data: pre-processed diffraction pattern (intensity data).
    :param coords: coordinates of projections.
    :param discretized_sys_mat: to calculate the step size.
    :param param: hyper-parameter, param=1 if Fourier transform is orthonormal.
    :return: updated estimate of complex object.
    """
    # Ax = FT{D*P_j*v}
    projected_patch = img2patch(current_est, coords, diffr_data.shape)
    # Ax = FT{D*P_j*v}
    spectrum = compute_ft(probe_ref * projected_patch)
    # Ax - y * Ax / |Ax| = - FT{D * P_j * v} - y * FT{D * P_j * v} / |FT{D * P_j * v}|
    spectrum_update = spectrum - diffr_data * np.exp(1j * np.angle(spectrum))
    # A^(H){```} = P^t D^* IFFT{```}
    patch_update = compute_ift(spectrum_update)
    # back projection
    img_update = patch2img(np.conj(probe_ref) * patch_update, coords, current_est.shape)
    # step_sz = 1/biggest eigenvalue of semi positive deifnite matrix =1/\lambda_max(A^(H)A)=1/(alpha*sum_j |D_j|^2)
    output = current_est - img_update / np.amax(param * discretized_sys_mat)

    return output


def wf_probe_func(current_est, projected_img, diffr_data, discretized_sys_mat, param=1):
    """
    Function to determine the gradient descent update of complex probe estimate.
    :param current_est: current estimate of complex probe.
    :param projected_img: projected image patch of current estimate of complex object.
    :param diffr_data: pre-processed diffraction pattern (intensity data).
    :param discretized_sys_mat: to calculate step size.
    :param param: hypter-param, param=1 if Fourier transform is orthonormal.
    :return: new estimate of complex probe.
    """

    # Bd = FT{D*X_j*d}
    freq = compute_ft(current_est * projected_img)
    # Bd - y * Bd / |Bd| =  FT{D*X_j*d} - y * FT{D*X_j*d} / |FT{D*X_j*d}|
    freq_update = freq - diffr_data * np.exp(1j * np.angle(freq))
    # B^(H){```} = (X_j)^(H) IFFT{```}
    probe_mat = np.conj(projected_img) * compute_ift(freq_update)
    # step_sz = 1/biggest eigenvalue of semi positive deifnite matrix =1/\lambda_max(A^(H)A)=1/(alpha*sum_j |D_j|^2)
    output = current_est - probe_mat / np.amax(param * discretized_sys_mat)

    return output


def wf_recon(dp, project_coords, init_obj, init_probe=None, obj_ref=None, probe_ref=None,
             accel=True, num_iter=100, joint_recon=False, cstr_win=None, save_dir=None):
    """
    Function to perform WF/AWF reconstruction on ptychographic data.
    :param dp: pre-processed diffraction pattern (intensity data).
    :param project_coords: scan coordinates of projections.
    :param init_obj: formulated initial guess of complex object.
    :param init_probe: formulated initial guess of complex probe.
    :param obj_ref: complex reference image.
    :param probe_ref: complex reference image.
    :param accel: with or without Nesterov acceleration.
    :param num_iter: number of iterations.
    :param joint_recon: option to recover complex probe for blind ptychography.
    :param cstr_win: pre-defined/assigned window for comparing reconstruction results.
    :param save_dir: directory to save reconstruction results.
    :return: reconstructed images and error metrics.
    """
    # check directory
    approach = 'AWF' if accel else 'WF'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # initialization
    if cstr_win is None:
        cstr_win = np.ones_like(init_obj)

    obj_est = np.copy(init_obj)
    obj_old_est = np.copy(init_obj)
    probe_est = np.copy(init_probe) if joint_recon else np.copy(probe_ref)
    probe_old_est = np.copy(init_probe) if joint_recon else np.copy(probe_ref)

    obj_nrmse_ls = []
    probe_nrmse_ls = []
    dp_nrmse_ls = []
    time_ls = []

    # WF reconstruction
    print('{} recon starts ...'.format(approach))
    start_time = time.time()
    for i in range(num_iter):
        if accel:
            beta = (i + 2) / (i + 4)
        else:
            beta = 0

        # update estimate of complex object
        probe_mat = [probe_est] * len(dp)
        obj_step_sz_mat = patch2img(np.abs(probe_mat) ** 2, project_coords, init_obj.shape)

        current_obj_est = obj_est + beta * (obj_est - obj_old_est)
        obj_old_est = np.copy(obj_est)
        obj_est = wf_obj_func(current_obj_est, probe_est, dp, project_coords, obj_step_sz_mat)

        # update estimate of complex probe
        if joint_recon:
            projected_img_patch = img2patch(obj_est, project_coords, dp.shape)
            probe_step_sz_mat = np.sum(np.abs(projected_img_patch) ** 2, 0)

            current_probe_est = probe_est + beta * (probe_est - probe_old_est)
            probe_old_est = np.copy(probe_est)
            probe_mat_est = wf_probe_func(current_probe_est, projected_img_patch, dp, probe_step_sz_mat)
            probe_est = np.average(probe_mat_est, axis=0)

        # calculate time consumption
        elapsed_time = time.time() - start_time
        time_ls.append(elapsed_time)

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
