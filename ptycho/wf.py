from utils.utils import *
from utils.nrmse import *
from utils.display import *


def wf(current_est, probe_ref, diffr_data, coords, norm, param=1):
    """
    Function to determine the gradient descent updates.
    :param current_est: the current estimate of complex object.
    :param probe_ref: known or estimated complex probe.
    :param diffr_data: pre-processed diffraction patterns (phaseless measurements).
    :param coords: coordinates of projections.
    :param norm: weights.
    :param param: 1 by default (assuming the FT is orthonormal).
    :return: new esimate of complex object.
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
    output = current_est - img_update / np.amax(param * norm)

    return output


def wf_recon(init_obj_guess, diffract_data, projection_coords, obj_ref, probe_ref,
             num_iter=100, param=1, accel=True, display_win=None, display=False, save_dir=None):
    """
    WF/AWF reconstruction to perform single estimate on complex transmittance of complex
    object assuming known complex probe function.
    :param init_obj_guess: formulated initial guess of complex transmittance of complex object image.
    :param diffract_data: pre-processed diffraction patterns (phase-less measurements).
    :param projection_coords: scan coordinates of projections.
    :param obj_ref: ground truth image or reference image.
    :param probe_ref: known or estimated complex probe function.
    :param num_iter: number of iterations.
    :param param: parameter is 1 by default.
    :param param: with Nesterov Acceleration, AWF is performed.
    :param display_win: pre-defined cover/window for comparing reconstruction results.
    :param display: option to display the reconstruction results.
    :param save_dir: save reconstruction results to the given directory.
    :return: reconstructed complex transmittance of complex object and error metrics.
    """
    # check directories for saving files
    if accel:
        approach = 'AWF'
    else:
        approach = 'WF'

    if save_dir is not None:
        obj_fname = save_dir + '{}_img_revy'.format(approach)
        os.makedirs(save_dir, exist_ok=True)

    if display_win is None:
        display_win = np.ones(init_obj_guess.shape)

    # initialization
    probe = np.copy(probe_ref)
    img_norm = patch2img([np.abs(probe)**2] * len(diffract_data), projection_coords, init_obj_guess.shape)
    obj_est = np.copy(init_obj_guess)
    old_est = np.copy(init_obj_guess)
    obj_nrmse_ls = []
    diffr_nrmse_ls = []
    time_ls = []
    start_time = time.time()

    # WF/AWF reconstruction
    print('{} starts ...'.format(approach))
    for i in range(num_iter):
        if accel:
            beta = (i + 2) / (i + 4)
            # beta = (i + 2) / (i + 10)
        else:
            beta = 0
        current_est = obj_est + beta * (obj_est - old_est)
        old_est = np.copy(obj_est)
        obj_est = wf(current_est, probe, diffract_data, projection_coords, img_norm, param=param)

        # compute time consumption
        elapsed_time = time.time() - start_time
        time_ls.append(elapsed_time)

        # compute time consumption
        elapsed_time = time.time() - start_time
        time_ls.append(elapsed_time)

        # compute the nrmse error between propagted reconstruction result and recorded measurements
        diffraction_est = np.abs(
            compute_ft(probe * img2patch(np.copy(obj_est), projection_coords, diffraction_data.shape)))
        diffr_nrmse_val = compute_nrmse(diffraction_est, diffraction_data, np.ones(diffraction_data.shape))
        diffr_nrmse_ls.append(diffr_nrmse_val)

        # phase normalization and compute the nrmse error between reconstructed image and reference image
        obj_revy = phase_norm(np.copy(obj_est) * display_win, np.copy(obj_ref) * display_win)
        obj_nrmse_val = compute_nrmse(obj_revy * display_win, obj_ref * display_win, display_win)
        obj_nrmse_ls.append(obj_nrmse_val)
        print('iter =', i, 'img_error =', obj_nrmse_val, 'diffr_err =', diffr_nrmse_val)

    # save recon results
    save_tiff(obj_est, save_dir + 'iter_{}.tiff'.format(i + 1))
    save_array(obj_nrmse_ls, save_dir + 'obj_nrmse')
    save_array(diffr_nrmse_ls, save_dir + 'diffr_nrmse')

    # display the reconstructed image and convergence plot

    plot_cmplx_obj(obj_revy, obj_ref, img_title='{}'.format(approach), display_win=display_win,
                   display=display, save_fname=obj_fname)
    xlabel, ylabel, line_label = 'Number of iterations', 'object NRMSE', r'{}'.format(approach)
    plot_nrmse(obj_nrmse_ls, title='Convergence plot of {} algorithm'.format(approach),
               label=[xlabel, ylabel, line_label],
               step_sz=15, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_obj_nrmse')
    xlabel, ylabel, line_label = 'Number of iterations', 'diffraction NRMSE', r'{} '.format(approach)
    plot_nrmse(diffr_nrmse_ls, title='Convergence plot of {} algorithm'.format(approach),
               label=[xlabel, ylabel, line_label],
               step_sz=15, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_diffr_nrmse')

    # return the result
    keys = ['obj_revy', 'obj_err', 'diffr_err']
    vals = [obj_revy, obj_nrmse_ls, diffr_nrmse_ls]
    output = dict(zip(keys, vals))

    return output


def wf_obj_func(current_est, probe_ref, diffr_data, coords, discretized_sys_mat, param=1):
    """
    Function to determine the gradient descent updates of complex object estimate.
    :param current_est: the current estimate of complex object.
    :param probe_ref: known complex probe.
    :param diffr_data: phaseless measurements.
    :param coords: coordinates of projections.
    :param discretized_sys_mat: to calculate the step size.
    :param param: hyper-parameter, param=1 if Fourier transform is orthonormal.
    :return: updated esimate of complex object.
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
    :param diffr_data: phaseless measurements.
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


def wf_joint_recon(init_obj, init_probe, diffraction_data, projection_coords, obj_ref=None, probe_ref=None,
                   num_iter=100, obj_pm=1, probe_pm=1, accel=False, display_win=None, display=False, save_dir=None):
    """
    Function to perform Wirtinger Flow (WF) algorithm and Accelerated Wirtinger Flow (AWF) reconstruction.
    :param init_obj: initialized complex object.
    :param init_probe: initialized complex probe.
    :param diffraction_data: pre-processed phaseless measurements / diffraction patterns.
    :param projection_coords: coordinates of scan positions.
    :param obj_ref: ground truth complex object.
    :param probe_ref: ground truth complex probe.
    :param num_iter: number of iterations.
    :param obj_pm: scaler to adjust the step-size of object update function.
    :param probe_pm: scaler to adjust the step-size of probe update function.
    :param accel: with or without acceleration.
    :param display_win: reconstruction window.
    :param display: display the reconstruction results.
    :param save_dir: path for saving reconstructed images.
    :return: reconstructed complex image and probe and nrmse.
    """
    # determine approach
    if accel:
        approach_name = 'AWF'
    else:
        approach_name = 'WF'

    # check directories for saving files
    if save_dir is None:
        obj_fname = None
        probe_fname = None
    else:
        obj_fname = save_dir + '{}_img_revy'.format(approach_name)
        probe_fname = save_dir + '{}_probe_revy'.format(approach_name)
        os.makedirs(save_dir, exist_ok=True)

    if display_win is None:
        display_win = np.ones(init_obj.shape)

    # initialization
    obj_nrmse_ls = []
    probe_nrmse_ls = []
    diffr_nrmse_ls = []

    obj_est = np.copy(init_obj)
    obj_old_est = np.copy(init_obj)

    probe_est = np.copy(init_probe)
    probe_old_est = np.copy(init_probe)

    start_time = time.time()
    # WF reconstruction
    print('{} joint recon starts ...'.format(approach_name))
    for i in range(num_iter):
        if accel:
            beta = (i + 2) / (i + 4)
        else:
            beta = 0

        # update estimate of complex object
        # probe_mat = np.ones(diffraction_data.shape, dtype=np.complex128) * probe_est
        probe_mat = [probe_est] * len(diffraction_data)
        obj_step_sz_mat = patch2img(np.abs(probe_mat) ** 2, projection_coords, init_obj.shape)

        current_obj_est = obj_est + beta * (obj_est - obj_old_est)
        obj_old_est = np.copy(obj_est)
        obj_est = wf_obj_func(current_obj_est, probe_est, diffraction_data, projection_coords, obj_step_sz_mat, param=obj_pm)

        # update estimate of complex probe
        projected_img_patch = img2patch(obj_est, projection_coords, diffraction_data.shape)
        probe_step_sz_mat = np.sum(np.abs(projected_img_patch) ** 2, 0)

        current_probe_est = probe_est + beta * (probe_est - probe_old_est)
        probe_old_est = np.copy(probe_est)
        probe_mat_est = wf_probe_func(current_probe_est, projected_img_patch, diffraction_data, probe_step_sz_mat, param=probe_pm)
        probe_est = np.average(probe_mat_est, axis=0)

        # compute the nrmse error between propagted reconstruction result and recorded measurements
        diffraction_est = np.abs(compute_ft(probe_est * img2patch(np.copy(obj_est), projection_coords, diffraction_data.shape)))
        diffr_nrmse_val = compute_nrmse(diffraction_est, diffraction_data)
        diffr_nrmse_ls.append(diffr_nrmse_val)

        # phase normalization and scale image to minimize the intensity difference
        if obj_ref is not None:
            obj_revy = phase_norm(np.copy(obj_est) * display_win, obj_ref * display_win)
        else:
            obj_revy = obj_est
            obj_ref = obj_revy
        if probe_ref is not None:
            probe_revy = phase_norm(np.copy(probe_est), probe_ref)
        else:
            probe_revy = probe_est
            probe_ref = probe_revy

        # compute the nrmse error
        img_nrmse_val = compute_nrmse(obj_revy * display_win, obj_ref * display_win, display_win)
        obj_nrmse_ls.append(img_nrmse_val)
        probe_nrmse_val = compute_nrmse(probe_revy, probe_ref, np.ones(probe_revy.shape))
        probe_nrmse_ls.append(probe_nrmse_val)
        print('iter =', i, 'img_nrmse =', img_nrmse_val, 'probe_nrmse =', probe_nrmse_val, 'diffr_err =', diffr_nrmse_val)

    # calculate time consumption
    elapsed_time = time.time() - start_time
    print('Time consumption of ePIE:', elapsed_time)

    # save recon results
    save_tiff(obj_est, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
    save_tiff(probe_est, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
    save_array(obj_nrmse_ls, save_dir + 'obj_nrmse')
    save_array(probe_nrmse_ls, save_dir + 'probe_nrmse')
    save_array(diffr_nrmse_ls, save_dir + 'diffr_nrmse')

    # display the reconstructed image and probe
    plot_cmplx_obj(obj_revy, obj_ref, img_title='{}'.format(approach_name), display_win=display_win, display=display, save_fname=obj_fname)
    plot_cmplx_probe(probe_revy, probe_ref, img_title='{}'.format(approach_name), display=display, save_fname=probe_fname)

    # return the result
    keys = ['obj_revy', 'probe_revy', 'obj_err', 'probe_err', 'diffr_err']
    vals = [obj_revy, probe_revy, obj_nrmse_ls, probe_nrmse_ls, diffr_nrmse_ls]
    output = dict(zip(keys, vals))

    return output
