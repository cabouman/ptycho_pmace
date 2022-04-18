from utils.utils import *
from utils.nrmse import *
from utils.display import *


def fourier_projector(frame_data, diffr_data):
    """
    This phase projector projects the frames onto the Fourier magnitude constraint.
                   P_a z = [P_a1 z_1, P_a2 z_2, ..., P_aJ z_J]
    where
    P_aj z_j = F^* {y_j * F z_j / |F z_j|}.
    :param frame_data: image patches multiplied by beam profile function at each scan position, i.e. D*P_j*x.
    :param diffr_data: pre-processed diffraction patterns (recorded phase-less measurements).
    :return: new estimates of frames.
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
    :param norm: \sum P_j^t D^* D P_j
    :param img_sz: the shape of full-size image.
    :return: new estimates of frames.
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


def sharp_recon(init_obj_guess, diffraction_data, projection_coords, obj_ref, probe_ref,
                num_iter=100, relax_pm=0.75, display_win=None, display=False, save_dir=None):
    """
    SHARP reconstruction to perform single estimate on complex transmittance of complex object assuming
    known complex probe function. The SHARP is introduced in: https://doi.org/10.1107/S1600576716008074.
    :param init_obj_guess: formulated initial guess of complex transmittance of complex object image.
    :param diffraction_data: pre-processed diffraction patterns (phase-less measurements).
    :param projection_coords: scan coordinates of projections.
    :param obj_ref: ground truth image or reference image.
    :param probe_ref: known or estimated complex probe function.
    :param num_iter: number of iterations.
    :param relax_pm: relaxation parameter inherited from RAAR algorithm.
    :param display_win: pre-defined cover/window for comparing reconstruction results.
    :param display: option to display the reconstruction results.
    :param save_dir: save reconstruction results to the given directory.
    :return: reconstructed complex transmittance of complex object and error metrics.
    """
    # check directories for saving files
    if save_dir is not None:
        obj_fname = save_dir + 'SHARP_img_revy'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if display_win is None:
        display_win = np.ones(init_obj_guess.shape)

    # initialization
    probe = np.copy(probe_ref)
    img_norm = patch2img(np.ones(diffraction_data.shape) * (np.abs(probe) ** 2), projection_coords, init_obj_guess.shape)
    projected_patch = img2patch(init_obj_guess, projection_coords, diffraction_data.shape)
    frame_data = np.copy(projected_patch) * probe
    frame_update = np.copy(frame_data)
    obj_nrmse_ls = []
    diffr_nrmse_ls = []
    time_ls = []
    start_time = time.time()

    # SHARP reconstruction
    print('SHARP starts ...')
    for i in range(num_iter):
        frame_data = frame_update
        # # # Error reduction
        # frame_update = space_projector(fourier_projector(frame_data, diffraction_data), probe, projection_coords, img_norm, init_obj_guess.shape)
        # SHARP
        temp_fourier = fourier_projector(frame_data, diffraction_data)
        temp_space = space_projector(frame_data, probe, projection_coords, img_norm, init_obj_guess.shape)
        frame_update = 2 * relax_pm * space_projector(temp_fourier, probe, projection_coords, img_norm, init_obj_guess.shape) + \
                        (1 - 2 * relax_pm) * temp_fourier + relax_pm * (temp_space - frame_data)

        # obtain the new estimate of complex object of current iteration
        obj_est = patch2img(frame_update * np.conj(probe), projection_coords, init_obj_guess.shape, img_norm)

        # # dynamic strategy for updating beta
        # beta = beta + (1 - beta) * (1 - np.exp(-(i/7)**3))

        # compute the time consumption
        elapsed_time = time.time() - start_time
        time_ls.append(elapsed_time)

        # compute the nrmse error between propagted reconstruction result and recorded measurements
        diffraction_est = np.abs(compute_ft(probe * img2patch(np.copy(obj_est), projection_coords, diffraction_data.shape)))
        diffr_nrmse_val = compute_nrmse(diffraction_est, diffraction_data, np.ones(diffraction_data.shape))
        diffr_nrmse_ls.append(diffr_nrmse_val)

        # phase normalization and compute the nrmse error between reconstructed image and reference image
        obj_revy = phase_norm(np.copy(obj_est) * display_win, np.copy(obj_ref) * display_win)
        obj_nrmse_val = compute_nrmse(obj_revy * display_win, obj_ref * display_win, display_win)
        obj_nrmse_ls.append(obj_nrmse_val)
        print('iter =', i, 'img_error =', obj_nrmse_val, 'diffr_err =', diffr_nrmse_val)

    # save recon results
    save_tiff(obj_est, save_dir + 'iter_{}.tiff'.format(i + 1))
    save_array(obj_nrmse_ls,  save_dir + 'obj_nrmse')
    save_array(diffr_nrmse_ls, save_dir + 'diffr_nrmse')

    # display the reconstructed image and convergence plot
    plot_cmplx_obj(obj_revy, obj_ref, img_title='SHARP', display_win=display_win, display=display, save_fname=obj_fname)
    xlabel, line_label = 'Number of iterations', r'SHARP ($relax param$ = {})'.format(relax_pm)
    plot_nrmse(obj_nrmse_ls, title='Convergence plot of SHARP', label=[xlabel, 'object NRMSE', line_label],
               step_sz=15, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_obj_nrmse')
    plot_nrmse(diffr_nrmse_ls, title='Convergence plot of SHARP', label=[xlabel, 'diffraction NRMSE', line_label],
               step_sz=15, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_diffr_nrmse')

    # return the result
    keys = ['obj_revy', 'obj_err', 'diffr_err']
    vals = [obj_revy, obj_nrmse_ls, diffr_nrmse_ls]
    output = dict(zip(keys, vals))

    return output


def sharp_plus_recon(init_obj_guess, diffraction_data, projection_coords, obj_ref, probe_ref,
                     num_iter=100, relax_pm=0.75, display_win=None, display=False, save_dir=None):
    """
    SHARP+ reconstruction to perform single estimate on complex transmittance of complex object assuming
    known complex probe function. The SHARP+ is introduced in: https://arxiv.org/abs/2111.14240.
    :param init_obj_guess: formulated initial guess of complex transmittance of complex object image.
    :param diffraction_data: pre-processed diffraction patterns (phase-less measurements).
    :param projection_coords: scan coordinates of projections.
    :param obj_ref: ground truth image or reference image.
    :param probe_ref: known or estimated complex probe function.
    :param num_iter: number of iterations.
    :param relax_pm: relaxation parameter inherited from RAAR algorithm.
    :param display_win: pre-defined cover/window for comparing reconstruction results.
    :param display: option to display the reconstruction results.
    :param save_dir: save reconstruction results to the given directory.
    :return: reconstructed complex transmittance of complex object and error metrics.
    """
    # check directories for saving files
    if save_dir is not None:
        obj_fname = save_dir + 'sharp_plus_img_revy'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if display_win is None:
        display_win = np.ones(init_obj_guess.shape)

    # initialization
    probe = np.copy(probe_ref)
    img_norm = patch2img(np.ones(diffraction_data.shape) * (np.abs(probe) ** 2), projection_coords, init_obj_guess.shape)
    projected_patch = img2patch(init_obj_guess, projection_coords, diffraction_data.shape)
    frame_data = np.copy(projected_patch) * probe
    frame_update = np.copy(frame_data)
    obj_nrmse_ls = []
    diffr_nrmse_ls = []
    time_ls = []
    start_time = time.time()

    # SHARP reconstruction
    print('SHARP+ starts ...')
    for i in range(num_iter):
        frame_data = frame_update
        # # # Error reduction
        # frame_update = space_projector(fourier_projector(frame_data, diffraction_data), probe, projection_coords, img_norm, init_obj_guess.shape)
        # SHARP
        temp_fourier = fourier_projector(frame_data, diffraction_data)
        temp_space = space_projector(frame_data, probe, projection_coords, img_norm, init_obj_guess.shape)
        frame_update = 2 * relax_pm * space_projector(temp_fourier, probe, projection_coords, img_norm, init_obj_guess.shape) + \
                        (1 - 2 * relax_pm) * temp_fourier - relax_pm * (temp_space - frame_data)

        # obtain the new estimate of complex object of current iteration
        obj_est = patch2img(frame_update * np.conj(probe), projection_coords, init_obj_guess.shape, img_norm)

        # # dynamic strategy for updating beta
        # beta = beta + (1 - beta) * (1 - np.exp(-(i/7)**3))

        # compute the time consumption
        elapsed_time = time.time() - start_time
        time_ls.append(elapsed_time)

        # compute the nrmse error between propagted reconstruction result and recorded measurements
        diffraction_est = np.abs(compute_ft(probe * img2patch(np.copy(obj_est), projection_coords, diffraction_data.shape)))
        diffr_nrmse_val = compute_nrmse(diffraction_est, diffraction_data, np.ones(diffraction_data.shape))
        diffr_nrmse_ls.append(diffr_nrmse_val)

        # phase normalization and compute the nrmse error between reconstructed image and reference image
        obj_revy = phase_norm(np.copy(obj_est) * display_win, np.copy(obj_ref) * display_win)
        obj_nrmse_val = compute_nrmse(obj_revy * display_win, obj_ref * display_win, display_win)
        obj_nrmse_ls.append(obj_nrmse_val)
        print('iter =', i, 'img_error =', obj_nrmse_val, 'diffr_err =', diffr_nrmse_val)

    # save recon results
    save_tiff(obj_est, save_dir + 'iter_{}.tiff'.format(i + 1))
    save_array(obj_nrmse_ls,  save_dir + 'obj_nrmse')
    save_array(diffr_nrmse_ls, save_dir + 'diffr_nrmse')

    # display the reconstructed image and convergence plot
    plot_cmplx_obj(obj_revy, obj_ref, img_title='SHARP+', display_win=display_win, display=display, save_fname=obj_fname)
    xlabel, ylabel, line_label = 'Number of iterations', 'object NRMSE', r'SHARP+ ($relax param$ = {})'.format(relax_pm)
    plot_nrmse(obj_nrmse_ls, title='Convergence plot of SHARP+ algorithm', label=[xlabel, ylabel, line_label],
               step_sz=15, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_obj_nrmse')
    plot_nrmse(diffr_nrmse_ls, title='Convergence plot of SHARP+ algorithm', label=[xlabel, 'diffraction NRMSE', line_label],
               step_sz=15, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_diffr_nrmse')

    # return the result
    keys = ['obj_revy', 'obj_err', 'diffr_err']
    vals = [obj_revy, obj_nrmse_ls, diffr_nrmse_ls]
    output = dict(zip(keys, vals))

    return output


def sharp_joint_recon(init_obj, init_probe, diffraction_data, projection_coords, obj_ref=None, probe_ref=None,
                      num_iter=100, relax_pm=0.5, display_win=None, display=False, save_dir=None):
    """
    The SHARP is introduced in: https://doi.org/10.1107/S1600576716008074. The frame estimate update function:
        $z_j \gets 2 \beta P_{Q,j} P_{a,j} +(1 - 2 \beta) P_{a,j} + \beta ( P_{Q,j} - I)  z_j (*)$
    :param init_obj: initial guess of complex image.
    :param init_probe: initial guess of probe function.
    :param diffraction_data: pre-processed phaseless measurements / diffraction patterns.
    :param projection_coords: coordinates of scan positions.
    :param obj_ref: ground truth complex image.
    :param probe_ref: ground truth complex probe.
    :param num_iter: number of iterations.
    :param relax_pm: relaxation parameter.
    :param display_win: reconstruction window.
    :param display: display the reconstruction results.
    :param save_dir: path for saving reconstructed images.
    :return: reconstructed complex image and probe and nrmse.
    """
    # check directories for saving files
    if save_dir is None:
        obj_fname = None
        probe_fname = None
    else:
        obj_fname = save_dir + 'SHARP_img_revy'
        probe_fname = save_dir + 'SHARP_probe_revy'
        os.makedirs(save_dir, exist_ok=True)

    if display_win is None:
        display_win = np.ones(init_obj.shape)

    # initialization
    obj_nrmse_ls = []
    probe_nrmse_ls = []
    diffr_nrmse_ls = []

    obj_est = np.copy(init_obj)
    projected_img_mat = img2patch(init_obj, projection_coords, diffraction_data.shape)

    probe_est = np.copy(init_probe)
    current_probe_est = np.copy(probe_est)

    frm_est = np.copy(projected_img_mat) * init_probe
    current_frm_est = np.copy(frm_est)

    full_img_sz = init_obj.shape
    start_time = time.time()

    # SHARP reconstruction
    print('SHARP joint recon starts ...')
    for i in range(num_iter):
        # calculate spatially-varying weights
        probe_mat = np.ones(diffraction_data.shape, dtype=np.complex128) * current_probe_est
        img_wgt = patch2img(np.abs(probe_mat) ** 2, projection_coords, full_img_sz) # + 1e-15

        proj_frm_fourier = fourier_projector(current_frm_est, diffraction_data)
        proj_frm_space = space_projector(current_frm_est, current_probe_est, projection_coords, img_wgt, full_img_sz)

        # # SHARP
        # # frm_est = 2 * relax_pm * space_projector(fourier_projector(current_frm_est, diffr), current_probe_est, scan_coords, img_wgt, full_img_sz) + \
        # #           (1 - 2 * relax_pm) * fourier_projector(current_frm_est, diffr) + \
        # #           relax_pm * (space_projector(current_frm_est, current_probe_est, scan_coords, img_wgt, full_img_sz) - current_frm_est)
        frm_est = 2 * relax_pm * space_projector(proj_frm_fourier, current_probe_est, projection_coords, img_wgt, full_img_sz) + \
                  (1 - 2 * relax_pm) * proj_frm_fourier + relax_pm * (proj_frm_space - current_frm_est)

        # update current frame data estimate
        current_frm_est = np.copy(frm_est)

        # obtain the current estimate of complex object
        obj_est = patch2img(frm_est * np.conj(current_probe_est), projection_coords, full_img_sz, img_wgt)

        # obtain the current estimate of probe function
        division_op_num = np.average(img2patch(np.conj(obj_est), projection_coords, diffraction_data.shape) * frm_est, axis=0)
        division_op_denom = np.average(img2patch(np.abs(obj_est) ** 2, projection_coords, diffraction_data.shape), axis=0)
        # division_op_num = np.sum(img2patch(np.conj(obj_est), scan_coords, diffr.shape) * frm_est, 0)
        # division_op_denom = np.sum(img2patch(np.abs(obj_est) ** 2, scan_coords, diffr.shape), 0)
        probe_est = divide_cmplx_numbers(division_op_num, division_op_denom)
        current_probe_est = np.copy(probe_est)

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
    print('Time consumption of SHARP:', elapsed_time)

    # save recon results
    save_tiff(obj_est, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
    save_tiff(probe_est, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
    save_array(obj_nrmse_ls, save_dir + 'obj_nrmse')
    save_array(probe_nrmse_ls, save_dir + 'probe_nrmse')
    save_array(diffr_nrmse_ls, save_dir + 'diffr_nrmse')

    # display the reconstructed image and probe
    plot_cmplx_obj(obj_revy, obj_ref, img_title='SHARP', display_win=display_win, display=display, save_fname=obj_fname)
    plot_cmplx_probe(probe_revy, probe_ref, img_title='SHARP', display=display, save_fname=probe_fname)

    # return the result
    keys = ['obj_revy', 'probe_revy', 'obj_err', 'probe_err', 'diffr_err']
    vals = [obj_revy, probe_revy, obj_nrmse_ls, probe_nrmse_ls, diffr_nrmse_ls]
    output = dict(zip(keys, vals))

    return output



def sharp_plus_joint_recon(init_obj, init_probe, diffraction_data, projection_coords, obj_ref=None, probe_ref=None,
                           num_iter=100, relax_pm=0.5, display_win=None, display=False, save_dir=None):
    """
    The SHARP+ is introduced in: https://arxiv.org/pdf/2111.14240.pdf. The frame estimate update function:
        $z_j \gets 2 \beta P_{Q,j} P_{a,j} +(1 - 2 \beta) P_{a,j} - \beta ( P_{Q,j} - I)  z_j (*)$
    :param init_obj: initial guess of complex image.
    :param init_probe: initial guess of probe function.
    :param diffraction_data: pre-processed phaseless measurements / diffraction patterns.
    :param projection_coords: coordinates of scan positions.
    :param obj_ref: ground truth complex image.
    :param probe_ref: ground truth complex probe.
    :param num_iter: number of iterations.
    :param relax_pm: relaxation parameter.
    :param display_win: reconstruction window.
    :param display: display the reconstruction results.
    :param save_dir: path for saving reconstructed images.
    :return: reconstructed complex image and probe and nrmse.
    """
    # check directories for saving files
    if save_dir is None:
        obj_fname = None
        probe_fname = None
    else:
        obj_fname = save_dir + 'SHARP_plus_img_revy'
        probe_fname = save_dir + 'SHARP_plus_probe_revy'
        os.makedirs(save_dir, exist_ok=True)

    if display_win is None:
        display_win = np.ones(init_obj.shape)

    # initialization
    obj_nrmse_ls = []
    probe_nrmse_ls = []
    diffr_nrmse_ls = []

    obj_est = np.copy(init_obj)
    projected_img_mat = img2patch(init_obj, projection_coords, diffraction_data.shape)

    probe_est = np.copy(init_probe)
    current_probe_est = np.copy(probe_est)

    frm_est = np.copy(projected_img_mat) * init_probe
    current_frm_est = np.copy(frm_est)

    full_img_sz = init_obj.shape
    start_time = time.time()

    # SHARP+ reconstruction
    print('SHARP+ joint recon starts ...')
    for i in range(num_iter):
        # calculate spatially-varying weights
        probe_mat = np.ones(diffraction_data.shape, dtype=np.complex128) * current_probe_est
        img_wgt = patch2img(np.abs(probe_mat) ** 2, projection_coords, full_img_sz) # + 1e-15

        proj_frm_fourier = fourier_projector(current_frm_est, diffraction_data)
        proj_frm_space = space_projector(current_frm_est, current_probe_est, projection_coords, img_wgt, full_img_sz)

        # # SHARP
        # # frm_est = 2 * relax_pm * space_projector(fourier_projector(current_frm_est, diffr), current_probe_est, scan_coords, img_wgt, full_img_sz) + \
        # #           (1 - 2 * relax_pm) * fourier_projector(current_frm_est, diffr) + \
        # #           relax_pm * (space_projector(current_frm_est, current_probe_est, scan_coords, img_wgt, full_img_sz) - current_frm_est)
        frm_est = 2 * relax_pm * space_projector(proj_frm_fourier, current_probe_est, projection_coords, img_wgt, full_img_sz) + \
                  (1 - 2 * relax_pm) * proj_frm_fourier - relax_pm * (proj_frm_space - current_frm_est)

        # update current frame data estimate
        current_frm_est = np.copy(frm_est)

        # obtain the current estimate of complex object
        obj_est = patch2img(frm_est * np.conj(current_probe_est), projection_coords, full_img_sz, img_wgt)

        # obtain the current estimate of probe function
        division_op_num = np.average(img2patch(np.conj(obj_est), projection_coords, diffraction_data.shape) * frm_est, axis=0)
        division_op_denom = np.average(img2patch(np.abs(obj_est) ** 2, projection_coords, diffraction_data.shape), axis=0)
        # division_op_num = np.sum(img2patch(np.conj(obj_est), scan_coords, diffr.shape) * frm_est, 0)
        # division_op_denom = np.sum(img2patch(np.abs(obj_est) ** 2, scan_coords, diffr.shape), 0)
        probe_est = divide_cmplx_numbers(division_op_num, division_op_denom)
        current_probe_est = np.copy(probe_est)

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
    print('Time consumption of SHARP+:', elapsed_time)

    # save recon results
    save_tiff(obj_est, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
    save_tiff(probe_est, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
    save_array(obj_nrmse_ls, save_dir + 'obj_nrmse')
    save_array(probe_nrmse_ls, save_dir + 'probe_nrmse')
    save_array(diffr_nrmse_ls, save_dir + 'diffr_nrmse')

    # display the reconstructed image and probe
    plot_cmplx_obj(obj_revy, obj_ref, img_title='SHARP+', display_win=display_win, display=display, save_fname=obj_fname)
    plot_cmplx_probe(probe_revy, probe_ref, img_title='SHARP+', display=display, save_fname=probe_fname)

    # return the result
    keys = ['obj_revy', 'probe_revy', 'obj_err', 'probe_err', 'diffr_err']
    vals = [obj_revy, probe_revy, obj_nrmse_ls, probe_nrmse_ls, diffr_nrmse_ls]
    output = dict(zip(keys, vals))

    return output
