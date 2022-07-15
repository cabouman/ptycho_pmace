from utils.utils import *


def fourier_projector(frame_data, y_meas):
    """
    This phase projector projects the frames onto the Fourier magnitude constraint.
                   P_a z = [P_a1 z_1, P_a2 z_2, ..., P_aJ z_J]
    where
    P_aj z_j = F^* {y_j * F z_j / |F z_j|}.
    Args:
        frame_data: image patches multiplied by beam profile function at each scan position, i.e. D*P_j*x.
        diffr_data: pre-processed diffraction patterns (recorded phase-less measurements).
    Returns:
        revised estimates of frames.
    """
    # FT{D*P_j*v}]
    f = compute_ft(frame_data)
    # IFT { y \times FT{D*P_j*v}] / |FT{D*P_j*v}]| }
    output = compute_ift(y_meas * np.exp(1j * np.angle(f)))

    return output


def space_projector(frame_data, probe, coords, norm, img_sz):
    """
    The image projector matches the object with object domain constraint.
                           P_Q = Q (Q^* Q)^(-1) Q^*
    Or equivalently
                            P_Q = Q Lambda^(-1) Q^*
    where
    Lambda = \sum_j P_j^t D^* D P_j.
    Args:
        frame_data: the extracted frames z_j = D P_j x.
        probe: the beam profile function.
        coords: coordinates of projections.
        norm: \sum P_j^t D^* D P_j.
        img_sz: the shape of full-size image.
    Returns:
        revised estimates of frames.
    """
    wgt_img = patch2img(frame_data * np.conj(probe), coords, img_sz, norm)
    output = img2patch(wgt_img, coords, frame_data.shape) * probe

    return output


def sharp_recon(y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None, 
                num_iter=100, joint_recon=False, recon_win=None, save_dir=None, relax_pm=0.75):
    """
    Function to perform PMACE reconstruction on ptychographic data.
    Args:
        y_meas: pre-processed measurements (diffraction patterns / intensity data).
        patch_bounds: scan coordinates of projections.
        init_obj: formulated initial guess of complex object.
        init_probe: formulated initial guess of complex probe.
        ref_obj: complex reference image for object.
        ref_probe: complex reference image for probe.
        num_iter: number of iterations.
        joint_recon: option to estimate complex probe for blind ptychography.
        recon_win: pre-defined window for showing and comparing reconstruction results.
        save_dir: directory to save reconstruction results.
        relax_pm: relaxation parameter.
    Returns:
        Reconstructed complex images and NRMSE between reconstructions and reference images.
    """
    approach = 'SHARP'
    cdtype = np.complex64
    # check directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # initialization
    if recon_win is None:
        recon_win = np.ones_like(init_obj)

    nrmse_obj = []
    nrmse_probe = []

    est_obj = np.copy(init_obj).astype(cdtype)
    est_patch = img2patch(est_obj, patch_bounds, y_meas.shape)
    est_probe = np.asarray(init_probe, dtype=cdtype) if joint_recon else ref_probe.astype(cdtype)
    est_frm = est_patch * est_probe
    cur_frm = np.copy(est_frm)
    
    # calculate spatially-varying image weights
    img_sz = est_obj.shape
    img_wgt = patch2img(np.abs([est_probe] * len(y_meas)) ** 2, patch_bounds, img_sz)


    # SHARP reconstruction
    start_time = time.time()
    print('SHARP recon starts ...')
    for i in range(num_iter):
        # take projections
        tmp_frm_f = fourier_projector(cur_frm, y_meas)
        tmp_frm_s = space_projector(cur_frm, est_probe, patch_bounds, img_wgt, img_sz)
        # SHARP+ updates 
        est_frm = 2 * relax_pm * space_projector(tmp_frm_f, est_probe, patch_bounds, img_wgt, img_sz) + (1 - 2 * relax_pm) * tmp_frm_f + relax_pm * (tmp_frm_s - cur_frm)
        # update current estimate of frame data
        cur_frm = np.copy(est_frm)

        # obtain estimate of complex object
        est_obj = patch2img(est_frm * np.conj(est_probe), patch_bounds, img_sz, img_wgt)
        if joint_recon:
            # obtain estimate of complex probe
            tmp_n = np.average(img2patch(np.conj(est_obj), patch_bounds, y_meas.shape) * est_frm, axis=0)
            tmp_d = np.average(img2patch(np.abs(est_obj) ** 2, patch_bounds, y_meas.shape), axis=0)
            est_probe = divide_cmplx_numbers(tmp_n, tmp_d)
            # update image weights
            img_wgt = patch2img(np.abs([est_probe] * len(y_meas)) ** 2, patch_bounds, img_sz)

        # # dynamic strategy for updating beta
        # beta = beta + (1 - beta) * (1 - np.exp(-(i/7)**3))
 
        # phase normalization and scale image to minimize the intensity difference
        if ref_obj is not None:
            revy_obj = phase_norm(np.copy(est_obj) * recon_win, ref_obj * recon_win, cstr=recon_win)
            err_obj = compute_nrmse(revy_obj * recon_win, ref_obj * recon_win, cstr=recon_win)
            nrmse_obj.append(err_obj)
        else:
            revy_obj = est_obj 
        if joint_recon:
            if ref_probe is not None:
                revy_probe = phase_norm(np.copy(est_probe), ref_probe)
                err_probe = compute_nrmse(revy_probe, ref_probe)
                nrmse_probe.append(err_probe)
            else:
                revy_probe = est_probe
        else:
            revy_probe = est_probe

    # calculate time consumption
    print('Time consumption of {}:'.format(approach), time.time() - start_time)

    # save recon results
    if save_dir is not None:
        save_tiff(est_obj, save_dir + 'est_obj_iter_{}.tiff'.format(i + 1))
        save_array(nrmse_obj, save_dir + 'nrmse_obj_' + str(nrmse_obj[-1]))
        if joint_recon:
            save_tiff(est_probe, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
            save_array(nrmse_probe, save_dir + 'nrmse_probe_' + str(nrmse_probe[-1]))

    # return recon results
    keys = ['object', 'probe', 'err_obj', 'err_probe']
    vals = [revy_obj, revy_probe, nrmse_obj, nrmse_probe]
    output = dict(zip(keys, vals))

    return output



def sharp_plus_recon(y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None,
                     num_iter=100, joint_recon=False, recon_win=None, save_dir=None, relax_pm=0.75):
    """
    Function to perform PMACE reconstruction on ptychographic data.
    Args:
        y_meas: pre-processed measurements (diffraction patterns / intensity data).
        patch_bounds: scan coordinates of projections.
        init_obj: formulated initial guess of complex object.
        init_probe: formulated initial guess of complex probe.
        ref_obj: complex reference image for object.
        ref_probe: complex reference image for probe.
        num_iter: number of iterations.
        joint_recon: option to estimate complex probe for blind ptychography.
        recon_win: pre-defined window for showing and comparing reconstruction results.
        save_dir: directory to save reconstruction results.
        relax_pm: relaxation parameter.
    Returns:
        Reconstructed complex images and NRMSE between reconstructions and reference images.
    """
    approach = 'SHARP+'
    cdtype = np.complex64
    # check directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # initialization
    if recon_win is None:
        recon_win = np.ones_like(init_obj)

    nrmse_obj = []
    nrmse_probe = []

    est_obj = np.copy(init_obj).astype(cdtype)
    est_patch = img2patch(est_obj, patch_bounds, y_meas.shape)
    est_probe = np.asarray(init_probe, dtype=cdtype) if joint_recon else ref_probe.astype(cdtype)
    est_frm = est_patch * est_probe
    cur_frm = np.copy(est_frm)
    
    # calculate spatially-varying image weights
    img_sz = est_obj.shape
    img_wgt = patch2img(np.abs([est_probe] * len(y_meas)) ** 2, patch_bounds, img_sz)


    # SHARP+ reconstruction
    start_time = time.time()
    print('SHARP+ recon starts ...')
    for i in range(num_iter):
        # take projections
        tmp_frm_f = fourier_projector(cur_frm, y_meas)
        tmp_frm_s = space_projector(cur_frm, est_probe, patch_bounds, img_wgt, img_sz)
        # SHARP+ updates 
        est_frm = 2 * relax_pm * space_projector(tmp_frm_f, est_probe, patch_bounds, img_wgt, img_sz) + (1 - 2 * relax_pm) * tmp_frm_f - relax_pm * (tmp_frm_s - cur_frm)
        # update current estimate of frame data
        cur_frm = np.copy(est_frm)

        # obtain estimate of complex object
        est_obj = patch2img(est_frm * np.conj(est_probe), patch_bounds, img_sz, img_wgt)
        if joint_recon:
            # obtain estimate of complex probe
            tmp_n = np.average(img2patch(np.conj(est_obj), patch_bounds, y_meas.shape) * est_frm, axis=0)
            tmp_d = np.average(img2patch(np.abs(est_obj) ** 2, patch_bounds, y_meas.shape), axis=0)
            est_probe = divide_cmplx_numbers(tmp_n, tmp_d)
            # update image weights
            img_wgt = patch2img(np.abs([est_probe] * len(y_meas)) ** 2, patch_bounds, img_sz)

        # # dynamic strategy for updating beta
        # beta = beta + (1 - beta) * (1 - np.exp(-(i/7)**3))
 
        # phase normalization and scale image to minimize the intensity difference
        if ref_obj is not None:
            revy_obj = phase_norm(np.copy(est_obj) * recon_win, ref_obj * recon_win, cstr=recon_win)
            err_obj = compute_nrmse(revy_obj * recon_win, ref_obj * recon_win, cstr=recon_win)
            nrmse_obj.append(err_obj)
        else:
            revy_obj = est_obj 
        if joint_recon:
            if ref_probe is not None:
                revy_probe = phase_norm(np.copy(est_probe), ref_probe)
                err_probe = compute_nrmse(revy_probe, ref_probe)
                nrmse_probe.append(err_probe)
            else:
                revy_probe = est_probe
        else:
            revy_probe = est_probe

    # calculate time consumption
    print('Time consumption of {}:'.format(approach), time.time() - start_time)

    # save recon results
    if save_dir is not None:
        save_tiff(est_obj, save_dir + 'est_obj_iter_{}.tiff'.format(i + 1))
        save_array(nrmse_obj, save_dir + 'nrmse_obj_' + str(nrmse_obj[-1]))
        if joint_recon:
            save_tiff(est_probe, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
            save_array(nrmse_probe, save_dir + 'nrmse_probe_' + str(nrmse_probe[-1]))

    # return recon results
    keys = ['object', 'probe', 'err_obj', 'err_probe']
    vals = [revy_obj, revy_probe, nrmse_obj, nrmse_probe]
    output = dict(zip(keys, vals))

    return output
