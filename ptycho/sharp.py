import time
from utils.utils import *
from utils.nrmse import *


def fourier_projector(frame_data, y_meas):
    """Fourier projector.

    This Fourier projector projects frame data onto the Fourier magnitude constraints.

    Args:
        frame_data: product between illuminated object and probe.
        y_meas: pre-processed data (square root of recorded phaseless intensity measurements).

    Returns:
        revised estimates of frame data.
    """
    # FT
    f_tmp = compute_ft(frame_data)
    
    # IFT 
    output = compute_ift(y_meas * np.exp(1j * np.angle(f_tmp)))

    return output


def space_projector(frame_data, probe, coords, img_wgt, img_sz):
    """Space projector.
    
    This image projector matches the object with object domain constraint.
    
    Args:
        frame_data: product between illuminated object and probe.
        probe: complex beam profile function.
        coords: coordinates of projections.
        img_wgt: image weight.
        img_sz: shape of full-size image.
        
    Returns:
        revised estimates of frame data.
    """
    # frame data to weighted image
    img_tmp = patch2img(frame_data * np.conj(probe), coords, img_sz, img_wgt)
    
    # image to frame data
    output = img2patch(img_tmp, coords, frame_data.shape) * probe

    return output


def sharp_recon(y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None, 
                num_iter=100, joint_recon=False, recon_win=None, save_dir=None, relax_pm=0.75):
    """SHARP.
    
    Function to perform SHARP reconstruction on ptychographic data. 
    
    Args:
        y_meas: pre-processed data (square root of recorded phaseless intensity measurements).
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
    nrmse_meas = []

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
            est_probe = np.divide(tmp_n, tmp_d, where=(tmp_d!=0))
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

        # calculate error in measurement domain
        est_patch = img2patch(est_obj, patch_bounds, y_meas.shape).astype(cdtype)
        est_meas = np.abs(compute_ft(est_probe * est_patch))
        nrmse_meas.append(compute_nrmse(est_meas, y_meas))

        if (i+1) % 10 == 0:
            print('Finished {:d} of {:d} iterations.'.format(i+1, num_iter))

    # # calculate time consumption
    # print('Time consumption of {}:'.format(approach), time.time() - start_time)

    # save recon results
    if save_dir is not None:
        save_tiff(est_obj, save_dir + 'est_obj_iter_{}.tiff'.format(i + 1))
        if nrmse_obj:
            save_array(nrmse_obj, save_dir + 'nrmse_obj_' + str(nrmse_obj[-1]))
        if nrmse_meas:
            save_array(nrmse_meas, save_dir + 'nrmse_meas_' + str(nrmse_meas[-1]))
        if joint_recon:
            save_tiff(est_probe, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
            if nrmse_probe:
                save_array(nrmse_probe, save_dir + 'nrmse_probe_' + str(nrmse_probe[-1]))

    # return recon results
    print('{} recon completed.'.format(approach))
    keys = ['object', 'probe', 'err_obj', 'err_probe', 'err_meas']
    vals = [revy_obj, revy_probe, nrmse_obj, nrmse_probe, nrmse_meas]
    output = dict(zip(keys, vals))

    return output


def sharp_plus_recon(y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None,
                     num_iter=100, joint_recon=False, recon_win=None, save_dir=None, relax_pm=0.75):
    """SHARP+.
    
    Function to perform SHARP+ reconstruction on ptychographic data.
    
    Args:
        y_meas: pre-processed data (square root of recorded phaseless intensity measurements).
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
    nrmse_meas = []

    est_obj = np.copy(init_obj).astype(cdtype)
    est_patch = img2patch(est_obj, patch_bounds, y_meas.shape)
    est_probe = np.asarray(init_probe, dtype=cdtype) if joint_recon else ref_probe.astype(cdtype)
    est_frm = est_patch * est_probe
    cur_frm = np.copy(est_frm)
    
    # calculate spatially-varying image weights
    img_sz = est_obj.shape
    img_wgt = patch2img(np.abs([est_probe] * len(y_meas)) ** 2, patch_bounds, img_sz)

    # SHARP+ reconstruction
    # start_time = time.time()
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
            est_probe = np.divide(tmp_n, tmp_d, where=(tmp_d!=0))
            # update image weights
            img_wgt = patch2img(np.abs([est_probe] * len(y_meas)) ** 2, patch_bounds, img_sz)

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

        # calculate error in measurement domain
        est_patch = img2patch(est_obj, patch_bounds, y_meas.shape).astype(cdtype)
        est_meas = np.abs(compute_ft(est_probe * est_patch))
        nrmse_meas.append(compute_nrmse(est_meas, y_meas))

        if (i+1) % 10 == 0:
            print('Finished {:d} of {:d} iterations.'.format(i+1, num_iter))

    # # calculate time consumption
    # print('Time consumption of {}:'.format(approach), time.time() - start_time)

    # save recon results
    if save_dir is not None:
        save_tiff(est_obj, save_dir + 'est_obj_iter_{}.tiff'.format(i + 1))
        if nrmse_obj:
            save_array(nrmse_obj, save_dir + 'nrmse_obj_' + str(nrmse_obj[-1]))
        if nrmse_meas:
            save_array(nrmse_meas, save_dir + 'nrmse_meas_' + str(nrmse_meas[-1]))
        if joint_recon:
            save_tiff(est_probe, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
            if nrmse_probe:
                save_array(nrmse_probe, save_dir + 'nrmse_probe_' + str(nrmse_probe[-1]))

    # return recon results
    print('{} recon completed.'.format(approach))
    keys = ['object', 'probe', 'err_obj', 'err_probe', 'err_meas']
    vals = [revy_obj, revy_probe, nrmse_obj, nrmse_probe, nrmse_meas]
    output = dict(zip(keys, vals))

    return output
