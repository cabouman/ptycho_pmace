from utils.utils import *


def wf_obj_func(cur_est, probe, y_meas, patch_bounds, discretized_sys_mat, prm=1):
    """
    Function to revise estimate of complex object using WF.
    Args:
        cur_est: current estimate of complex object. 
        probe: complex probe. 
        y_meas: pre-processed measurements.
        patch_bounds: scan coordinates of projections.
        discretized_sys_mat: the eigen value is used to obtain step size.
        prm: prm is 1 when FT is orthonormal.
    Return:
        revised estimate of complex object.
    """
    # take projection of image
    patch = img2patch(cur_est, patch_bounds, y_meas.shape)
    # Ax = FT{D*P_j*v}
    f = compute_ft(patch * probe)
    # Ax - y * Ax / |Ax| = - FT{D * P_j * v} - y * FT{D * P_j * v} / |FT{D * P_j * v}|
    # A^(H){```} = P^t D^* IFFT{```}
    inv_f = compute_ift(f - y_meas * np.exp(1j * np.angle(f)))
    # back projection
    output = cur_est - patch2img(inv_f * np.conj(probe), patch_bounds, cur_est.shape) / np.amax(prm * discretized_sys_mat)
    
    return output.astype(np.complex64)


def wf_probe_func(cur_est, img_patch, y_meas, discretized_sys_mat, prm=1):
    """
    Function to revise estimate of complex probe using WF.
    Args:
        cur_est: current estimate of complex probe.
        img_patch: projected image patches.
        y_meas: pre-processed measurements.
        discretized_sys_mat: the eigen value is used to obtain step size for probe function.
        prm: prm = 1 if FT is orthonormal.
    Return:
        new estimate of complex probe.
    """
    # Bd = FT{D*X_j*d}
    f = compute_ft(cur_est * img_patch)
    # Bd - y * Bd / |Bd| =  FT{D*X_j*d} - y * FT{D*X_j*d} / |FT{D*X_j*d}|
    # B^(H){```} = (X_j)^(H) IFFT{```}
    inv_f = compute_ift(f - y_meas * np.exp(1j * np.angle(f)))
    # step_sz = 1/biggest eigenvalue of semi positive deifnite matrix =1/\lambda_max(A^(H)A)=1/(alpha*sum_j |D_j|^2)
    output = cur_est - np.average(np.conj(img_patch) * inv_f / np.amax(prm * discretized_sys_mat), axis=0)
    
    return output.astype(np.complex64)


def wf_recon(y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None, 
             num_iter=100, joint_recon=False, recon_win=None, save_dir=None, accel=True):
    """
    Function to perform WF/AWF reconstruction on ptychographic data.
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
        accel: option to add Nesterov's acceleration.
    Return:
        Reconstructed complex images and NRMSE between reconstructions and reference images.
    """
    cdtype = np.complex64
    approach = 'AWF' if accel else 'WF'
    # check directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # initialization
    if recon_win is None:
        recon_win = np.ones_like(init_obj)

    nrmse_obj = []
    nrmse_probe = []
    nrmse_meas = []

    est_obj = np.asarray(init_obj, dtype=cdtype)
    old_obj = np.copy(est_obj)

    est_probe = np.asarray(init_probe, dtype=cdtype) if joint_recon else ref_probe.astype(cdtype)
    old_probe = np.copy(est_probe)

    # calculate weight matrix for object update function
    obj_wgt_mat = patch2img(np.abs([est_probe] * len(y_meas)) ** 2, patch_bounds, est_obj.shape)

    # WF reconstruction
    start_time = time.time()
    print('{} recon starts ...'.format(approach))
    for i in range(num_iter):
        if accel:
            beta = (i + 2) / (i + 4)
        else:
            beta = 0
        # revise estimate of complex object
        cur_obj = est_obj + beta * (est_obj - old_obj)
        old_obj = np.copy(est_obj)
        est_obj = wf_obj_func(cur_obj, est_probe, y_meas, patch_bounds, obj_wgt_mat)

        if joint_recon:
            # calculate weight matrix for probe update function
            est_patch = img2patch(est_obj, patch_bounds, y_meas.shape)
            probe_wgt_mat = np.sum(np.abs(est_patch) ** 2, 0)
            # revise estimate of complex probe
            cur_probe = est_probe + beta * (est_probe - old_probe)
            old_probe = np.copy(est_probe)
            est_probe = wf_probe_func(cur_probe, est_patch, y_meas, probe_wgt_mat)
            # update weight matrix for object update function
            obj_wgt_mat = patch2img(np.abs([est_probe] * len(y_meas)) ** 2, patch_bounds, est_obj.shape)

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

    # calculate time consumption
    print('Time consumption of {}:'.format(approach), time.time() - start_time)

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
    keys = ['object', 'probe', 'err_obj', 'err_probe', 'err_meas']
    vals = [revy_obj, revy_probe, nrmse_obj, nrmse_probe, nrmse_meas]
    output = dict(zip(keys, vals))

    return output
