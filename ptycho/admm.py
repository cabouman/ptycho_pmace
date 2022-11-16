from utils.utils import *


def admm(y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None, 
         num_iter=100, joint_recon=False, recon_win=None, save_dir=None, beta=1):
    """
    ADMM for blind ptychography reconstruction. 
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
        beta: positive parameter in augmented Lagrangian function.
    Returns:
        Reconstructed complex images and NRMSE between reconstructions and reference images.
    """
    approach = 'ADMM'
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
    est_meas = compute_ft(est_probe * est_patch)

    multiplier_lambda = 0
    img_sz = est_obj.shape

    # SHARP reconstruction
    start_time = time.time()
    print('Generalized ADMM recon starts ...')
    for i in range(num_iter):
        # calculate y_hat
        est_meas_hat = est_meas + multiplier_lambda / beta
        ift_meas_hat = compute_ift(est_meas_hat)
        # update img
        img_wgt = patch2img(np.abs([est_probe] * len(y_meas)) ** 2, patch_bounds, img_sz)
        est_obj = patch2img(np.conj(est_probe) * ift_meas_hat, patch_bounds, img_sz, img_wgt)
        # update probe
        est_probe = divide_cmplx_numbers(np.sum(np.conj(est_patch) * ift_meas_hat, axis=0), np.sum(np.abs(est_patch)**2, axis=0))

        # update est
        est_meas_tmp = compute_ft(est_probe * est_patch)
        z_dagger = est_meas_tmp - multiplier_lambda / beta 
        est_meas = ((y_meas + beta * np.abs(z_dagger)) / (1 + beta)) * np.exp(1j * np.angle(z_dagger))

        # update multiplier_lambda
        multiplier_lambda = multiplier_lambda + beta * (est_meas - est_meas_tmp)
 
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
        est_meas = np.abs(compute_ft(est_probe * est_patch))
        nrmse_meas.append(compute_nrmse(est_meas, y_meas))
        print('iter = ', i, err_obj, err_probe, np.asarray(nrmse_meas)[-1])

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
