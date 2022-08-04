from utils.utils import *
import random


def epie_recon(y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None,
               num_iter=100, joint_recon=False, recon_win=None, save_dir=None,
               obj_step_sz=0.5, probe_step_sz=0.5):
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
        obj_step_sz: step size parameter for updating object estimate.
        probe_step_sz: step size parameter for updating probe estimate.
    Return:
        Reconstructed complex images and NRMSE between reconstructions and reference images.
    """
    cdtype = np.complex64
    approach = 'ePIE'
    # check directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # initialization
    if recon_win is None:
        recon_win = np.ones_like(init_obj)

    nrmse_obj = []
    nrmse_probe = []
    seq = np.arange(0, len(y_meas), 1).tolist()

    est_obj = np.copy(init_obj).astype(cdtype)
    est_probe = np.copy(init_probe).astype(cdtype) if joint_recon else np.copy(ref_probe).astype(cdtype)

    # ePIE reconstruction
    start_time = time.time()
    print('ePIE recon starts ...')
    for i in range(num_iter):
        random.shuffle(seq)
        for j in seq:
            crd0, crd1, crd2, crd3 = patch_bounds[j, 0], patch_bounds[j, 1], patch_bounds[j, 2], patch_bounds[j, 3]
            projected_img = np.copy(est_obj[crd0:crd1, crd2:crd3])
            frm = projected_img * est_probe
            # take Fourier Transform
            f = compute_ft(frm)
            # revise estimate of frame data
            delta_frm = compute_ift(y_meas[j] * np.exp(1j * np.angle(f))) - frm
            # revise estimates of complex object
            est_obj[crd0:crd1, crd2:crd3] += obj_step_sz * np.conj(est_probe) * delta_frm / (np.amax(np.abs(est_probe)) ** 2)
            if joint_recon:
                est_probe += probe_step_sz * np.conj(projected_img) * delta_frm / (np.amax(np.abs(projected_img)) ** 2)
    
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
        if nrmse_obj:
            save_array(nrmse_obj, save_dir + 'nrmse_obj_' + str(nrmse_obj[-1]))
        if joint_recon:
            save_tiff(est_probe, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
            if nrmse_probe:
                save_array(nrmse_probe, save_dir + 'nrmse_probe_' + str(nrmse_probe[-1]))

    # return recon results
    keys = ['object', 'probe', 'err_obj', 'err_probe']
    vals = [revy_obj, revy_probe, nrmse_obj, nrmse_probe]
    output = dict(zip(keys, vals))

    return output


