from ptycho_pmace.utils.utils import *
from ptycho_pmace.utils.nrmse import *
import random


def epie_recon(dp, project_coords, init_obj, init_probe=None, obj_ref=None, probe_ref=None,
               num_iter=100, obj_step_sz=1, probe_step_sz=1, joint_recon=False, cstr_win=None, save_dir=None):
    """
    Function to perform ePIE reconstruction on ptychographic data.
    :param dp: pre-processed diffraction pattern (intensity data).
    :param project_coords: scan coordinates of projections.
    :param init_obj: formulated initial guess of complex object.
    :param init_probe: formulated initial guess of complex probe.
    :param obj_ref: complex reference image.
    :param probe_ref: complex reference image.
    :param num_iter: number of iterations.
    :param obj_step_sz: step size parameter for updating object estimate.
    :param probe_step_sz: step size parameter for updating probe estiamte.
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

    num_agts, m, n = dp.shape
    seq = np.arange(0, num_agts, 1).tolist()
    obj_est = np.asarray(np.copy(init_obj), dtype=np.complex128)
    probe_est = np.asarray(np.copy(init_probe), dtype=np.complex128) if joint_recon else np.copy(probe_ref)
    obj_nrmse_ls = []
    probe_nrmse_ls = []
    dp_nrmse_ls = []
    time_ls = []

    # ePIE reconstruction
    print('ePIE recon starts ...')
    start_time = time.time()
    for i in range(num_iter):
        print('iter=', i)
        random.shuffle(seq)
        for j in seq:
            crd0, crd1, crd2, crd3 = project_coords[j, 0], project_coords[j, 1], project_coords[j, 2], project_coords[j, 3]
            projected_img = np.copy(obj_est[crd0:crd1, crd2:crd3])
            frm = projected_img * probe_est
            # take Fourier Transform
            spectr = compute_ft(frm)
            updated_spectr = dp[j] * divide_cmplx_numbers(np.copy(spectr), np.abs(np.copy(spectr)))
            # update object and probe estimates
            delta_frm = compute_ift(updated_spectr) - frm
            obj_est[crd0:crd1, crd2:crd3] += obj_step_sz * np.conj(probe_est) * delta_frm / (np.amax(np.abs(probe_est)) ** 2)
            if joint_recon:
                probe_est += probe_step_sz * np.conj(projected_img) * delta_frm / (np.amax(np.abs(projected_img)) ** 2)

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
    print('Time consumption of ePIE:', elapsed_time)

    # save recon results
    save_tiff(obj_est, save_dir + 'obj_est_iter_{}.tiff'.format(i + 1))
    save_array(obj_nrmse_ls,  save_dir + 'obj_nrmse')
    save_array(dp_nrmse_ls, save_dir + 'diffr_nrmse')
    if joint_recon:
        save_tiff(probe_est, save_dir + 'probe_est_iter_{}.tiff'.format(i + 1))
        save_array(probe_nrmse_ls, save_dir + 'probe_nrmse')

    # return recon results
    keys = ['obj_revy', 'probe_revy', 'obj_err', 'probe_err', 'diffr_err']
    vals = [obj_revy, probe_revy, obj_nrmse_ls, probe_nrmse_ls, dp_nrmse_ls]
    output = dict(zip(keys, vals))

    return output

