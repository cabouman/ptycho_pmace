from utils.utils import *
from bm4d import bm4d, BM4DProfile, BM4DStages, BM4DProfile2D, BM4DProfileComplex, BM4DProfileBM3DComplex


def prox_map_op(cur_est, joint_est, y_meas, data_fit_prm):
    """
    The weighted proximal map operator F to revise estiamtes of complex patches and probe.
    Args:
        cur_est: current estimate of projected images or complex probe.
        joint_est: current estimate of complex probe or projected images.
        y_meas: pre-processed ptychographic measurements.
        data_fit_prm: prm/(1-prm) denotes noise-to-signal ratio of data.
    Return:
        New estiamtes of projected image patches or complex probe.
    """
    # DFT{D * P_j * v}
    f = compute_ft(cur_est * joint_est)
    ## y * DFT{D * P_j * v} / | DFT{D * P_j * v} |
    #new_f = (y_meas * np.angle(f)).astype(np.complex64)

    # IDFT{y * DFT{D * P_j * v} / | DFT{D * P_j * v} |}
    inv_f = compute_ift(y_meas * divide_cmplx_numbers(f, np.abs(f)))
    # take weighted average of current estimate and closest data-fitting point
    output = (1 - data_fit_prm) * cur_est + data_fit_prm * divide_cmplx_numbers(inv_f, joint_est)

    return output


def consens_op(patch, patch_bounds, img_sz, img_wgt, add_reg=False, bm3d_psd=0.1, blk_idx=None):
    """
    The consensus operator G takes weighted average of projections and 
    realloates the consensus results.
    Args:
        patch: current estimate of image patches.
        patch_bounds: scan coordinates of projections.
        img_sz: full image size.
        img_wgt: image weight.
        add_reg: option to apply denoiser.
        bm3d_psd: psd of complex bm3d denoising.
        blk_idx: pre-defines region for applying denoisers.
    Return:
        new estimate of projected images.
    """
    # take weighted average of input patches
    cmplx_img = patch2img(patch, patch_bounds, img_sz, img_wgt)
    
    # add regularzation
    if add_reg:
        if blk_idx is None:
            blk_idx = [0, img_sz[0], 0, img_sz[1]]
        tmp_img = cmplx_img[blk_idx[0]: blk_idx[1], blk_idx[2]: blk_idx[3]]
        # apply complex bm3d
        denoised_img = bm4d(tmp_img, bm3d_psd, profile=BM4DProfileBM3DComplex())[:, :, 0]
        cmplx_img[blk_idx[0]: blk_idx[1], blk_idx[2]: blk_idx[3]] = denoised_img

    # reallocate consensus result
    new_patch = img2patch(cmplx_img, patch_bounds, patch.shape)

    return cmplx_img, new_patch


def pmace_recon(y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None,
                num_iter=100, joint_recon=False, recon_win=None, save_dir=None,
                obj_data_fit_prm=0.5, probe_data_fit_prm=0.5, 
                rho=0.5, probe_exp=1.5, obj_exp=0.5, add_reg=True, sigma=0.1):
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
        obj_data_fit_prm: averaging weight in object update function. Param near 1 gives closer fit to data.
        probe_data_fit_prm: averaging weight in probe update function. 
        rho: Mann averaging parameter.
        probe_exp: exponent of probe weighting in consensus calculation of image estimate.
        obj_exp: exponent of image weighting in consensus calculation of probe estimate.
        add_reg: option to apply denoiser.
        sigma: denoising parameter.
    Return:
        Reconstructed complex images and NRMSE between reconstructions and reference images.
    """
    approach = 'reg-PMACE' if add_reg else 'PMACE'
    cdtype = np.complex64
    # check directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # initialization
    if recon_win is None:
        recon_win = np.ones_like(init_obj)

    nrmse_obj = []
    nrmse_probe = []

    est_obj = np.asarray(init_obj, dtype=cdtype)
    cur_patch = img2patch(est_obj, patch_bounds, y_meas.shape).astype(cdtype)
    new_patch = np.copy(cur_patch)

    est_probe = np.asarray(init_probe, dtype=cdtype) if joint_recon else ref_probe.astype(cdtype)
    consens_probe = np.copy(est_probe).astype(cdtype)
    cur_probe_arr = [est_probe] * len(y_meas)
    new_probe_arr = np.copy(cur_probe_arr)

    # calculate image weight and patch weight
    image_sz = est_obj.shape
    patch_weight = np.abs(consens_probe) ** probe_exp
    image_weight = patch2img([patch_weight] * len(y_meas), patch_bounds, image_sz)

    # determine the area for applying denoiser
    denoising_blk = patch2img(np.ones_like(y_meas), patch_bounds, init_obj.shape)
    dn_idx = np.nonzero(denoising_blk)
    crd0, crd1 = np.max([0, np.amin(dn_idx[0])]), np.min([np.amax(dn_idx[0])+1, est_obj.shape[0]])
    crd2, crd3 = np.max([0, np.amin(dn_idx[1])]), np.min([np.amax(dn_idx[1])+1, est_obj.shape[1]])
    blk_idx = [crd0, crd1, crd2, crd3]

    start_time = time.time()
    print('{} recon starts ...'.format(approach))
    # PMACE reconstruction
    for i in range(num_iter):
        # w <- F(v; w)
        cur_patch = prox_map_op(new_patch, consens_probe, y_meas, obj_data_fit_prm)
        # z <- G(2w - v)
        est_obj, consens_patch = consens_op((2 * cur_patch - new_patch) * patch_weight, patch_bounds, img_wgt=image_weight, 
                                            img_sz=image_sz, add_reg=add_reg, bm3d_psd=sigma, blk_idx=blk_idx)
        # v <- v + 2 \rho (z - w)
        new_patch = new_patch + 2 * rho * (consens_patch - cur_patch)
        # obtain estimate of complex object
        if not add_reg:
            est_obj = patch2img(new_patch * patch_weight, patch_bounds, image_sz, image_weight)

        if joint_recon:
            # calculate probe weights
            probe_arr_weight = np.abs(consens_patch) ** obj_exp
            probe_weight = np.sum(probe_arr_weight, 0)
            # w <- F(v; w)
            cur_probe_arr = prox_map_op(new_probe_arr, consens_patch, y_meas, probe_data_fit_prm)
            # z <- G(2w - v)
            consens_probe = np.sum((2 * cur_probe_arr - new_probe_arr) * probe_arr_weight, 0) / probe_weight
            # v <- v + 2 \rho (z - w)
            new_probe_arr = new_probe_arr + 2 * rho * (consens_probe - cur_probe_arr)
            # update image weights
            patch_weight = np.abs(consens_probe) ** probe_exp
            image_weight = patch2img([patch_weight] * len(y_meas), patch_bounds, image_sz)
            # obtain estiamte of complex probe
            est_probe = np.sum(new_probe_arr * probe_arr_weight, 0) / probe_weight

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
