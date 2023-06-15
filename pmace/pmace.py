import time, os
from tqdm import tqdm
from utils.utils import *
from utils.nrmse import *
from bm4d import bm4d, BM4DProfile, BM4DStages, BM4DProfile2D, BM4DProfileComplex, BM4DProfileBM3DComplex


class PMACE():
    """This class is a decorator that can be used to prepare a function before it is called.

    Args:
        func (function): The function to be decorated.
        *args: Positional arguments to be passed to the decorated function.
        **kwargs: Keyword arguments to be passed to the decorated function.
    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs   
        
    def __call__(self):
        def wrapper(*args, **kwargs):
            print('Preparing function ...')
            return_val = self.func(*args, **kwargs)
            return return_val
        
        return wrapper(*self.args, **self.kwargs)
    

def get_data_fit_pt(cur_est, joint_est, y_meas):
    """Data-fitting point.
    
    The function calculates the closest data fitting point given measurement and current estimate.
    
    Args:
        cur_est: current estimate of projected images or complex probe.
        joint_est: current estimate of complex probe or projected images.
        y_meas: pre-processed ptychographic measurements.
        
    Returns:
        closest data fitting point.  
    """
    # DFT{D * P_j * v}
    f = compute_ft(cur_est * joint_est)
    
    # IDFT{y * DFT{D * P_j * v} / | DFT{D * P_j * v} |}
    inv_f = compute_ift(y_meas * np.exp(1j * np.angle(f)))
    
    # calculate closest data-fitting point
    output = divide_cmplx_numbers(inv_f, joint_est)

    return output


# def prox_map_op(cur_est, joint_est, y_meas, data_fit_prm):
#     r"""Data-fitting operator.

#     The weighted proximal map operator :math:`F` is a stack of data-fitting agents,
#     which revises estiamtes of complex patches or probe.

#     Args:
#         cur_est: current estimate of projected images or complex probe.
#         joint_est: current estimate of complex probe or projected images.
#         y_meas: pre-processed ptychographic measurements.
#         data_fit_prm: prm/(1-prm) denotes noise-to-signal ratio of data.

#     Returns:
#         New estimates of projected image patches or complex probe.
#     """
#     # calculate closest data-fitting point
#     data_fit_pt = get_data_fit_pt(cur_est, joint_est, y_meas)
    
#     # take weighted average of current estimate and closest data-fitting point
#     output = (1 - data_fit_prm) * cur_est + data_fit_prm * data_fit_pt

#     return output


def consens_op(patch, patch_bounds, img_sz, img_wgt, add_reg=False, bm3d_psd=0.1, blk_idx=None):
    r"""Consensus operator.

    The consensus operator :math:`G` takes weighted average of projections and 
    reallocates the results.
    
    Args:
        patch: current estimate of image patches.
        patch_bounds: scan coordinates of projections.
        img_sz: full image size.
        img_wgt: image weight.
        add_reg: option to apply denoiser.
        bm3d_psd: psd of complex bm3d denoising.
        blk_idx: pre-defines region for applying denoisers.
        
    Returns:
        new estimate of projected images.
    """
    # take weighted average of input patches
    cmplx_img = patch2img(patch, patch_bounds, img_sz, img_wgt)
    
    # add regularzation
    if add_reg:
        if blk_idx is None:
            blk_idx = [0, img_sz[0], 0, img_sz[1]]
        # specify region
        tmp_img = cmplx_img[blk_idx[0]: blk_idx[1], blk_idx[2]: blk_idx[3]]
        # apply complex bm3d
        denoised_img = bm4d(tmp_img, bm3d_psd, profile=BM4DProfileBM3DComplex())[:, :, 0]
        cmplx_img[blk_idx[0]: blk_idx[1], blk_idx[2]: blk_idx[3]] = denoised_img

    # reallocate result
    new_patch = img2patch(cmplx_img, patch_bounds, patch.shape)

    return cmplx_img, new_patch


            
def search_offset(img, prb, patch_crd, given_meas):
    """
    The function to try the offset that matches current estimate with measurements.
    
    Args:
        img: current estimate of full-sized images.
        prb: current estimate of complex probe.
        patch_crd: current coordinates that describes the scan position.
        data_fit_prm: pre-processed ptychographic measurements.
        
    Returns:
        offset along x-axis, offset along y-axis. 
    """
    patch_bound = np.copy(patch_crd)
    meas = np.copy(given_meas)
    est_obj = np.copy(img)
    est_probe = np.copy(prb)

    x_offset = [-1, 0, 1]
    y_offset = [-1, 0, 1]
    # x_offset = [0]
    # y_offset = [0]
    offsets = product(x_offset, y_offset)
    offset_ls = []
    err_ls = []
    for offset in offsets:
        curr_offset = np.asarray(offset)
        offset_ls.append(curr_offset)
        shifted_patch, shifted_patch_crds = shift_position(patch_bound, curr_offset, est_obj)
        tmp_meas = np.abs(est_probe * shifted_patch)
        shift_err = np.linalg.norm(meas - phase_norm(np.copy(tmp_meas), meas))
        err_ls.append(shift_err)
    idx = np.array(err_ls).argmin()

    return offset_ls[idx]


def shift_position(img, patch_bound, offset=[0, 0]):
    """
    The function to shift the scan position and extract new patch from complex image. 
    
    Args:
        img: current estimate of full-sized images.
        patch_bound: current coordinates that describes the scan position of a patch.
        offset: amount of shifting current patch.
        
    Return:
        shifted patch, coordinates of the shifted patch.   
    """
    given_offset = np.copy(offset)
    est_obj = np.copy(img)
    patch_crd = np.copy(patch_bound)

    x_offset, y_offset = given_offset[0], given_offset[1]
    crd0, crd1, crd2, crd3 = patch_crd[0], patch_crd[1], patch_crd[2], patch_crd[3]
    patch_width, patch_height = patch_crd[1] - patch_crd[0], patch_crd[3] - patch_crd[2]

    if crd0 + x_offset < 0:
        shifted_crd0, shifted_crd1 = 0, patch_width
    elif crd1 + x_offset > est_obj.shape[0]:
        shifted_crd0, shifted_crd1 = est_obj.shape[0] - patch_width, est_obj.shape[0]
    else:
        shifted_crd0, shifted_crd1 = np.max([0, crd0 + x_offset]), np.min([crd1 + x_offset, est_obj.shape[0]])

    if crd2 + y_offset < 0:
        shifted_crd2, shifted_crd3 = 0, patch_height
    elif crd3 + y_offset > est_obj.shape[1]:
        shifted_crd2, shifted_crd3 = est_obj.shape[1] - patch_height, est_obj.shape[1]
    else:
        shifted_crd2, shifted_crd3 = np.max([0, crd2 + y_offset]), np.min([crd3 + y_offset, est_obj.shape[1]])
        
    output_patch = img[shifted_crd0:shifted_crd1, shifted_crd2:shifted_crd3]
    output_crds = [shifted_crd0, shifted_crd1, shifted_crd2, shifted_crd3]
        
    return output_patch, output_crds


def data_fit_op(cur_est, joint_est, y_meas, data_fit_prm, diff_intsty=None, est_intsty=None, mode_energy_coeff=None):
    r"""Data-fitting operator.

    The weighted proximal map operator :math:`F` is a stack of data-fitting agents,
    which revises estiamtes of complex patches or probe.

    Args:
        cur_est: current estimate of projected images or complex probe.
        joint_est: current estimate of complex probe or projected images.
        y_meas: pre-processed ptychographic measurements.
        data_fit_prm: prm/(1-prm) denotes noise-to-signal ratio of data.
        diff_intsty: difference between measured intensity data and estimated intensity value.
        est_intsty: estimated intensity.
        mode_energy_coeff: coefficients of data-fitting points associated with each probe mode.

    Returns:
        New estimates of projected image patches or complex probe.
    """
    # calculate closest data-fitting point
    if (diff_intsty is not None) and (est_intsty is not None) and (mode_energy_coeff is not None): 
        data_fit_pt = np.zeros_like(cur_est, dtype=np.complex64)
        probe_modes = np.copy(joint_est)
        for mode_idx, cur_mode in enumerate(probe_modes):
            # res_meas = np.sqrt(np.asarray(y_intsty - sum_intsty + est_intsty[mode_idx]).clip(0, None))
            res_meas = np.sqrt(np.asarray(diff_intsty + est_intsty[mode_idx]).clip(0, None))
            # # complex sqrt
            # res_meas = np.emath.sqrt(np.asarray(y_intsty - sum_intsty + est_intsty[mode_idx]))
            # w <- \sum_k F_{j, k}(v; w)
            data_fit_pt += mode_energy_coeff[mode_idx] * get_data_fit_pt(cur_est, cur_mode, res_meas)
    else:
        data_fit_pt = get_data_fit_pt(cur_est, joint_est, y_meas) 
        
    # take weighted average of current estimate and closest data-fitting point
    output = (1 - data_fit_prm) * cur_est + data_fit_prm * data_fit_pt

    return output


def pmace_recon(y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None,
                num_iter=100, joint_recon=False, recon_win=None, save_dir=None,
                obj_data_fit_prm=0.5, probe_data_fit_prm=0.5, 
                rho=0.5, probe_exp=1.5, obj_exp=0, add_reg=False, sigma=0.02,
                position_correction=False, add_mode=None, gamma=3):
    """Projected Multi-Agent Consensus Equilibrium.
    
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
        position_correction: option to refine scan positions.
        add_mode: the index of reconstruction iterations to add new probe modes.
        gamma: power parameter.
        
    Returns:
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
    nrmse_meas = []

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
    denoising_blk = recon_win
    dn_idx = np.nonzero(denoising_blk)
    crd0, crd1 = np.max([0, np.amin(dn_idx[0])]), np.min([np.amax(dn_idx[0]) + 1, est_obj.shape[0]])
    crd2, crd3 = np.max([0, np.amin(dn_idx[1])]), np.min([np.amax(dn_idx[1]) + 1, est_obj.shape[1]])
    blk_idx = [crd0, crd1, crd2, crd3]

    # To incorporate multi probe modes
    probe_modes = np.expand_dims(np.copy(est_probe), axis=0)  # [num_mode, mode_w, mode_h]
    y_intsty = y_meas ** 2
    probe_dict = {0: new_probe_arr}

    # start_time = time.time()
    print('{} recon starts ...'.format(approach))
    # PMACE reconstruction
    for i in tqdm(range(num_iter)):
        # w <- F(v; w)
        if len(probe_modes) > 1:
            est_intsty = [np.abs(compute_ft(np.copy(tmp_mode) * new_patch)) ** 2 for tmp_mode in probe_modes]
            sum_intsty = np.sum(est_intsty, axis=0)
            diff_intsty = y_intsty - sum_intsty
            mode_energy = [np.linalg.norm(tmp_mode) ** gamma for tmp_mode in probe_modes]
            energy_coeff = mode_energy / np.sum(mode_energy, axis=0)
            cur_patch = data_fit_op(new_patch, probe_modes, y_meas, obj_data_fit_prm, diff_intsty, est_intsty, energy_coeff)
        else:
            cur_patch = data_fit_op(new_patch, probe_modes[0], y_meas, obj_data_fit_prm)

        # z <- G(2w - v)
        est_obj, consens_patch = consens_op((2 * cur_patch - new_patch) * patch_weight, patch_bounds, img_wgt=image_weight,
                                            img_sz=image_sz, add_reg=add_reg, bm3d_psd=sigma, blk_idx=blk_idx)

        # v <- v + 2 \rho (z - w)
        new_patch = new_patch + 2 * rho * (consens_patch - cur_patch)
        
        # obtain estimate of complex object
        if not add_reg:
            est_obj = patch2img(new_patch * patch_weight, patch_bounds, image_sz, image_weight)

        if joint_recon:
            est_intsty = [np.abs(compute_ft(tmp_mode * consens_patch)) ** 2 for tmp_mode in probe_modes]
            sum_intsty = np.sum(est_intsty, axis=0)
            for mode_idx, cur_mode in enumerate(probe_modes):
                # w <- F(v; w)
                res_meas = np.sqrt(np.asarray(y_intsty - sum_intsty + est_intsty[mode_idx]).clip(0, None))
                new_probe_arr = probe_dict[mode_idx]
                cur_probe_arr = data_fit_op(new_probe_arr, consens_patch, res_meas, probe_data_fit_prm)

                # z <- G(2w - v)
                consens_probe = np.average((2 * cur_probe_arr - new_probe_arr), axis=0)
                # v <- v + 2 \rho (z - w)
                new_probe_arr = new_probe_arr + 2 * rho * (consens_probe - cur_probe_arr)

                # update probe modes
                probe_modes[mode_idx] = consens_probe
                probe_dict[mode_idx] = new_probe_arr

            # update image weights
            patch_weight = np.sum(np.abs(probe_modes) ** probe_exp, axis=0)
            image_weight = patch2img([patch_weight] * len(y_meas), patch_bounds, image_sz)

            if add_mode:
                if i + 1 in add_mode:
                    est_intsty = [np.abs(compute_ft(tmp_mode * new_patch)) ** 2 for tmp_mode in probe_modes]
                    sum_intsty = np.sum(est_intsty, axis=0)
                    # clip-to-zero strategy
                    res_meas = np.sqrt(np.asarray(y_intsty - sum_intsty).clip(0, None))
                    # back propagation residual meas to get new probe mode
                    new_probe_arr = np.asarray(divide_cmplx_numbers(compute_ift(res_meas), new_patch))
                    new_probe_mode = np.average(new_probe_arr, axis=0)
                    # update probe_dict and probe_modes
                    probe_dict[len(probe_modes)] = new_probe_arr
                    probe_modes = np.concatenate((probe_modes, np.expand_dims(np.copy(new_mode), axis=0)), axis=0)
                    
        # phase normalization and scale image to minimize the intensity difference
        # TODO: compare probe modes with gt probe
        if ref_obj is not None:
            revy_obj = phase_norm(np.copy(est_obj) * recon_win, ref_obj * recon_win, cstr=recon_win)
            err_obj = compute_nrmse(revy_obj * recon_win, ref_obj * recon_win, cstr=recon_win)
            nrmse_obj.append(err_obj)
        else:
            revy_obj = est_obj
        # if joint_recon:
        #     if ref_probe is not None:
        #         revy_probe = phase_norm(np.copy(est_probe), ref_probe)
        #         err_probe = compute_nrmse(revy_probe, ref_probe)
        #         nrmse_probe.append(err_probe)
        #     else:
        #         revy_probe = est_probe
        # else:
        #     revy_probe = est_probe

        # calculate error in measurement domain
        est_patch = img2patch(est_obj, patch_bounds, y_meas.shape).astype(cdtype)
        est_meas = np.sum([np.abs(compute_ft(tmp_mode * consens_patch)) for tmp_mode in probe_modes], axis=0)
        nrmse_meas.append(compute_nrmse(est_meas, y_meas))

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
            for mode_idx, cur_mode in enumerate(probe_modes):
                save_tiff(cur_mode, save_dir + 'probe_est_iter_{}_mode_{}.tiff'.format(i + 1, mode_idx))
            if nrmse_probe:
                save_array(nrmse_probe, save_dir + 'nrmse_probe_' + str(nrmse_probe[-1]))

    # return recon results
    print('{} recon completed.'.format(approach))
    keys = ['object', 'probe', 'err_obj', 'err_probe', 'err_meas']
    vals = [revy_obj, probe_modes, nrmse_obj, nrmse_probe, nrmse_meas]
    output = dict(zip(keys, vals))

    return output
    


