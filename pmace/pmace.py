import pymp
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from bm4d import bm4d, BM4DProfile, BM4DStages, BM4DProfile2D, BM4DProfileComplex, BM4DProfileBM3DComplex
from functools import partial
from .utils import *
from .nrmse import *
from .scan_refinement_funcs import *


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
        cur_est (array-like): Current estimate of projected images or complex probe.
        joint_est (array-like): Current estimate of complex probe or projected images.
        y_meas (array-like): Pre-processed ptychographic measurements.
        
    Returns:
        array-like: Closest data-fitting point.
    """
    # Calculate DFT{D * P_j * v}
    f = compute_ft(cur_est * joint_est)
    
    # Calculate IDFT{y * DFT{D * P_j * v} / | DFT{D * P_j * v} |}
    inv_f = compute_ift(y_meas * np.exp(1j * np.angle(f)))
    
    # Calculate closest data-fitting point
    output = divide_cmplx_numbers(inv_f, joint_est)

    return output


def consens_op(patch, patch_bounds, img_sz, img_wgt, add_reg=False, bm3d_psd=0.1, blk_idx=None):
    r"""Consensus operator.

    The consensus operator :math:`G` takes weighted average of projections and 
    reallocates the results.
    
    Args:
        patch (numpy.ndarray): Current estimate of image patches.
        patch_bounds (list): Scan coordinates of projections.
        img_sz (tuple): Full image size (height, width).
        img_wgt (float): Image weight.
        add_reg (bool, optional): Option to apply denoising regularization. Default is False.
        bm3d_psd (float, optional): Power spectral density (PSD) of complex BM3D denoising. Default is 0.1.
        blk_idx (list, optional): Pre-defined region for applying denoisers. 
    
    Returns:
        A tuple containing: 
            - cmplx_img (numpy.ndarray): New estimate of projected images as a complex numpy array.
            - new_patch (numpy.ndarray): New estimate of image patches.
    """
    # Take weighted average of input patches
    cmplx_img = patch2img(patch, patch_bounds, img_sz, img_wgt)
    
    # add regularzation
    if add_reg:
        if blk_idx is None:
            blk_idx = [0, img_sz[0], 0, img_sz[1]]
        # Specify the region
        tmp_img = cmplx_img[blk_idx[0]: blk_idx[1], blk_idx[2]: blk_idx[3]]
        # Apply complex BM3D denoising
        denoised_img = bm4d(tmp_img, bm3d_psd, profile=BM4DProfileBM3DComplex())[:, :, 0]
        cmplx_img[blk_idx[0]: blk_idx[1], blk_idx[2]: blk_idx[3]] = denoised_img

    # Reallocate result
    new_patch = img2patch(cmplx_img, patch_bounds, patch.shape)

    return cmplx_img, new_patch


def object_data_fit_op(cur_est, joint_est, y_meas, data_fit_prm, diff_intsty=None, est_intsty=None, mode_energy_coeff=None):
    r"""Data-fitting operator.

    The weighted proximal map operator :math:`F` is a stack of data-fitting agents,
    which revises estiamtes of complex patches or probe.

    Args:
        cur_est (numpy.ndarray): Current estimate of projected images or complex probe.
        joint_est (list of numpy.ndarray): Current estimate of complex probe or projected images.
        y_meas (numpy.ndarray): Pre-processed ptychographic measurements.
        data_fit_prm (float): Parameter associated with noise-to-signal ratio of the data.
        diff_intsty (list of numpy.ndarray): Difference between measured intensity data and estimated intensity value.
        est_intsty (list of numpy.ndarray): Estimated intensity.
        mode_energy_coeff (list of float): Coefficients of data-fitting points associated with each probe mode.
    
    Returns:
        numpy.ndarray: New estimates of projected image patches or complex probe.
    """
    # start_time = time.time()
    
    # Initialize an array for storing the output estimates
    output = pymp.shared.array(cur_est.shape, dtype='cfloat')
    
    # Check if there are multiple probe modes
    if len(joint_est) > 1:
        # Parallel processing with the number of CPU logical cores
        with pymp.Parallel(psutil.cpu_count(logical=True)) as p:
            for idx in p.range(len(cur_est)):
                output[idx] = (1 - data_fit_prm) * cur_est[idx]
                for mode_idx, cur_mode in enumerate(joint_est):
                    # Calculate the residual measurements
                    res_meas = np.sqrt(np.asarray(diff_intsty[idx] + est_intsty[mode_idx][idx]).clip(0, None))
                    data_fit_pt = get_data_fit_pt(cur_est[idx], cur_mode, res_meas)
                    output[idx] += data_fit_pt * mode_energy_coeff[mode_idx]
    else:
        # Parallel processing with the number of CPU logical cores
        with pymp.Parallel(psutil.cpu_count(logical=True)) as p:
            for idx in p.range(len(cur_est)):
                # Calculate data-fitting point
                data_fit_pt = get_data_fit_pt(cur_est[idx], joint_est[0], y_meas[idx])
                output[idx] = (1 - data_fit_prm) * cur_est[idx] + data_fit_prm * data_fit_pt
                
    # print(time.time() - start_time)
    
    return output


def probe_data_fit_op(cur_est, joint_est, y_meas, data_fit_prm):
    r"""Data-fitting operator.

    The weighted proximal map operator :math:`F` is a stack of data-fitting agents,
    which revises estiamtes of complex patches or probe.

    Args:
        cur_est (array): Current estimate of projected images or complex probe.
        joint_est (array): Current estimate of complex probe or projected images.
        y_meas (array): Pre-processed ptychographic measurements.
        data_fit_prm (float): Weighting parameter representing the noise-to-signal ratio of the data.

    Returns:
        array: New estimates of projected image patches or complex probe.
    """
    # start_time = time.time()   
    
    # Create an output array to store the results
    output = pymp.shared.array(cur_est.shape, dtype='cfloat')
    
    # Use parallel processing to update estimates
    with pymp.Parallel(8) as p:
        # for idx in p.iterate(p.range(len(cur_est))):
        for idx in p.range(len(cur_est)):
            # Calculate the data-fitting point for the current index
            data_fit_pt = get_data_fit_pt(cur_est[idx], joint_est[idx], y_meas[idx])
            
            # Update the estimate using the data-fitting parameter
            output[idx] = (1 - data_fit_prm) * cur_est[idx] + data_fit_prm * data_fit_pt
            
    # print(time.time() - start_time)

    return output


def pmace_recon(y_meas, patch_bounds, init_obj, init_probe=None, ref_obj=None, ref_probe=None,
                num_iter=100, joint_recon=False, recon_win=None, save_dir=None,
                obj_data_fit_prm=0.5, probe_data_fit_prm=0.5, 
                rho=0.5, probe_exp=1.5, obj_exp=0, add_reg=False, sigma=0.02,
                scan_loc_refinement_iterations=[], scan_loc_search_step_sz=2, scan_loc_refine_step_sz=1, gt_scan_loc=None,
                add_mode=None, gamma=2):
    """Projected Multi-Agent Consensus Equilibrium (PMACE).
    
    Args:
        y_meas (numpy.ndarray): Pre-processed measurements (diffraction patterns / intensity data).
        patch_bounds (list): Scan coordinates of projections.
        init_obj (numpy.ndarray): Formulated initial guess of the complex object.
        init_probe (numpy.ndarray): Formulated initial guess of the complex probe.
        ref_obj (numpy.ndarray): Complex reference image for the object.
        ref_probe (numpy.ndarray): Complex reference image for the probe.
        num_iter (int): Number of iterations.
        joint_recon (bool): Option to estimate the complex probe for blind ptychography.
        recon_win (numpy.ndarray): Pre-defined window for showing and comparing reconstruction results.
        save_dir (str): Directory to save reconstruction results.
        obj_data_fit_prm (float): Averaging weight in the object update function. Parameter near 1 gives a closer fit to data.
        probe_data_fit_prm (float): Averaging weight in the probe update function.
        rho (float): Mann averaging parameter.
        probe_exp (float): Exponent of probe weighting in the consensus calculation of the image estimate.
        obj_exp (float): Exponent of image weighting in the consensus calculation of the probe estimate.
        add_reg (bool): Option to apply denoiser.
        sigma (float): Denoising parameter.
        scan_loc_refinement_iterations (list): List of iterations when scan location refinement is applied.
        scan_loc_search_step_sz (int): Step size for scan location search.
        scan_loc_refine_step_sz (float): Step size for refining scan locations.
        gt_scan_loc (numpy.ndarray): Ground truth scan locations.
        add_mode (list): The index of reconstruction iterations to add new probe modes.
        gamma (int): Power parameter for energy weighting.
        
    Returns:
        dict: Reconstructed complex images and NRMSE between reconstructions and reference images.
            Keys:
                - 'object' (numpy.ndarray): Reconstructed complex object.
                - 'probe' (numpy.ndarray): Reconstructed complex probe.
                - 'err_obj' (list): NRMSE values for the object reconstructions.
                - 'err_probe' (list): NRMSE values for the probe reconstructions (if joint_recon is True).
                - 'err_meas' (list): NRMSE values for the measured data.
    """
    approach = 'reg-PMACE' if add_reg else 'PMACE'
    cdtype = np.complex64
    
    # Check if the save directory exists and create it if not.
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Initialization
    if recon_win is None:
        recon_win = np.ones_like(init_obj)

    nrmse_obj = []
    nrmse_probe = []
    nrmse_meas = []

    est_obj = np.asarray(init_obj, dtype=cdtype)
    cur_patch = img2patch(est_obj, patch_bounds, y_meas.shape).astype(cdtype)
    consens_patch = np.copy(cur_patch)
    new_patch = np.copy(cur_patch)

    est_probe = np.asarray(init_probe, dtype=cdtype) if joint_recon else ref_probe.astype(cdtype)
    consens_probe = np.copy(est_probe).astype(cdtype)
    cur_probe_arr = [est_probe] * len(y_meas)
    new_probe_arr = np.copy(cur_probe_arr)

    # Calculate image weight and patch weight
    image_sz = est_obj.shape
    patch_weight = np.abs(consens_probe) ** probe_exp
    image_weight = patch2img([patch_weight] * len(y_meas), patch_bounds, image_sz)

    # Determine the area for applying denoiser
    denoising_blk = recon_win
    dn_idx = np.nonzero(denoising_blk)
    crd0, crd1 = np.max([0, np.amin(dn_idx[0])]), np.min([np.amax(dn_idx[0]) + 1, est_obj.shape[0]])
    crd2, crd3 = np.max([0, np.amin(dn_idx[1])]), np.min([np.amax(dn_idx[1]) + 1, est_obj.shape[1]])
    blk_idx = [crd0, crd1, crd2, crd3]

    # To incorporate multiple probe modes
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
            cur_patch = object_data_fit_op(new_patch, probe_modes, y_meas, obj_data_fit_prm, diff_intsty, est_intsty, energy_coeff)
        else:
            cur_patch = object_data_fit_op(new_patch, probe_modes, y_meas, obj_data_fit_prm)

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
            diff_intsty = y_intsty - sum_intsty
            for mode_idx, cur_mode in enumerate(probe_modes):
                # w <- F(v; w)
                res_meas = np.sqrt(np.asarray(diff_intsty + est_intsty[mode_idx]).clip(0, None))
                new_probe_arr = probe_dict[mode_idx]
                cur_probe_arr = probe_data_fit_op(new_probe_arr, consens_patch, res_meas, probe_data_fit_prm)

                # z <- G(2w - v)
                consens_probe = np.average((2 * cur_probe_arr - new_probe_arr), axis=0)
                # v <- v + 2 \rho (z - w)
                new_probe_arr = new_probe_arr + 2 * rho * (consens_probe - cur_probe_arr)

                # Update probe modes
                probe_modes[mode_idx] = consens_probe
                probe_dict[mode_idx] = new_probe_arr

            # Update image weights
            patch_weight = np.sum(np.abs(probe_modes) ** probe_exp, axis=0)
            image_weight = patch2img([patch_weight] * len(y_meas), patch_bounds, image_sz)

            if add_mode:
                if i + 1 in add_mode:
                    est_intsty = [np.abs(compute_ft(tmp_mode * consens_patch)) ** 2 for tmp_mode in probe_modes]
                    sum_intsty = np.sum(est_intsty, axis=0)
                    # Clip-to-zero strategy
                    res_meas = np.sqrt(np.asarray(y_intsty - sum_intsty).clip(0, None))
                    # Back propagation residual meas to get new probe mode
                    new_probe_arr = np.asarray(divide_cmplx_numbers(compute_ift(res_meas), consens_patch))
                    new_probe_mode = np.average(new_probe_arr, axis=0)
                    # Update probe_dict and probe_modes
                    probe_dict[len(probe_modes)] = new_probe_arr
                    probe_modes = np.concatenate((probe_modes, np.expand_dims(np.copy(new_mode), axis=0)), axis=0)
        
        # Check if the current iteration requires scan location refinement
        if i + 1 in scan_loc_refinement_iterations:
            # Initialize new scan locations
            scan_loc = np.zeros((len(patch_bounds), 2))
            reinitialization = False
            
            for curr_idx in range(len(patch_bounds)):
                curr_offset = search_offset(est_obj, est_probe, patch_bounds[curr_idx], y_meas[curr_idx], step_sz=scan_loc_search_step_sz)
                is_origin = np.array_equal(np.asarray(curr_offset), np.array([0, 0]))

                if not is_origin:
                    scan_loc[curr_idx, 1] = (patch_bounds[curr_idx, 2] + patch_bounds[curr_idx, 3]) / 2
                    scan_loc[curr_idx, 0] = (patch_bounds[curr_idx, 0] + patch_bounds[curr_idx, 1]) / 2
                    print('curr_iter =', i + 1, 'curr_idx =', curr_idx,  'gt_scan_loc = ', np.round(gt_scan_loc[curr_idx]), 'curr_scan_loc =', scan_loc[curr_idx], 'true offset =', gt_scan_loc[curr_idx] - scan_loc[curr_idx])
                    print('offset = ', curr_offset)
                    if curr_idx == 73:
                        true_offset = gt_scan_loc[curr_idx] - scan_loc[curr_idx]
                        plot_heap_map(est_obj, est_probe, patch_bounds[curr_idx], y_meas[curr_idx], step_sz=scan_loc_search_step_sz, save_dir=save_dir + 'heap_map_patch_idx_{}_iter_{}.png'.format(curr_idx, i+1), true_offset=true_offset)

                    new_patch[curr_idx], patch_bounds[curr_idx] = shift_position(est_obj, patch_bounds[curr_idx], offset=scan_loc_refine_step_sz * curr_offset)
                else:
                    reinitialization = True

                scan_loc[curr_idx, 1] = (patch_bounds[curr_idx, 2] + patch_bounds[curr_idx, 3]) / 2
                scan_loc[curr_idx, 0] = (patch_bounds[curr_idx, 0] + patch_bounds[curr_idx, 1]) / 2

                if not is_origin:
                    print('offset = ', curr_offset, 'curr_scan_loc =', scan_loc[curr_idx], 'true offset =', gt_scan_loc[curr_idx] - scan_loc[curr_idx])
            
            # Reinitialization
            if reinitialization:
                patch_weight = np.abs(consens_probe) ** probe_exp
                image_weight = patch2img([patch_weight] * len(y_meas), patch_bounds, image_sz)
                cur_patch = np.copy(new_patch)
                est_obj = patch2img(new_patch * patch_weight, patch_bounds, image_sz, image_weight)

            # Update step size of search grid
            # scan_loc_search_step_sz = np.maximum(np.floor(scan_loc_search_step_sz / 2), 1)
            # scan_loc_search_step_sz = np.maximum(1, int(scan_loc_search_step_sz / ( 1 + i / 10) ))
            scan_loc_search_step_sz = np.maximum(1, scan_loc_search_step_sz-1)
            
            distance = np.linalg.norm(scan_loc - gt_scan_loc)
            print('Total distance = ', distance)
            avg_distance = np.average(np.sqrt(np.sum((scan_loc - gt_scan_loc) ** 2, axis=1)))
            print('Average distance =', avg_distance)
                
            if distance == 0:
                scan_loc_refinement_iterations = []
            else:
                # Save refined scan location
                compare_scan_loc(scan_loc, gt_scan_loc, save_dir=save_dir + 'iter_{}/'.format(i + 1))
                df = pd.DataFrame({'FCx': scan_loc[:, 0], 'FCy': scan_loc[:, 1]})
                df.to_csv(save_dir + 'iter_{}/revised_Translations.tsv.txt'.format(i + 1))
                 
        # Phase normalization and scale image to minimize the intensity difference
        if ref_obj is not None:
            revy_obj = phase_norm(np.copy(est_obj) * recon_win, ref_obj * recon_win, cstr=recon_win)
            err_obj = compute_nrmse(revy_obj * recon_win, ref_obj * recon_win, cstr=recon_win)
            nrmse_obj.append(err_obj)
        else:
            revy_obj = est_obj
            
        if joint_recon and (ref_probe is not None):
            if ref_probe.ndim > 2:
                err_probe = 0
                for mode_idx in range(np.minimum(len(probe_modes), len(ref_probe))):
                    probe_modes[mode_idx] = phase_norm(np.copy(probe_modes[mode_idx]), ref_probe[mode_idx])
                    err_probe += compute_nrmse(probe_modes[mode_idx], ref_probe[mode_idx])
                nrmse_probe.append(err_probe)
                revy_probe = probe_modes
            else:
                probe_modes[0] = phase_norm(np.copy(probe_modes[0]), ref_probe)
                err_probe = compute_nrmse(probe_modes[0], ref_probe)
                nrmse_probe.append(err_probe)
                revy_probe = probe_modes
        else:
            revy_probe = est_probe

        # calculate error in measurement domain
        est_patch = img2patch(est_obj, patch_bounds, y_meas.shape).astype(cdtype)
        est_meas = np.sum([np.abs(compute_ft(tmp_mode * est_patch)) for tmp_mode in probe_modes], axis=0)
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
    vals = [revy_obj, revy_probe, nrmse_obj, nrmse_probe, nrmse_meas]
    output = dict(zip(keys, vals))

    return output