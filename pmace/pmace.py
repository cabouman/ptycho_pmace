import pymp
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from bm4d import bm4d, BM4DProfile, BM4DStages, BM4DProfile2D, BM4DProfileComplex, BM4DProfileBM3DComplex
from functools import partial
from .utils import *
from .nrmse import *
from .sample_position_refinement_funcs import *
from .drift_correction_funcs import *
import scico.linop.optics as op


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
        cur_est (numpy.ndarray): Current estimate of projected images or complex probe.
        joint_est (numpy.ndarray): Current estimate of complex probe or projected images.
        y_meas (numpy.ndarray): Measurements.

    Returns:
        numpy.ndarray: Closest data-fitting point.
    """
    # Initialize an array for storing the output estimates
    output = pymp.shared.array(cur_est.shape, dtype='cfloat')
    
    # Parallel processing with the number of CPU logical cores
    with pymp.Parallel(psutil.cpu_count(logical=True)) as p:
        for idx in p.range(len(cur_est)):
            # Calculate DFT{D * P_j * v}
            tmp_f = compute_ft(cur_est[idx] * joint_est[idx])

            # Calculate IDFT{y * DFT{D * P_j * v} / | DFT{D * P_j * v} |}
            inv_f = compute_ift(y_meas[idx] * np.exp(1j * np.angle(tmp_f)))

            # Calculate closest data-fitting point
            output[idx] = divide_cmplx_numbers(inv_f, joint_est[idx])

    return output


def object_data_fit_op(projected_patches, probe_modes, y_intsty, data_fit_param, gamma=2):
    """
    Update the current patch using a weighted combination of current estimate and
    data-fitting points associated with each probe mode.

    Args:
        projected_patches (numpy.ndarray): Current estimate of projected patches.
        probe_modes (list of ndarrays): List of probe modes.
        y_intsty (numpy.ndarray): Measured intensity data.
        data_fit_param (float): data fitting parameter.
        gamma (float): Weighting exponent for probe modes.

    Returns:
        numpy.ndarray: Updated image patches.
        list: Mode weights for probe modes used in the update.
    """
    # Update patch using data fitting with probe modes and measured data
    output = (1 - data_fit_param) * projected_patches

    # Calculate estimated intensity for each probe mode
    est_intsty = [np.abs(compute_ft(np.copy(tmp_mode) * projected_patches)) ** 2 for tmp_mode in probe_modes]

    # Calculate the sum of estimated intensities
    sum_intsty = np.sum(est_intsty, axis=0)

    # Calculate the energy of each probe mode
    mode_energy = [np.linalg.norm(tmp_mode) ** gamma for tmp_mode in probe_modes]

    # Calculate the weight for each probe mode
    mode_weight = mode_energy / np.sum(mode_energy, axis=0)

    # Update the current patch using probe modes and measured data
    for mode_idx, cur_mode in enumerate(probe_modes):
        # res_meas = np.emath.sqrt(np.asarray(diff_intsty + est_intsty[mode_idx]))
        cur_meas = np.sqrt(y_intsty / (sum_intsty + 1e-6)) * np.abs(compute_ft(np.copy(cur_mode) * projected_patches))
        output += data_fit_param * mode_weight[mode_idx] * get_data_fit_pt(projected_patches, [cur_mode] * len(cur_meas), cur_meas)

    return output, mode_weight


def pixel_weighted_avg_op(projected_patches, probe_modes, mode_weights,
                          patch_bounds, image_sz, probe_exp=1,
                          regularization=False, bm3d_psd=0.1, blk_idx=None):
    """
    Calculate the pixel-weighted average of projected patches and reallocates the result.

    Args:
        projected_patches (numpy.ndarray): Array of projected patches.
        probe_modes (list of ndarrays): List of probe modes.
        mode_weights (list): list of mode weights.
        patch_bounds (numpy.ndarray): List of patch coordinates.
        image_sz (list): The dimensions of the output image.
        probe_exp (float, optional): Exponent for probe mode weight calculation. Default is 1.
        regularization (bool, optional): Apply regularization using BM3D denoising if True. Default is False.
        bm3d_psd (float, optional): Power spectral density parameter for BM3D denoising. Default is 0.1.
        blk_idx (list, optional): Block indices for BM3D denoising region. Default is None.

    Returns:
        ndarray: Resulting complex image.
        ndarray: MUpdated image patches.
    """
    # Initialize output complex image with zeros.
    output_img = np.zeros(image_sz, dtype=np.complex64)

    # Loop through each probe mode and apply weighted averaging.
    for mode_idx, cur_mode in enumerate(probe_modes):
        # Calculate weight for the patches based on current probe mode.
        patch_weight = np.abs(cur_mode) ** probe_exp
        # Compute weight for the full-sized image based on current probe mode.
        image_weight = patch2img([patch_weight] * len(projected_patches), patch_bounds, image_sz)
        # Apply weighted average accross different scan locations
        tmp_img = patch2img(projected_patches * patch_weight, patch_bounds, image_sz, image_weight)
        # Accumulate the result using the mode weight.
        output_img += mode_weights[mode_idx] * tmp_img

    # If regularization is enabled, apply BM3D denoising.
    if regularization:
        # Define the region for BM3D denoising.
        if blk_idx is None:
            blk_idx = [0, image_sz[0], 0, image_sz[1]]
        # Extract the region for denoising.
        tmp_img = output_img[blk_idx[0]: blk_idx[1], blk_idx[2]: blk_idx[3]]
        # Apply complex BM3D denoising to the specified region.
        denoised_img = bm4d(tmp_img, bm3d_psd, profile=BM4DProfileBM3DComplex())[:, :, 0]
        # Replace the denoised region in the output image.
        output_img[blk_idx[0]: blk_idx[1], blk_idx[2]: blk_idx[3]] = denoised_img

    # Convert the resulting complex image to patches.
    output_patch = img2patch(output_img, patch_bounds, projected_patches.shape)

    return output_img, output_patch


def add_probe_mode(probe_modes, projected_patches, y_intsty, probe_dict,
                   fresnel_propagation=False, dx=None, wavelength=None, propagation_dist=None):
    """
    Add a new probe mode to the existing list of probe modes and update the probe dictionary.

    Args:
        probe_modes (numpy.ndarray): Array of existing probe modes.
        projected_patches (numpy.ndarray): The estimates of projected patches.
        y_intsty (numpy.ndarray): The intensity measurement.

    Returns:
        numpy.ndarray: Updated array of probe modes with the newly added probe mode.
        dictionary: Updated dictionary of probe modes with the newly added probe mode.
    """
    # Calculate estimated intensity for each probe mode
    est_intsty = [np.abs(compute_ft(np.copy(tmp_mode) * projected_patches)) ** 2 for tmp_mode in probe_modes]

    # Calculate the sum of estimated intensities
    sum_intsty = np.sum(est_intsty, axis=0)

    # Clip-to-zero strategy
    res_meas = np.sqrt(np.asarray(y_intsty - sum_intsty).clip(0, None))

    # Backpropagate the residual measurement to get a new probe mode
    tmp_probe_arr = np.asarray(divide_cmplx_numbers(compute_ift(res_meas), projected_patches))
    tmp_probe_mode = np.average(tmp_probe_arr, axis=0)

    # Fresnel propagate new probe modes
    grid_size_0, grid_size_1 = tmp_probe_mode.shape[0], tmp_probe_mode.shape[1]
    if fresnel_propagation:
#         fres_op = op.FresnelPropagator(tuple([512, 512]), dx=4.52e-9, k0=2 * np.pi / 1.24e-9, z=5e-7)
        fres_op = op.FresnelPropagator(tuple([grid_size_0, grid_size_1]), dx=dx, k0=2 * np.pi / wavelength, z=propagation_dist)
        prop_probe_mode = fres_op(tmp_probe_mode)
        sqrt_avg_energy = np.linalg.norm(probe_modes)
        new_probe_mode = prop_probe_mode * sqrt_avg_energy / np.linalg.norm(prop_probe_mode)
        new_probe_arr = [new_probe_mode] * len(y_intsty)
    else:
        new_probe_mode = tmp_probe_mode
        new_probe_arr = tmp_probe_arr

    # Update probe_dict and probe_modes
    probe_dict[len(probe_modes)] = np.array(new_probe_arr)
    probe_modes = np.concatenate((probe_modes, np.expand_dims(np.copy(new_probe_mode), axis=0)), axis=0)

    # Scale probe_dict and probe_modes to ensure consistent probe energy 
    # ================================================================== TODO: balance energy ratio
    num_modes = len(probe_modes)
    for mode_idx, cur_mode in enumerate(probe_modes):
        cur_probe_arr = np.array(probe_dict[mode_idx])
        probe_modes[mode_idx] = cur_mode * (num_modes - 1) / num_modes
        probe_dict[mode_idx] = cur_probe_arr * (num_modes - 1) / num_modes

    return probe_modes, probe_dict


def determine_denoising_area(cmplx_img, predefined_mask=None):
    """
    Determine the area for applying a denoiser based on a binary mask.

    Args:
        cmplx_img (numpy.ndarray): Complex-valued image.
        predefined_mask (numpy.ndarray, optional): Binary mask representing the region of interest for denoising.
        If not provided, the entire image is considered.

    Returns:
        list: Row and column indices defining the denoising area [row_start, row_end, col_start, col_end].
    """
    # Initialize the predefined mask if not provided
    if predefined_mask is None:
        predefined_mask = np.ones_like(cmplx_img)

    # Find the indices of non-zero elements in the binary mask
    denoising_indices = np.nonzero(predefined_mask)

    # Calculate the denoising area boundaries
    row_start = max(0, denoising_indices[0].min())
    row_end = min(denoising_indices[0].max() + 1, cmplx_img.shape[0])
    col_start = max(0, denoising_indices[1].min())
    col_end = min(denoising_indices[1].max() + 1, cmplx_img.shape[1])

    # Define the denoising area using row and column indices
    denoising_area = [row_start, row_end, col_start, col_end]

    return denoising_area


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
                rho=0.5, probe_exp=1.5, obj_exp=0, add_reg=False, sigma=0.02, probe_center_correction=[],
                scan_loc_refinement_iterations=[], scan_loc_search_step_sz=2, scan_loc_refine_step_sz=1, gt_scan_loc=None,
                add_mode=[], img_px_sz=4.52e-9, wavelength=1.24e-9, propagation_dist=1e-7, gamma=2):
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

    # Determine the area for applying a denoiser.
    blk_idx = determine_denoising_area(init_obj, predefined_mask=recon_win)

    # Initialize error metrics
    nrmse_obj = []
    nrmse_probe = []
    nrmse_meas = []
    mse_meas = []
    mse_intsty = []

    # Initialize estimates with specific data type and creat current patches
    est_obj = np.array(init_obj, dtype=cdtype)
    new_patch = img2patch(est_obj, patch_bounds, y_meas.shape).astype(cdtype)
    image_sz = est_obj.shape
    
    # Expand the dimensions of reference probe
    if ref_probe is not None:
        if np.array(ref_probe).ndim == 2:
            ref_probe_modes = np.expand_dims(ref_probe.astype(cdtype), axis=0)              # [num_mode, mode_h, mode_w]
        elif np.array(ref_probe).ndim > 2:
            ref_probe_modes = ref_probe
    else:
        ref_probe_modes = None

    # Initialize the probe estimate and create a list of probe estimates
    y_intsty = y_meas ** 2
    if joint_recon:
        if np.array(init_probe).ndim == 2:
            probe_modes = np.expand_dims(init_probe, axis=0)
        elif np.array(init_probe).ndim == 3:
            probe_modes = np.array(init_probe, dtype=cdtype)

        probe_dict = {}
        for mode_idx, cur_mode in enumerate(probe_modes):
            new_probe_arr = np.array([cur_mode] * len(y_meas))
            probe_dict[mode_idx] = new_probe_arr
        # est_probe = np.asarray(init_probe, dtype=cdtype)                                        # [mode_h, mode_w]
        # consens_probe = np.copy(est_probe).astype(cdtype)                                       # [mode_h, mode_w]
        # new_probe_arr = np.asarray([est_probe] * len(y_meas))                                   # [num_meas, mode_h, mode_w]
        # probe_modes = np.expand_dims(np.copy(est_probe), axis=0)                                # [num_mode, mode_h, mode_w]
        # probe_dict = {0: new_probe_arr}                                                         # {mode_idx: mode_array}
    else:
        probe_modes = ref_probe_modes

    # start_time = time.time()
    
    print('{} recon starts ...'.format(approach))
    
    # PMACE reconstruction
    for i in tqdm(range(num_iter)):
        # Update the current patch using data fitting: w <- F(v; w)
        cur_patch, mode_weights = object_data_fit_op(new_patch, probe_modes, y_intsty, obj_data_fit_prm, gamma=gamma)

        # Obtain an estimate of the complex object using weighted averaging: z <- G(2w - v)
        est_obj, consens_patch = pixel_weighted_avg_op(2 * cur_patch - new_patch, probe_modes, mode_weights,
                                                       patch_bounds, image_sz, probe_exp=probe_exp,
                                                       regularization=add_reg, bm3d_psd=sigma, blk_idx=blk_idx)

        # Update the projected patch: v <- v + 2 \rho (z - w)
        new_patch = new_patch + 2 * rho * (consens_patch - cur_patch)

        # Obtain estimate of complex object using weighted averaging without regularization.
        if not add_reg:
            est_obj, _ = pixel_weighted_avg_op(new_patch, probe_modes, mode_weights, patch_bounds, image_sz, probe_exp=probe_exp)


        if joint_recon:
            # Add another mode
            if i + 1 in add_mode:
                probe_modes, probe_dict = add_probe_mode(probe_modes, consens_patch, y_intsty, probe_dict,
                                                         fresnel_propagation=True, dx=img_px_sz, wavelength=wavelength, propagation_dist=propagation_dist)
                
            # Calculate estimated intensity for each probe mode
            est_intsty = [np.abs(compute_ft(np.copy(tmp_mode) * consens_patch)) ** 2 for tmp_mode in probe_modes]

            # Calculate the sum of estimated intensities
            sum_intsty = np.sum(est_intsty, axis=0)

            # # Calculate the difference between measured intensity and the sum of estimated intensities
            # diff_intsty = y_intsty - sum_intsty

            # Calculate the energy of each probe mode
            test_mode_energy = [np.linalg.norm(tmp_mode) ** gamma for tmp_mode in probe_modes]

            # Calculate the weight for each probe mode
            test_mode_weight = test_mode_energy / np.sum(test_mode_energy, axis=0)
            
            # Loop through probe_modes to update each mode
            for mode_idx, cur_mode in enumerate(probe_modes):
                # # Calculate residual measurements
                # res_meas = np.sqrt(np.asarray(diff_intsty + est_intsty[mode_idx]).clip(0, None))
                res_meas = np.sqrt(y_intsty / (sum_intsty + 1e-6))  * np.abs(compute_ft(np.copy(cur_mode) * consens_patch))

                # Get the current probe data
                new_probe_arr = probe_dict[mode_idx]

                # Apply the probe data fitting operation: w <- F(v; w)
                # print(type(probe_data_fit_prm), type(new_probe_arr), type(get_data_fit_pt(new_probe_arr, consens_patch, res_meas)))
                cur_probe_arr = (1 - probe_data_fit_prm) * new_probe_arr + probe_data_fit_prm * get_data_fit_pt(new_probe_arr, consens_patch, res_meas)

                # Calculate the consensus probe: z <- G(2w - v)
                consens_probe = np.average((2 * cur_probe_arr - new_probe_arr), axis=0)

                # Update the probe data: v <- v + 2 * rho * (z - w)
                new_probe_arr = new_probe_arr + 2 * rho * (consens_probe - cur_probe_arr)

                # Update probe modes
                probe_modes[mode_idx] = consens_probe
                probe_dict[mode_idx] = new_probe_arr
                
            # Calculate the energy of each probe mode
            test_mode_energy = [np.linalg.norm(tmp_mode) ** gamma for tmp_mode in probe_modes]

            # Calculate the weight for each probe mode
            test_mode_weight = test_mode_energy / np.sum(test_mode_energy, axis=0)

            # +++++++++++++++++++++++++++++++++++++++++++ TOBECORRECTED
            est_probe = consens_probe

            # ======================================================== TODO: Probe center correction
            # ======================================================== TODO: Sample poisition refinement
            
        # Phase normalization and scale image to minimize the intensity difference
        if ref_obj is not None:
            revy_obj = phase_norm(np.copy(est_obj) * recon_win, ref_obj * recon_win, cstr=recon_win)
            err_obj = compute_nrmse(revy_obj * recon_win, ref_obj * recon_win, cstr=recon_win)
            nrmse_obj.append(err_obj)
        else:
            revy_obj = est_obj

        if joint_recon and (ref_probe_modes is not None):
            tmp_probe_err = 0
            for mode_idx in range(min(len(probe_modes), len(ref_probe_modes))):
                tmp_probe_mode = phase_norm(np.copy(probe_modes[mode_idx]), ref_probe_modes[mode_idx])
                tmp_probe_err += compute_nrmse(tmp_probe_mode, ref_probe_modes[mode_idx])
            nrmse_probe.append(tmp_probe_err)
            revy_probe = probe_modes
        else:
            revy_probe = probe_modes

        # calculate error in measurement domain
        est_patch = consens_patch
        est_meas = np.sqrt(np.sum([np.abs(compute_ft(tmp_mode * est_patch))**2 for _, tmp_mode in enumerate(probe_modes)], axis=0))
        nrmse_meas.append(compute_nrmse(est_meas, y_meas))
        mse_meas.append(compute_mse(est_meas, y_meas))
        
        if (i+1)%20 == 0:
            # SAVE INTER RESULT
            save_tiff(est_obj, save_dir + 'est_obj_iter_{}.tiff'.format(i + 1))
            for mode_idx, cur_mode in enumerate(probe_modes):
                save_tiff(cur_mode, save_dir + 'probe_est_iter_{}_mode_{}.tiff'.format(i + 1, mode_idx))
        

    # # calculate time consumption
    # print('Time consumption of {}:'.format(approach), time.time() - start_time)

    # save recon results
    if save_dir is not None:
        save_tiff(est_obj, save_dir + 'est_obj_iter_{}.tiff'.format(i + 1))
        if nrmse_obj:
            save_array(nrmse_obj, save_dir + 'nrmse_obj_' + str(nrmse_obj[-1]))
        if nrmse_meas:
            save_array(nrmse_meas, save_dir + 'nrmse_meas_' + str(nrmse_meas[-1]))
        if mse_meas:
            save_array(mse_meas, save_dir + 'mse_meas_' + str(mse_meas[-1]))
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