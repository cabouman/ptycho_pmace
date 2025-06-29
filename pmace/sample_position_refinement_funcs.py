import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from multiprocessing import Pool
from itertools import product
from .utils import *


def forward_propagation(cmplx_img, cmplx_probe, patch_crd, curr_meas, x_ofs=0, y_ofs=0):
    """
    Perform forward propagation of complex-valued image patch with a shifted scan location and 
    compare it to the measurement.

    Args:
        cmplx_img (numpy.ndarray): Full-sized complex transmittance image.
        cmplx_probe (numpy.ndarray): Complex-valued probe function.
        patch_crd (numpy.ndarray): Coordinates that determine the current scan location.
        curr_meas (numpy.ndarray): Current measurement associated with the patch_crd.
        x_ofs (int): Apply an offset to the scan location along the horizontal axis (default is 0).
        y_ofs (int): Apply an offset to the scan location along the vertical axis (default is 0).

    Returns:
        float: Frobenius norm between the forward-propagated patch and the collected measurement.
    """
    # Scan location offset
    curr_offset = [x_ofs, y_ofs]
    
    # Shift the image patch to the new location
    shifted_patch, shifted_patch_crds = shift_position(cmplx_img, patch_crd, curr_offset)
    
    # Compute absolute value of the Fourier transform of the probe multiplied by the shifted patch
    tmp_meas = np.abs(compute_ft(cmplx_probe * shifted_patch, threads=1))
    
    # Calculate the Frobenius norm between the current measurement and the computed measurement
    err_meas = np.linalg.norm(curr_meas - tmp_meas)
    
    return err_meas

    
def search_offset(cmplx_img, cmplx_probe, patch_crd, curr_meas, step_sz, grid_size=5):
    """
    Search for the offset within a grid, with each neighboring point being step_sz apart.

    Args:
        cmplx_img (numpy.ndarray): Full-sized complex transmission image.
        cmplx_probe (numpy.ndarray): Complex-valued probe function.
        patch_crd (tuple): Coordinates that determine the current scan location (e.g., (x, y)).
        curr_meas (float): Current measurement associated with the patch_crd.
        step_sz (int): Predefined value that determines the size of the search grid.
        grid_size (int): The size of the search grid (default is 3 for a 3x3 grid).
        
    Returns:
        list: [offset along x-axis, offset along y-axis]. 
    """
    partial_function = partial(forward_propagation, cmplx_img, cmplx_probe, patch_crd, curr_meas)
    
    # Define the grid of offsets based on the specified grid size
    half_grid = grid_size // 2
    x_offset = np.arange(-half_grid * step_sz, (half_grid + 1) * step_sz, step_sz)
    y_offset = np.arange(-half_grid * step_sz, (half_grid + 1) * step_sz, step_sz)
    
    # Initialize an array to store the function values
    fval = np.zeros((1, len(x_offset) * len(y_offset)))
    
    # Generate all possible combinations of offsets
    offsets = product(x_offset, y_offset)
    
    # Use multiprocessing to parallelize the computation
    with Pool(processes=mp.cpu_count()) as p:
        fval = p.starmap(partial_function, offsets)
        p.close()
        p.join()

    # Reshape the computed function values into a grid
    fvmx = np.reshape(fval, [len(x_offset), len(y_offset)])
    
    # Find the index of the minimum value in the grid
    sidx = np.unravel_index(np.nanargmin(fvmx), fvmx.shape)

    return [x_offset[sidx[0]], y_offset[sidx[1]]]
    

def shift_position(cmplx_img, patch_crds, offset=[0, 0]):
    """
    Shifts the scan position and extracts a new patch from a complex image.
    
    Args:
        cmplx_img (numpy.ndarray): Current estimate of a full-sized complex image.
        patch_crds (list): A list of four coordinates [crd0, crd1, crd2, crd3] that describe the scan position of a patch.
        offset (list): A list specifying the horizontal and vertical shift to apply to the current patch position.
            Defaults to [0, 0].
        
    Returns:
        numpy.ndarray: The shifted patch.
        list: The resulting coordinates [shifted_crd0, shifted_crd1, shifted_crd2, shifted_crd3].
    """
    
    # Unpack patch coordinates
    crd0, crd1, crd2, crd3 = patch_crds
    
    # Calculate patch height and width
    patch_h, patch_w = crd1 - crd0, crd3 - crd2
    
    # Corresponding offsets
    h_ofs, w_ofs = round(offset[0]), round(offset[1])
    
    # Calculate new vertical coordinates for the patch
    if crd0 + h_ofs < 0:
        shifted_crd0, shifted_crd1 = 0, patch_h
    elif crd1 + h_ofs > cmplx_img.shape[0]:
        shifted_crd0, shifted_crd1 = cmplx_img.shape[0] - patch_h, cmplx_img.shape[0]
    else:
        shifted_crd0, shifted_crd1 = np.max([0, crd0 + h_ofs]), np.min([crd1 + h_ofs, cmplx_img.shape[0]])
        
    # Calculate new horizontal coordinates for the patch
    if crd2 + w_ofs < 0:
        shifted_crd2, shifted_crd3 = 0, patch_w
    elif crd3 + w_ofs > cmplx_img.shape[1]:
        shifted_crd2, shifted_crd3 = cmplx_img.shape[1] - patch_w, cmplx_img.shape[1]
    else:
        shifted_crd2, shifted_crd3 = np.max([0, crd2 + w_ofs]), np.min([crd3 + w_ofs, cmplx_img.shape[1]])
    
    # Extract the shifted patch from the complex image
    output_patch = cmplx_img[shifted_crd0:shifted_crd1, shifted_crd2:shifted_crd3]
    
    # Create a list of the resulting coordinates
    output_crds = [shifted_crd0, shifted_crd1, shifted_crd2, shifted_crd3]
        
    return output_patch, output_crds 