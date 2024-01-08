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
    tmp_meas = np.abs(compute_ft(cmplx_probe * shifted_patch))
    
    # Calculate the Frobenius norm between the current measurement and the computed measurement
    err_meas = np.linalg.norm(curr_meas - tmp_meas)
    
    return err_meas

    
def search_offset(cmplx_img, cmplx_probe, patch_crd, curr_meas, step_sz):
    """
    Search for the offset within a 3x3 grid, with each neighboring point being step_sz apart.

    Args:
        cmplx_img (numpy.ndarray): Full-sized complex transmission image.
        cmplx_probe (numpy.ndarray): Complex-valued probe function.
        patch_crd (tuple): Coordinates that determine the current scan location (e.g., (x, y)).
        curr_meas (float): Current measurement associated with the patch_crd.
        step_sz (int): Predefined value that determines the size of the search grid.
        
    Returns:
        list: [offset along x-axis, offset along y-axis]. 
    """
    partial_function = partial(forward_propagation, cmplx_img, cmplx_probe, patch_crd, curr_meas)
    
    # Define the 3x3 grid of offsets
    x_offset = [-step_sz, 0, step_sz]
    y_offset = [-step_sz, 0, step_sz]
    
    # Convert lists to numpy arrays
    x_offset = np.asarray(x_offset)
    y_offset = np.asarray(y_offset)
    
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


# ============================================================================================
# The following code has been added to check the results. TODO: remove following functions.
# ============================================================================================
def compare_result_with_ground_truth_img(cmplx_img, ref_img, display_win=None, save_dir=None,
                                         mag_vmax=1.1, mag_vmin=0.9, phase_vmax=0, phase_vmin=-0.8, 
                                         real_vmax=1.1, real_vmin=0.8, imag_vmax=0, imag_vmin=-0.6):
    """
    Compare a reconstructed image with a ground truth image and plot error images for various components.

    Args:
        cmplx_img (numpy.ndarray): Complex image array representing the reconstructed image.
        ref_img (numpy.ndarray or None): Complex image array representing the ground truth image. Use None if not available.
        display_win (numpy.ndarray or None): Optional window to display only specific regions. Use None for the entire image.
        save_dir (str or None): The directory where the plot images will be saved. Use None if no saving is required.
        mag_vmax (float): Maximum value for magnitude component visualization.
        mag_vmin (float): Minimum value for magnitude component visualization.
        phase_vmax (float): Maximum value for phase component visualization.
        phase_vmin (float): Minimum value for phase component visualization.
        real_vmax (float): Maximum value for real component visualization.
        real_vmin (float): Minimum value for real component visualization.
        imag_vmax (float): Maximum value for imaginary component visualization.
        imag_vmin (float): Minimum value for imaginary component visualization.

    This function compares the reconstructed complex image with a ground truth image if available and generates a set of plots for visualizing the differences. The plots include:
    - Magnitude of the ground truth and reconstructed images.
    - Phase of the ground truth and reconstructed images.
    - Real part of the ground truth and reconstructed images.
    - Imaginary part of the ground truth and reconstructed images.
    - Amplitude of the difference between the reconstructed and ground truth images.
    - Phase difference between the reconstructed and ground truth images.
    - Real part of the error between the reconstructed and ground truth images.
    - Imaginary part of the error between the reconstructed and ground truth images.

    The plots will be saved in the specified 'save_dir' directory, if a 'save_dir' is specified.
    """
    # Initialize window and determine area for showing and comparing images
    if display_win is None:
        display_win = np.ones_like(cmplx_img, dtype=np.complex128)
    non_zero_idx = np.nonzero(display_win)
    blk_idx = [np.amin(non_zero_idx[0]), np.amax(non_zero_idx[0])+1, np.amin(non_zero_idx[1]), np.amax(non_zero_idx[1])+1]
    cmplx_img_rgn = cmplx_img[blk_idx[0]:blk_idx[1], blk_idx[2]:blk_idx[3]]
    
    # Prepare reference image
    if ref_img is not None:
        ref_img_rgn = ref_img[blk_idx[0]:blk_idx[1], blk_idx[2]:blk_idx[3]]
        
    # Phase normalization
    cmplx_img_rgn = phase_norm(cmplx_img_rgn, ref_img_rgn)
    plt.figure(num=None, figsize=(10, 6), dpi=400, facecolor='w', edgecolor='k')
    # Mag of reference image
    img_title = 'GT'
    plt.subplot(3, 4, 1)
    plt.imshow(np.abs(ref_img_rgn), cmap='gray', vmax=mag_vmax, vmin=mag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'mag of {}'.format(img_title))
    # Phase of reference image
    plt.subplot(3, 4, 2)
    plt.imshow(np.angle(ref_img_rgn), cmap='gray', vmax=phase_vmax, vmin=phase_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'phase of {}'.format(img_title))
    # Real part of reference image
    plt.subplot(3, 4, 3)
    plt.imshow(np.real(ref_img_rgn), cmap='gray', vmax=real_vmax, vmin=real_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('real of {}'.format(img_title))
    # Imag part of reference image
    plt.subplot(3, 4, 4)
    plt.imshow(np.imag(ref_img_rgn), cmap='gray', vmax=imag_vmax, vmin=imag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('imag of {}'.format(img_title))
    
    # Amplitude of reconstructed complex image
    img_title = 'recon'
    plt.subplot(3, 4, 5)
    plt.imshow(np.abs(cmplx_img_rgn), cmap='gray', vmax=mag_vmax, vmin=mag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'mag of {}'.format(img_title))
    # Phase of reconstructed complex image
    plt.subplot(3, 4, 6)
    plt.imshow(np.angle(cmplx_img_rgn), cmap='gray', vmax=phase_vmax, vmin=phase_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'phase of {}'.format(img_title))
    # Real part of reconstructed complex image
    plt.subplot(3, 4, 7)
    plt.imshow(np.real(cmplx_img_rgn), cmap='gray', vmax=real_vmax, vmin=real_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('real of {}'.format(img_title))
    # Imag part of reconstructed complex image
    plt.subplot(3, 4, 8)
    plt.imshow(np.imag(cmplx_img_rgn), cmap='gray', vmax=imag_vmax, vmin=imag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('imag of {}'.format(img_title))
    
    # Mmplitude of difference between complex reconstruction and ground truth image
    plt.subplot(3, 4, 9)
    plt.imshow(np.abs(cmplx_img_rgn - ref_img_rgn), cmap='gray', vmax=0.2, vmin=0)
    plt.title(r'error - amp')
    plt.colorbar()
    plt.axis('off')
    # Phase difference between complex reconstruction and ground truth image
    ang_err = pha_err(cmplx_img_rgn, ref_img_rgn)
    plt.subplot(3, 4, 10)
    plt.imshow(ang_err, cmap='gray' , vmax=0.2, vmin=-0.2)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'phase error')
    # Real part of error image between complex reconstruction and ground truth image
    err = cmplx_img_rgn - ref_img_rgn
    plt.subplot(3, 4, 11)
    plt.imshow(np.real(err), cmap='gray' , vmax=0.2, vmin=-0.2)
    plt.colorbar()
    plt.axis('off')
    plt.title('err - real')
    # Image part of error between complex reconstruction and ground truth image
    plt.subplot(3, 4, 12)
    plt.imshow(np.imag(err), cmap='gray' , vmax=0.2, vmin=-0.2)
    plt.colorbar()
    plt.axis('off')
    plt.title('err - imag')
    
    plt.savefig(save_dir + 'reconstructed_img.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.clf()

    
def compare_scan_loc(curr_scan_loc, ground_truth_scan_loc, save_dir=None):
    """
    Compare the current scan location with the ground truth scan location and plot a histogram of offsets.

    Args:
        curr_scan_loc (array-like): Current scan location data as an array.
        ground_truth_scan_loc (array-like): Ground truth scan location data as an array.
        save_dir (str): The directory where the plot images will be saved.

    This function visualizes and compares scan locations, both the ground truth and the current scan locations.
    It creates two plots: one displaying the locations in blue (ground truth) and red (current), and another
    showing the histogram of the horizontal and vertical offsets between corresponding points.
    The plots are saved as 'plot_scan_loc.png' and 'scan_loc_offset_hist.png' in the specified directory.

    Example:
    compare_scan_loc(curr_scan_loc, ground_truth_scan_loc, save_dir='/path/to/save/directory/')
    """
    # Initialization
    array1 = ground_truth_scan_loc
    array2 = curr_scan_loc
    # check directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # ===================== Plot ground truth and current scan locations =======================
    # Plot ground truth locations in blue
    plt.figure(figsize=[5, 8])
    plt.subplot(211)
    plt.plot(array1[:, 0], array1[:, 1], 'b.', markersize=3)
    plt.plot(array1[:, 0], array1[:, 1], 'b-', linewidth=0.5)
    plt.grid()
    
    # Plot current scan locations in red
    plt.subplot(212)
    plt.plot(array2[:, 0], array2[:, 1], 'r.', markersize=3)
    plt.plot(array2[:, 0], array2[:, 1], 'r-', linewidth=0.5)
    plt.grid()
    plt.savefig(save_dir + 'plot_scan_loc.png', transparent=True, bbox_inches='tight')
    plt.clf()
    
    # ===================== Plot offsets between ground truth and current scan locations =======================
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the points from array1 in blue
    ax.scatter(array1[:, 0], array1[:, 1], c='blue', s=10, label='ground truth scan loc')

    # Plot the points from array2 in red
    ax.scatter(array2[:, 0], array2[:, 1], c='red', s=10, label='output scan loc')

    # Draw lines between corresponding points
    for i in range(len(array1)):
        ax.plot([array1[i, 0], array2[i, 0]], [array1[i, 1], array2[i, 1]], c='gray')

    # Set labels and legend
    ax.set_xlabel('X (in px)')
    ax.set_ylabel('Y (in px)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2))

    # Save the figure with a transparent background and a tight bounding box
    plt.savefig(save_dir + 'compare_scan_loc.png', transparent=True, bbox_inches='tight')
    plt.clf()
    
    # ===================== Histogram of Distances Between Corresponding Points =======================
    # Calculate the Euclidean distances between corresponding points
    horizontal_distances = array1[:, 0] - array2[:, 0]
    vertical_distances = array1[:, 1] - array2[:, 1]
    
    # Calculate the maximum offset
    max_offset = np.maximum(np.max(np.abs(horizontal_distances)), np.max(np.abs(vertical_distances)))
    hist_range = int(np.ceil(max_offset))                          

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns of subplots

    # Plot the histogram of horizontal distances
    ax1.hist(horizontal_distances, bins=2 * hist_range, range=(-hist_range, hist_range), color='blue', alpha=0.7, density=True)
    ax1.set_xlabel('Horizontal Distance')
    # ax1.set_ylabel('Frequency')
    ax1.set_ylabel('Probability')
    ax1.set_title('Histogram of Horizontal Distances')
    ax1.set_xlim(-hist_range, hist_range)  # Set x-axis range
    ax1.set_ylim(0, 0.4)  # Set y-axis range
    ax1.grid()
    
    # Plot the histogram of vertical distances
    ax2.hist(vertical_distances, bins=2 * hist_range, range=(-hist_range, hist_range), color='blue', alpha=0.7, density=True)
    ax2.set_xlabel('Vertical Distance')
    # ax2.set_ylabel('Frequency')
    ax2.set_ylabel('Probability')
    ax2.set_title('Histogram of Vertical Distances')
    ax2.set_xlim(-hist_range, hist_range)  # Set x-axis range
    ax2.set_ylim(0, 0.4)  # Set y-axis range
    ax2.grid()
    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_dir + 'scan_loc_offset_hist.png', transparent=True, bbox_inches='tight')
    plt.clf()
    
    
def plot_heap_map(cmplx_img, cmplx_probe, patch_crd, curr_meas, step_sz, save_dir=None, true_offset=[0, 0], h_range=4):
    """
    The function searchs for the offset within a 3x3 grid, with each neighboring point being step_sz apart.

    
    Args:
        cmplx_img: full-sized complex tranmisttance image.
        cmplx_probe: complex-valued probe function.
        patch_crd: coordiantes that determines the current scan location.
        curr_meas: current measurement associated with the patch_crd.
        step_sz: predefined value that determines the size of search grid.
        
    Returns:
        [offset along x-axis, offset along y-axis]. 
    """
    partial_function = partial(forward_propagation, cmplx_img, cmplx_probe, patch_crd, curr_meas)
    v_offset = range(-h_range, h_range, 1)
    h_offset = range(-h_range, h_range, 1)
    v_offset = np.asarray(v_offset)
    h_offset = np.asarray(h_offset)
    fval = np.zeros((1, len(v_offset)*len(h_offset)))
    offsets = product(v_offset, h_offset)
    with Pool(processes=8) as p:
        fval = p.starmap(partial_function, offsets)
        p.close()
        p.join()
    
    # Create a 2D grid of X and Y coordinates using numpy.meshgrid
    V, H = np.meshgrid(v_offset, h_offset)
    fvmx = np.reshape(fval, [len(v_offset), len(h_offset)])
    sidx = np.unravel_index(np.nanargmin(fvmx), fvmx.shape)
    
    # Use pcolormesh to create the heatmap
    plt.rcdefaults()
    heatmap = plt.pcolormesh(H, V, np.transpose(fvmx), cmap='coolwarm') #, vmin=20, vmax=120)  # Adjust the cmap as needed
    
    # Add colorbar
    plt.colorbar(heatmap)
    
    # Highlight minimum value by drawing a red rectangle around it
    # min_x, min_y = np.unravel_index(np.argmin(fvmx), (9, 9))  # Find the indices of the minimum value
    rect = plt.Rectangle((h_offset[sidx[1]] - 0.5, v_offset[sidx[0]] - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)  # Add the rectangle to the plot

    # Highlight the precise location by drawing a green rectangle around it
    highlight_v, highlight_h = true_offset[0], true_offset[1]  # Specify the location to highlight
    rect_highlight = plt.Rectangle((highlight_h - 0.5, highlight_v - 0.5), 1, 1, linewidth=2, edgecolor='g', facecolor='none')
    plt.gca().add_patch(rect_highlight)  # Add the rectangle to the plot
    
    # Highlight the current location by drawing a yellow rectangle around it
    curr_x, curr_y = 0, 0  # Specify the location to highlight
    rect_curr = plt.Rectangle((curr_x - 0.5, curr_y - 0.5), 1, 1, linewidth=2, edgecolor='y', facecolor='none')
    plt.gca().add_patch(rect_curr)  # Add the rectangle to the plot
    
    # Create the heatmap plot
    plt.xlabel('X-offset')
    plt.ylabel('Y-offset')
    plt.title('Heatmap of Values over Offsets')
    # Set axis spines (boundary lines) color and style
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.savefig(save_dir, transparent=True, bbox_inches='tight')
    plt.clf()   