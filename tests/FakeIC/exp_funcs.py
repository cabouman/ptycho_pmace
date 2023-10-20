import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pmace.display import *


def plot_meas_img(y_meas, save_dir=None):
    """
    Plot and save a measurement image.

    Parameters:
    - y_meas (numpy.ndarray): Measurement data to be visualized.
    - save_dir (str): Directory to save the plot. If None, the current directory is used.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot measurement
    plt.imshow(np.abs(y_meas), cmap=cm.gray, norm=LogNorm(vmin=0.00001, vmax=1))
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'diffraction_pattern.png'))
    
    # Clear the current figure to avoid overlapping plots
    plt.clf()
    
    
def plot_FakeIC_img(cmplx_img, img_title, display_win=None, save_dir=None):
    """
    Display and save the complex image.

    Args:
        cmplx_img (array-like): A complex image represented as an array.
        img_title (str): The title of the plot image.
        display_win (array-like, optional): A window used to display the image. Defaults to None.
        save_dir (str, optional): The directory where the image will be saved. Defaults to None.

    This function displays the complex image and saves it to the specified directory if 'save_dir' is provided.
    You can predefine the window for showing the image using optional parameters 'display_win'.
    The saved image is named as '<img_title>_recon_cmplx_img.png'.

    Example:
    plot_FakeIC_img(cmplx_img=cmplx_img, img_title='Complex-valued Image', save_dir='/path/to/save/directory/')
    """
    # Check save_dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    save_fname = None if (save_dir is None) else f"{save_dir}{img_title}_cmplx_img"

    # Plot complex image
    plot_cmplx_img(cmplx_img, 
                   img_title=img_title, 
                   display_win=display_win, 
                   save_fname=save_fname,
                   mag_vmax=1, 
                   mag_vmin=0, 
                   phase_vmax=np.pi/2, 
                   phase_vmin=0,
                   real_vmax=1, 
                   real_vmin=0, 
                   imag_vmax=1, 
                   imag_vmin=0)
    

def plot_probe_img(cmplx_img, img_title, save_dir=None):
    """
    Display and save the complex image.

    Args:
        cmplx_img (array-like): A complex image represented as an array.
        img_title (str): The title of the plot image.
        save_dir (str, optional): The directory where the image will be saved. Defaults to None.

    This function displays the complex image and saves it to the specified directory if 'save_dir' is provided.
    The saved image is named as '<img_title>_recon_cmplx_img.png'.

    Example:
    plot_FakeIC_img(cmplx_img=cmplx_img, img_title='Complex-valued Image', save_dir='/path/to/save/directory/')
    """
    # Check save_dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    save_fname = None if (save_dir is None) else f"{save_dir}{img_title}_cmplx_probe"

    # Plot complex image
    plot_cmplx_img(cmplx_img, 
                   img_title=img_title,  
                   display_win=None,
                   save_fname=save_fname,
                   mag_vmax=100, 
                   mag_vmin=0, 
                   phase_vmax=np.pi, 
                   phase_vmin=-np.pi,
                   real_vmax=20, 
                   real_vmin=-60, 
                   imag_vmax=20, 
                   imag_vmin=-60)
    
    
def compare_result_with_ground_truth_img(cmplx_img, ref_img=None, display_win=None, save_dir=None,
                                         mag_vmax=1, mag_vmin=0, phase_vmax=np.pi/2, phase_vmin=0, 
                                         real_vmax=1, real_vmin=0, imag_vmax=1, imag_vmin=0):
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

    Example:
    compare_result_with_ground_truth_img(
        cmplx_img=reconstructed_img,
        ref_img=ground_truth_img,
        display_win=window,
        save_dir='/path/to/save/directory/',
        mag_vmax=1.0,
        mag_vmin=0.0,
        phase_vmax=3.14,
        phase_vmin=0.0,
        real_vmax=1.0,
        real_vmin=0.0,
        imag_vmax=1.0,
        imag_vmin=0.0
    )
    """
    # Initialize window and determine area for showing and comparing images
    if display_win is None:
        display_win = np.ones_like(cmplx_img, dtype=np.complex64)
    non_zero_idx = np.nonzero(display_win)
    blk_idx = [np.amin(non_zero_idx[0]), np.amax(non_zero_idx[0])+1, np.amin(non_zero_idx[1]), np.amax(non_zero_idx[1])+1]
    cmplx_img_rgn = cmplx_img[blk_idx[0]:blk_idx[1], blk_idx[2]:blk_idx[3]]
    
    # Prepare reference image
    if ref_img is not None:
        ref_img_rgn = ref_img[blk_idx[0]:blk_idx[1], blk_idx[2]:blk_idx[3]]
        
    # Phase normalization
    cmplx_img_rgn = phase_norm(cmplx_img_rgn, ref_img_rgn)
    plt.figure(num=None, figsize=(10, 6), dpi=400, facecolor='w', edgecolor='k')
    
    # ===================== Plots of reference image
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
    
    # ===================== Plots of reconstructed image
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
    
    # ===================== Plots of error/difference
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

    
def compare_result_with_ground_truth_probe(cmplx_img, ref_img=None, save_dir=None,
                                           mag_vmax=100, mag_vmin=0, phase_vmax=np.pi, phase_vmin=-np.pi, 
                                           real_vmax=20, real_vmin=-60, imag_vmax=20, imag_vmin=-60):
    """
    Compare a reconstructed image with a ground truth image and plot error images for various components.

    Args:
        cmplx_img (numpy.ndarray): Complex image array representing the reconstructed image.
        ref_img (numpy.ndarray or None): Complex image array representing the ground truth image. Use None if not available.
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

    Example:
    compare_result_with_ground_truth_img(
        cmplx_img=reconstructed_img,
        ref_img=ground_truth_img,
        display_win=window,
        save_dir='/path/to/save/directory/',
        mag_vmax=1.0,
        mag_vmin=0.0,
        phase_vmax=3.14,
        phase_vmin=0.0,
        real_vmax=1.0,
        real_vmin=0.0,
        imag_vmax=1.0,
        imag_vmin=0.0
    )
    """
    # Phase normalization
    cmplx_img_rgn = phase_norm(cmplx_img, ref_img)
    ref_img_rgn = np.copy(ref_img)
    plt.figure(num=None, figsize=(10, 6), dpi=400, facecolor='w', edgecolor='k')
    
    # ===================== Plots of reference image
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
    
    # ===================== Plots of reconstructed image
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
    
    # ===================== Plots of error/difference
    # Mmplitude of difference between complex reconstruction and ground truth image
    plt.subplot(3, 4, 9)
    plt.imshow(np.abs(cmplx_img_rgn - ref_img_rgn), cmap='gray', vmax=5, vmin=0)
    plt.title(r'error - amp')
    plt.colorbar()
    plt.axis('off')
    # Phase difference between complex reconstruction and ground truth image
    ang_err = pha_err(cmplx_img_rgn, ref_img_rgn)
    plt.subplot(3, 4, 10)
    plt.imshow(ang_err, cmap='gray' , vmax=np.pi/2, vmin=-np.pi/2)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'phase error')
    # Real part of error image between complex reconstruction and ground truth image
    err = cmplx_img_rgn - ref_img_rgn
    plt.subplot(3, 4, 11)
    plt.imshow(np.real(err), cmap='gray' , vmax=5, vmin=-5)
    plt.colorbar()
    plt.axis('off')
    plt.title('err - real')
    # Image part of error between complex reconstruction and ground truth image
    plt.subplot(3, 4, 12)
    plt.imshow(np.imag(err), cmap='gray' , vmax=5, vmin=-5)
    plt.colorbar()
    plt.axis('off')
    plt.title('err - imag')
    
    plt.savefig(save_dir + 'reconstructed_probe.png', dpi=300, bbox_inches='tight', transparent=True)
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
    
    # ===================== Plot ground truth and current scan locations 
    # Plot ground truth locations in blue
    figure(figsize=[5, 8])
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
    
    # ===================== Plot offsets between ground truth and current scan locations 
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
    
    # ===================== Histogram of Distances Between Corresponding Points
    # Calculate the Euclidean distances between corresponding points
    horizontal_distances = array1[:, 0] - array2[:, 0]
    vertical_distances = array1[:, 1] - array2[:, 1]
    
    # Calculate the maximum offset
    max_offset = np.maximum(np.max(np.abs(horizontal_distances)), np.max(np.abs(vertical_distances)))
    hist_range = int(np.ceil(max_offset))                          

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns of subplots

    # Plot the histogram of horizontal distances
    ax1.hist(horizontal_distances, bins=20 * hist_range, range=(-hist_range, hist_range), color='blue', alpha=0.7, density=True)
    ax1.set_xlabel('Horizontal Distance')
    # ax1.set_ylabel('Frequency')
    ax1.set_ylabel('Probability')
    ax1.set_title('Histogram of Horizontal Distances')
    ax1.set_xlim(-hist_range, hist_range)  # Set x-axis range
    # ax1.set_ylim(0, 40)  # Set y-axis range
    ax1.grid()
    
    # Plot the histogram of vertical distances
    ax2.hist(vertical_distances, bins=20 * hist_range, range=(-hist_range, hist_range), color='blue', alpha=0.7, density=True)
    ax2.set_xlabel('Vertical Distance')
    # ax2.set_ylabel('Frequency')
    ax2.set_ylabel('Probability')
    ax2.set_title('Histogram of Vertical Distances')
    ax2.set_xlim(-hist_range, hist_range)  # Set x-axis range
    # ax2.set_ylim(0, 40)  # Set y-axis range
    ax2.grid()
    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_dir + 'scan_loc_offset_hist.png', transparent=True, bbox_inches='tight')
    plt.clf()
    

def plot_nrmse_at_meas_plane(meas_nrmse, save_dir):
    """
    Plot the graph showing Normalized Root Mean Square Error (NRMSE) at the measurement plane.

    Args:
        meas_nrmse (array-like): A sequence of NRMSE values calculated at different iterations.
        save_dir (str): The directory where the plot image will be saved.
        
    This function generates a semilogarithmic plot to visualize the convergence behavior.
    The plot is saved as 'convergence_plot.png' in the specified 'save_dir' directory.
    
    Example:
    plot_nrmse_at_meas_plane(meas_nrmse=[0.1, 0.05, 0.02, 0.01], save_dir='/path/to/save/directory/')
    """
    rcdefaults()
    plt.semilogy(meas_nrmse)
    plt.ylabel('NRMSE at detector plane (in log scale)')
    plt.xlabel('Number of Iterations')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(save_dir + 'convergence_plot')
    plt.clf()
    

def plot_convergence_curve(n_iter=100, init_err=0.2, err_obj=None, err_probe=None, err_meas=None, save_dir=None):
    """
    Plot convergence curves of performance metrics over iterations.

    Args:
        n_iter (int, optional): Number of iterations. Default is 100.
        init_err (float, optional): Initial error value. Default is 0.2.
        err_obj (array-like, optional): Object-related error values over iterations.
        err_probe (array-like, optional): Probe-related error values over iterations.
        err_meas (array-like, optional): Measurement-related error values over iterations.
        save_dir (str, optional): The directory where the convergence plot will be saved. Default is None.

    This function generates and saves a plot displaying the convergence curves of one or more error metrics
    over a specified number of iterations. The plot is saved as 'convergence_plot.png' in the specified 'save_dir' directory.

    Example:
    plot_convergence_curve(n_iter=100, err_obj=obj_err, err_probe=probe_err, err_meas=meas_err, save_dir='/path/to/save/directory/')
    """
    # Check if a directory for saving the plot is specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Create a large figure with three subplots
    plt.figure(figsize=(30, 6))

    # Subplot for Object NRMSE
    if err_obj is not None:
        plt.subplot(131)
        plt.semilogy(np.arange(0, n_iter + 1, 1), np.insert(err_obj, 0, init_err), label='PMACE')
        plt.ylabel('Object NRMSE (in log scale)')
        plt.xlabel('Number of Iterations')
        plt.xticks(np.arange(0, n_iter + 1, 10))
        plt.legend(loc='best')
        plt.grid(which='both')
        plt.grid(which='minor', alpha=0.1)
        plt.grid(which='major', alpha=0.2)

    # Subplot for Probe NRMSE
    if err_probe is not None:
        plt.subplot(132)
        plt.semilogy(np.arange(0, n_iter + 1, 1), np.insert(err_probe, 0, init_err), label='PMACE')
        plt.ylabel('Probe NRMSE (in log scale)')
        plt.xlabel('Number of Iterations')
        plt.xticks(np.arange(0, n_iter + 1, 10))
        plt.legend(loc='best')
        plt.grid(which='both')
        plt.grid(which='minor', alpha=0.1)
        plt.grid(which='major', alpha=0.2)

    # Subplot for Measurement NRMSE
    if err_meas is not None:
        plt.subplot(133)
        plt.semilogy(np.arange(0, n_iter + 1, 1), np.insert(err_meas, 0, init_err), label='PMACE')
        plt.ylabel('Meas NRMSE (in log scale)')
        plt.xlabel('Number of Iterations')
        plt.xticks(np.arange(0, n_iter + 1, 10))
        plt.legend(loc='best')
        plt.grid(which='both')
        plt.grid(which='minor', alpha=0.1)
        plt.grid(which='major', alpha=0.2)

    # Save the plot as 'convergence_plot.png' in the specified directory
    plt.savefig(os.path.join(save_dir, 'convergence_plot.png'))
    plt.clf()