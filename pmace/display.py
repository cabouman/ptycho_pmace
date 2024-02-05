import numpy as np
import matplotlib.pyplot as plt
from .nrmse import *


def plot_scan_pt(scan_pt, save_dir):
    """
    Plot scan points.

    Args:
        scan_pt (numpy.ndarray): Array of scan points as (x, y) coordinates.
        save_dir (str): Directory to save the plot.
    """
    plt.plot(scan_pt[:, 0], scan_pt[:, 1], 'o-')
    plt.title('Scan Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(save_dir + 'scan_locations.png', transparent=True, bbox_inches='tight')
    plt.clf()
    

def plot_nrmse(nrmse_ls, title, label, abscissa=None, step_sz=15, fig_sz=[10, 4.8], display=False, save_fname=None):
    """Plot the NRMSE (Normalized Root Mean Squared Error) versus the number of iterations.

    Args:
        nrmse_ls (list or array): List of NRMSE values for each iteration or a dictionary of labels and NRMSE values.
        title (str): Title for the plot.
        label (list): List containing the X and Y axis labels and the label for the legend (e.g., ['X Label', 'Y Label', 'Legend Label']).
        abscissa (list or None): X-axis values corresponding to NRMSE data. If None, it is automatically generated.
        step_sz (int): Step size for X-axis ticks.
        fig_sz (list): Size of the figure in inches (width, height).
        display (bool): Display the plot if True.
        save_fname (str or None): Save the plot to a file with the specified filename (without extension).
    """
    xlabel, ylabel = label[0], label[1]
    plt.figure(num=None, figsize=(fig_sz[0], fig_sz[1]), dpi=100, facecolor='w', edgecolor='k')

    if np.all(abscissa == None):
        if isinstance(nrmse_ls, dict):
            length = 0
            for line_label, line in nrmse_ls.items():
                length = np.maximum(length, len(line))
                plt.semilogy(line, label=line_label)
        else:
            line = np.asarray(nrmse_ls)
            length = len(line)
            line_label = label[2]
            plt.semilogy(line, label=line_label)
        plt.xticks(np.arange(length, step=step_sz), np.arange(1, length + 1, step=step_sz))
        plt.legend(loc='best')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.grid(True)
        if save_fname is not None:
            plt.savefig('{}.png'.format(save_fname))
        if display:
            plt.show()
        plt.clf()
    else:
        if isinstance(nrmse_ls, dict):
            idx = 0
            xmin, xmax, ymin, ymax = 1, 0, 1, 0
            for line_label, line in nrmse_ls.items():
                xmin = np.minimum(xmin, np.amin(np.asarray(abscissa[idx])))
                xmax = np.maximum(xmax, np.amax(np.asarray(abscissa[idx])))
                ymin = np.minimum(ymin, np.amin(np.asarray(line)))
                ymax = np.maximum(ymax, np.amax(np.asarray(line)))
                plt.semilogy(line, abscissa[idx], label=line_label)
                idx = idx + 1
        else:
            line = np.asarray(nrmse_ls)
            plt.semilogy(line, abscissa, label=label[2])
            xmin, xmax = np.amin(abscissa), np.max(abscissa)
            ymin, ymax = np.amin(line), np.amax(line)
        plt.xlim([np.maximum(xmin, 0), xmax+1e-2])
        plt.ylim([ymin-1e-2, ymax+1e-2])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if save_fname is not None:
            plt.savefig('{}.png'.format(save_fname))
        if display:
            plt.show()
        plt.clf()


def plot_cmplx_img(cmplx_img, img_title='img', ref_img=None, display_win=None, display=False, save_fname=None,
                   fig_sz=[8, 3], mag_vmax=1, mag_vmin=0, phase_vmax=np.pi, phase_vmin=-np.pi,
                   real_vmax=1, real_vmin=0, imag_vmax=0, imag_vmin=-1):
    """Plot complex object images and error images compared with a reference image.

    Args:
        cmplx_img (numpy.ndarray): Complex-valued image.
        img_title (str): Title for the complex image.
        ref_img (numpy.ndarray or None): Reference image. If provided, error images will be displayed.
        display_win (numpy.ndarray or None): Pre-defined window for displaying images.
        display (bool): Display images if True.
        save_fname (str or None): Save images to the specified file directory.
        fig_sz (list): Size of image plots in inches (width, height).
        mag_vmax (float): Maximum value for showing image magnitude.
        mag_vmin (float): Minimum value for showing image magnitude.
        phase_vmax (float): Maximum value for showing image phase.
        phase_vmin (float): Minimum value for showing image phase.
        real_vmax (float): Maximum value for showing the real part of the image.
        real_vmin (float): Minimum value for showing the real part of the image.
        imag_vmax (float): Maximum value for showing the imaginary part of the image.
        imag_vmin (float): Minimum value for showing the imaginary part of the image.
    """
    # Plot error images if a reference image is provided
    show_err_img = False if (ref_img is None) or (np.linalg.norm(cmplx_img - ref_img) < 1e-9) else True

    # Initialize the window and determine the area for showing and comparing images
    if display_win is None:
        display_win = np.ones_like(cmplx_img, dtype=np.complex64)
    non_zero_idx = np.nonzero(display_win)
    blk_idx = [np.amin(non_zero_idx[0]), np.amax(non_zero_idx[0])+1, np.amin(non_zero_idx[1]), np.amax(non_zero_idx[1])+1]
    cmplx_img_rgn = cmplx_img[blk_idx[0]:blk_idx[1], blk_idx[2]:blk_idx[3]]
    if ref_img is not None:
        ref_img_rgn = ref_img[blk_idx[0]:blk_idx[1], blk_idx[2]:blk_idx[3]]
    
    # Display the amplitude and phase images
    plt.figure(num=None, figsize=(fig_sz[0], fig_sz[1]), dpi=400, facecolor='w', edgecolor='k')
    
    # Magnitude of the reconstructed complex image
    plt.subplot(2, 4, 1)
    plt.imshow(np.abs(cmplx_img_rgn), cmap='gray', vmax=mag_vmax, vmin=mag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'Magnitude of {}'.format(img_title))
                                  
    # Phase of the reconstructed complex image
    plt.subplot(2, 4, 2)
    plt.imshow(np.angle(cmplx_img_rgn), cmap='gray', vmax=phase_vmax, vmin=phase_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'Phase of {}'.format(img_title))
    
    # Real part of the reconstructed complex image
    plt.subplot(2, 4, 3)
    plt.imshow(np.real(cmplx_img_rgn), cmap='gray', vmax=real_vmax, vmin=real_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('Real Part of {}'.format(img_title))
    
    # Imaginary part of the reconstructed complex image
    plt.subplot(2, 4, 4)
    plt.imshow(np.imag(cmplx_img_rgn), cmap='gray', vmax=imag_vmax, vmin=imag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('Imaginary Part of {}'.format(img_title))

    if show_err_img:
        # Amplitude of the difference between complex reconstruction and ground truth image
        plt.subplot(2, 4, 5)
        plt.imshow(np.abs(cmplx_img_rgn - ref_img_rgn), cmap='gray', vmax=0.2, vmin=0)
        plt.title(r'Error - Amp')
        plt.colorbar()
        plt.axis('off')
        
        # Phase difference between complex reconstruction and ground truth image
        ang_err = pha_err(cmplx_img_rgn, ref_img_rgn)
        plt.subplot(2, 4, 6)
        plt.imshow(ang_err, cmap='gray', vmax=np.pi/2, vmin=-np.pi/2)
        plt.colorbar()
        plt.axis('off')
        plt.title('Phase Error')
        
        # Real part of the error image between complex reconstruction and ground truth image
        err = cmplx_img_rgn - ref_img_rgn
        plt.subplot(2, 4, 7)
        plt.imshow(np.real(err), cmap='gray', vmax=0.2, vmin=-0.2)
        plt.colorbar()
        plt.axis('off')
        plt.title('Error - Real')
        
        # Imaginary part of the error between complex reconstruction and ground truth image
        plt.subplot(2, 4, 8)
        plt.imshow(np.imag(err), cmap='gray', vmax=0.2, vmin=-0.2)
        plt.colorbar()
        plt.axis('off')
        plt.title('Error - Imag')

    if save_fname is not None:
        plt.savefig('{}.png'.format(save_fname))
    if display:
        plt.show()
    plt.clf()