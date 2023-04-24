from utils.utils import *
from utils.display import *

'''
This file defines the functions needed for displaying experimental results. 
'''


def plot_goldball_img(cmplx_img, ref_img=None, display_win=None, display=False, img_title=None, save_dir=None):
    """ Function to plot reconstruction results in this experiment. """
    # check directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    img_name = 'img' if img_title is None else img_title + '_obj'
    
    # initialize window and determine area for showing and comparing images
    if display_win is None:
        display_win = np.ones_like(cmplx_img, dtype=np.complex64)
    non_zero_idx = np.nonzero(display_win)
    a, b = np.maximum(0, np.amin(non_zero_idx[0])), np.minimum(cmplx_img.shape[0], np.amax(non_zero_idx[0]) + 1)
    c, d = np.maximum(0, np.amin(non_zero_idx[1])), np.minimum(cmplx_img.shape[1], np.amax(non_zero_idx[1]) + 1)
    win_img = cmplx_img[a:b, c:d]
    
    # normalization using reference image
    if ref_img is not None:
        win_ref_img = ref_img[a:b, c:d]
        win_img = phase_norm(win_img, win_ref_img)
    
    # subtract mean from phase
    norm_img = np.abs(win_img) * np.exp(1j * (np.angle(win_img) - np.mean(np.angle(win_img))))
 
    # define formatting of plots
    fig_args = dict(bbox_inches='tight', pad_inches=0, dpi=160)
    
    # plot real part of complex image
    real_plot = plt.imshow(np.real(norm_img), cmap='gray', vmax=140, vmin=70)
    plt.colorbar()
    plt.axis('off')
    if save_dir is not None:
        plt.savefig(save_dir + '{}_real'.format(img_name), **fig_args)
    if display:
        plt.show()
    plt.clf()
    
    # plot imaginary part of complex image
    imag_plot = plt.imshow(np.imag(norm_img), cmap='gray', vmax=60, vmin=-30)
    plt.colorbar()
    plt.axis('off')
    if save_dir is not None:
        plt.savefig(save_dir + '{}_imag'.format(img_name), **fig_args)
    if display:
        plt.show()
    plt.clf()

    # plot magnitude of complex image
    mag_plot = plt.imshow(np.abs(norm_img), cmap='gray', vmax=140, vmin=60)
    plt.colorbar()
    plt.axis('off')
    if save_dir is not None:
        plt.savefig(save_dir + '{}_mag'.format(img_name), **fig_args)
    if display:
        plt.show()
    plt.clf()
    
    # plot phase of complex image
    phase_plot = plt.imshow(np.angle(norm_img), cmap='gray', vmax=0.6, vmin=-0.2)
    plt.colorbar()
    plt.axis('off')
    if save_dir is not None:
        plt.savefig(save_dir + '{}_phase'.format(img_name), **fig_args)
    if display:
        plt.show()
    plt.clf()


def plot_goldball_probe(cmplx_probe, ref_probe=None, display=False, img_title=None, save_dir=None):
    """ Function to plot reconstruction results in this experiment. """
    # check directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    img_name = 'probe' if img_title is None else img_title + '_probe'

    # phase normalization
    cmplx_img = phase_norm(cmplx_probe, ref_probe) if ref_probe is not None else cmplx_probe

    # define formatting of plots
    fig_args = dict(bbox_inches='tight', pad_inches=0, dpi=160)
    
    # plot real part of complex image
    real_plot = plt.imshow(np.real(cmplx_img), cmap='gray')
    plt.colorbar()
    plt.axis('off')
    if save_dir is not None:
        plt.savefig(save_dir + '{}_real'.format(img_name), **fig_args)
    if display:
        plt.show()
    plt.clf()

    # plot imaginary part of complex image
    imag_plot = plt.imshow(np.imag(cmplx_img), cmap='gray')
    plt.colorbar()
    plt.axis('off')
    if save_dir is not None:
        plt.savefig(save_dir + '{}_imag'.format(img_name), **fig_args)
    if display:
        plt.show()
    plt.clf()

    # plot magnitude of complex image
    mag_plot = plt.imshow(np.abs(cmplx_img), cmap='gray')
    plt.colorbar()
    plt.axis('off')
    if save_dir is not None:
        plt.savefig(save_dir + '{}_mag'.format(img_name), **fig_args)
    if display:
        plt.show()
    plt.clf()

   # plot phase of complex image
    phase_plot = plt.imshow(np.angle(cmplx_img), cmap='gray')
    plt.colorbar()
    plt.axis('off')
    if save_dir is not None:
        plt.savefig(save_dir + '{}_phase'.format(img_name), **fig_args)
    if display:
        plt.show()
    plt.clf()


def plot_nrmse_at_meas_plane(meas_nrmse, save_dir):
    rcdefaults()
    plt.semilogy(meas_nrmse)
    plt.ylabel('NRMSE at detector plane (in log scale)')
    plt.xlabel('Number of Iterations')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(save_dir + 'convergence_plot')
    plt.clf()
# def plot_CuFoam_img(cmplx_img, img_title, display_win=None, display=False, save_dir=None):
#     """ Function to plot reconstruction results in recons experiment on CuFoam Data. """
#     save_fname = None if (save_dir is None) else save_dir + 'recon_cmplx_img'
#     plot_cmplx_img(cmplx_img, img_title=img_title, ref_img=None,
#                    display_win=display_win, display=display, save_fname=save_fname,
#                    fig_sz=[8, 3], mag_vmax=2, mag_vmin=0, real_vmax=1.9, real_vmin=-1.3, imag_vmax=1.3, imag_vmin=-1.9)
