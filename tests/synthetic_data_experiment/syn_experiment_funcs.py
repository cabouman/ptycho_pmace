from utils.utils import *
from utils.display import *

'''
This file defines the functions needed for displaying experimental results. 
'''


def plot_synthetic_img(cmplx_img, img_title, ref_img, display_win=None, display=False, save_dir=None):
    """ Function to plot reconstruction results in recon experiment on synthetic data. """
    save_fname = None if (save_dir is None) else save_dir + 'recon_cmplx_img'
    plot_cmplx_img(cmplx_img, img_title=img_title, ref_img=ref_img,
                   display_win=display_win, display=display, save_fname=save_fname,
                   fig_sz=[8, 3], mag_vmax=1, mag_vmin=0.5, phase_vmax=0, phase_vmin=-np.pi/4,
                   real_vmax=1.1, real_vmin=0.8, imag_vmax=0, imag_vmin=-0.6)
