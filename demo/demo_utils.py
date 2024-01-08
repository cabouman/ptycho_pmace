import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pmace.display import plot_cmplx_img


def plot_synthetic_img(cmplx_img, img_title, display_win=None, save_dir=None):
    """
    Display and save a demo result.

    Args:
        cmplx_img: complex image array.
        img_title: title of the plot image.
        display_win: window to plot the image.
        save_dir: directory for saving the image.
    
    Example:
    plot_synthetic_img(cmplx_img, img_title="demo_img", display_win=None, save_dir='/path/to/save/directory/')
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
    save_fname = None if (save_dir is None) else os.path.join(save_dir, f'{img_title}_recon_cmplx_img')

    # Plot complex image with customized parameters
    plot_cmplx_img(cmplx_img, 
                   img_title=img_title, 
                   display_win=display_win, 
                   save_fname=save_fname,
                   mag_vmax=1, 
                   mag_vmin=0.5, 
                   phase_vmax=0, 
                   phase_vmin=-np.pi/4,
                   real_vmax=1.1, 
                   real_vmin=0.8, 
                   imag_vmax=0, 
                   imag_vmin=-0.6)