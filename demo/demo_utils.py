import sys
import os
import yaml
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pmace.display import plot_cmplx_img


def load_configuration(config_dir):
    """
    Load configuration settings from a YAML file.

    Args:
        config_dir (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration settings loaded from the YAML file.
    """
    with open(config_dir, 'r') as f:
        return yaml.safe_load(f)
    
    
def setup_output_directory_with_timestamp(output_dir):
    """
    Set up an output directory with a timestamp.

    Args:
        output_dir (str): Output directory without timestamp.

    Returns:
        str: Timestamped output directory.
    """
    # Determine the timestamp for the output directory
    today_date = dt.date.today()
    date_time = dt.datetime.strftime(today_date, '%Y-%m-%d_%H_%M/')
    output_dir_with_timestamp = os.path.join(output_dir, date_time)

    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir_with_timestamp, exist_ok=True)
        print(f"Output directory '{output_dir_with_timestamp}' created successfully")
    except OSError as error:
        print(f"Output directory '{output_dir_with_timestamp}' cannot be created")

    return output_dir_with_timestamp


def plot_synthetic_img(cmplx_img, img_title, display=True, display_win=None, save_dir=None):
    """
    Display and save a demo result.

    Args:
        cmplx_img: complex image array.
        img_title: title of the plot image.
        display: display the image.
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
                   display=display,
                   save_fname=save_fname,
                   mag_vmax=1, 
                   mag_vmin=0.5, 
                   phase_vmax=0, 
                   phase_vmin=-np.pi/4,
                   real_vmax=1.1, 
                   real_vmin=0.8, 
                   imag_vmax=0, 
                   imag_vmin=-0.6)