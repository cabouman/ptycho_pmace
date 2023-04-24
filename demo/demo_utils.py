import sys, os
import numpy as np
import matplotlib.pyplot as plt
import glob
import urllib.request
import tarfile
from pathlib import Path
from utils.display import *


def plot_synthetic_img(cmplx_img, img_title, display_win=None, save_dir=None):
    """Display and save demo result.
    
    Args:
        cmplx_img: complex image array.
        img_title: title of plot image.
        display_win: window to plot image.
        save_dir: directory for saving image.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    save_fname = None if (save_dir is None) else save_dir + '{}_recon_cmplx_img'.format(img_title)

    # plot complex image
    plot_cmplx_img(cmplx_img, img_title=img_title, display_win=display_win, save_fname=save_fname,
                   mag_vmax=1, mag_vmin=0.5, phase_vmax=0, phase_vmin=-np.pi/4,
                   real_vmax=1.1, real_vmin=0.8, imag_vmax=0, imag_vmin=-0.6)
    
    
def download_and_extract(download_url, save_dir):
    """Download the file from ``download_url``, and save the file to ``save_dir``. 
    
    If the file already exists in ``save_dir``, user will be queried whether it is desired to download and overwrite the existing files.
    If the downloaded file is a tarball, then it will be extracted before saving. 
    Code reference: `https://github.com/cabouman/mbircone/`
    
    Args:
        download_url: url to download the data. This url needs to be public.
        save_dir (string): directory where downloaded file will be saved. 
        
    Returns:
        path to downloaded file. This will be ``save_dir``+ downloaded_file_name 
    """
    is_download = True
    local_file_name = download_url.split('/')[-1]
    save_path = os.path.join(save_dir, local_file_name)
    if os.path.exists(save_path):
        is_download = query_yes_no(f"{save_path} already exists. Do you still want to download and overwrite the file?")
    if is_download:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # download the data from url.
        print("Downloading file ...")
        try:
            urllib.request.urlretrieve(download_url, save_path)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise RuntimeError(
                    f'HTTP status code {e.code}: URL authentication failed! Currently we do not support downloading data from a url that requires authentication.')
            elif e.code == 403:
                raise RuntimeError(
                    f'HTTP status code {e.code}: URL forbidden! Please make sure the provided URL is public.')
            elif e.code == 404:
                raise RuntimeError(
                    f'HTTP status code {e.code}: URL not Found! Please check and make sure the download URL provided is correct.')
            else:
                raise RuntimeError(
                    f'HTTP status code {e.code}: {e.reason}. For more details please refer to https://en.wikipedia.org/wiki/List_of_HTTP_status_codes')
        except urllib.error.URLError as e:
            raise RuntimeError('URLError raised! Please check your internet connection.')
        print(f"Download successful! File saved to {save_path}")
    else:
        print("Skipped data download and extraction step.")
    # Extract the downloaded file if it is tarball
    if save_path.endswith(('.tar', '.tar.gz', '.tgz')):
        if is_download:
            tar_file = tarfile.open(save_path)
            print(f"Extracting tarball file to {save_dir} ...")
            # Extract to save_dir.
            tar_file.extractall(save_dir)
            tar_file.close
            print(f"Extraction successful! File extracted to {save_dir}")
        save_path = save_dir
        # Remove invisible files with "._" prefix 
        for spurious_file in glob.glob(save_dir + "/**/._*", recursive=True):
            os.remove(spurious_file)
    # Parse extracted dir and extract data if necessary
    return save_path


def query_yes_no(question, default="n"):
    """Ask a yes/no question via input() and return the answer.
    
    Code reference: `https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input/3041990`.
        
    Args:
        question (string): Question that is presented to the user.
        
    Returns:
        Boolean value: True for "yes" or "Enter", or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = f" [y/n, default={default}] "
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
    return
