import sys
import os
import pyfftw
import re
import imageio
import scico.linop.optics as op
import tifffile as tiff
import pandas as pd
import numpy as np
from numpy import linalg as LA
import multiprocessing as mp
from scipy import signal
from scipy.ndimage import gaussian_filter
import glob
import urllib.request
import tarfile
from pathlib import Path


def int2float(arg):
    """Convert an integer argument to a floating-point number.

    Args:
        arg (int): Integer argument.

    Returns:
        Floating-point number.
    """
    output = arg.astype(np.float64) if isinstance(arg, int) else arg
    
    return output


def float2cmplx(arg):
    """Convert a floating-point argument to a complex number.

    Args:
        arg (float): Float argument.

    Returns:
        Complex number.
    """
    output = arg.astype(np.complex64) if isinstance(arg, float) else arg
    
    return output


def gen_gif(cmplx_images, fps=5, save_dir=None):
    """Generate a .gif image from a sequence of complex-valued images.

    Args:
        cmplx_images (list of numpy.ndarray): List of complex images.
        fps (int): Frames per second.
        save_dir (str): Output directory.
    """
    # Convert real and imaginary parts to uint8
    real_images = [scale(np.real(cmplx_img), [0, 1]).astype(np.float64) + 1e-16 for cmplx_img in cmplx_images]
    imag_images = [scale(np.imag(cmplx_img), [0, 1]).astype(np.float64) + 1e-16 for cmplx_img in cmplx_images]
    real_uint = np.asarray([255 * real_img / np.amax(real_img) for real_img in real_images]).astype(np.uint8)
    imag_uint = np.asarray([255 * imag_img / np.amax(imag_img) for imag_img in imag_images]).astype(np.uint8)
    
    # Save to file
    stack_img = [np.hstack((real_uint[idx], imag_uint[idx])) for idx in range(len(real_uint))]
    imageio.mimsave(save_dir + 'real_imag_reconstruction.gif', stack_img, fps=fps)
    

def load_img(img_dir):
    """Read an image from a directory.

    Args:
        img_dir (str): Directory of the image.

    Returns:
        numpy.ndarray: Complex image array.
    """
    # Read a TIFF image
    img = tiff.imread(img_dir)
    real, imag, mag, pha = img[0], img[1], img[2], img[3]
    cmplx_img = real + 1j * imag
                 
    return cmplx_img


def gen_scan_loc(cmplx_obj, cmplx_probe, num_pt, probe_spacing, randomization=True, max_offset=5):
    """Generate scan locations.

    Args:
        cmplx_obj (numpy.ndarray): Complex sample image to be scanned.
        cmplx_probe (numpy.ndarray): Complex probe.
        num_pt (int): Number of scan points.
        probe_spacing (float): Probe spacing between neighboring scan positions.
        randomization (bool): Option to add random offsets to each scan point.
        max_offset (int): Maximum offsets to be added to scan points along each dimension.

    Returns:
        numpy.ndarray: Generated scan points.
    """
    # Initialization
    x, y = cmplx_obj.shape
    m, n = cmplx_probe.shape
    num_pt_x, num_pt_y = int(np.sqrt(num_pt)), int(np.sqrt(num_pt))

    # Generate scan points in raster order
    scan_pt = [((i - num_pt_x / 2 + 1 / 2) * probe_spacing + x / 2, (j - num_pt_y / 2 + 1 / 2) * probe_spacing + y / 2)
               for j in range(num_pt_x)
               for i in range(num_pt_y)]

    # Add random offsets to each scan point
    if randomization:
        offset = np.random.uniform(low=-max_offset, high=(max_offset + 1), size=(num_pt, 2))
        scan_pt = np.asarray(scan_pt + offset)

    if ((int(np.amin(scan_pt) - m / 2) < 0) or (int(np.amax(scan_pt) + n / 2 ) >= np.max([x, y]))):
        print('Warning: Scanning beyond valid region! Please extend image or reduce probe spacing. ')

    return scan_pt


def gen_syn_data(cmplx_obj, cmplx_probe, patch_bounds, add_noise=True, peak_photon_rate=1e5, shot_noise_pm=0.5, save_dir=None):
    """Simulate ptychographic intensity measurements.

    Args:
        cmplx_obj (numpy.ndarray): Complex object.
        cmplx_probe (numpy.ndarray): Complex probe.
        patch_bounds (list of tuple): Scan coordinates of projections.
        add_noise (bool): Option to add noise to data.
        peak_photon_rate (float): Peak rate of photon detection at the detector.
        shot_noise_pm (float): Expected number of Poisson distributed dark current noise.
        save_dir (str): Directory for saving generated data.

    Returns:
        numpy.ndarray: Simulated ptychographic data.
    """
    # Initialization
    m, n = cmplx_probe.shape
    num_pts = len(patch_bounds)
    
    # Extract patches x_j from the full-sized object
    projected_patches = img2patch(cmplx_obj, patch_bounds, (num_pts, m, n))
    
    # Take 2D DFT and generate noiseless measurements
    noiseless_data = np.abs(compute_ft(cmplx_probe * projected_patches)) ** 2
        
    # Introduce photon noise
    if add_noise:
        # Get peak signal value
        peak_signal_val = np.amax(noiseless_data)
        # Calculate expected photon rate given peak signal value and peak photon rate
        expected_photon_rate = noiseless_data * peak_photon_rate / peak_signal_val
        # Poisson random values realization
        meas_in_photon_ct = np.random.poisson(expected_photon_rate, (num_pts, m, n))
        # Add dark current noise
        noisy_data = meas_in_photon_ct + np.random.poisson(lam=shot_noise_pm, size=(num_pts, m, n))
        # Return the numbers
        output = np.asarray(noisy_data, dtype=int)
    else:
        # Return the floating-point numbers without Poisson random variable realization
        output = np.asarray(noiseless_data)

    # Check directories and save simulated data
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for j in range(num_pts):
            tiff.imwrite(save_dir + 'frame_data_{}.tiff'.format(j), output[j])

    return output


def load_measurement(fpath):
    """Read measurements from a path and pre-process data.

    Args:
        fpath (str): File directory.

    Returns:
        numpy.ndarray: Pre-processed measurement (square root of non-negative data).
    """
    # Specify the order of measurement
    def key_func(fname):
        non_digits = re.compile("\D")
        output = int(non_digits.sub("", fname))
        return output

    # Read the measurements and remove negative values
    meas_ls = []
    work_dir = os.listdir(fpath)
    if '.DS_Store' in work_dir:
        work_dir.remove('.DS_Store')
        
    work_dir.sort(key=key_func)
    for fname in work_dir:
        # Load measurements
        meas = tiff.imread(os.path.join(fpath, fname))
        meas[meas < 0] = 0      
        # Take square root of the non-negative values
        meas_ls.append(np.sqrt(int2float(meas)))

    # Stack the measurements
    output = np.asarray(meas_ls)
        
    return output


def gen_init_obj(y_meas, coords, img_sz, ref_probe=None, lpf_sigma=10):
    """Formulate an initial guess of a complex object for reconstruction.

    Args:
        y_meas (numpy.ndarray): Pre-processed intensity measurements.
        coords (numpy.ndarray): Coordinates of projections.
        img_sz (tuple): Size of the full complex image (rows, columns).
        ref_probe (numpy.ndarray): Known or estimated complex probe function.
        lpf_sigma (float): Standard deviation of the Gaussian kernel for low-pass filtering the initialized guess.

    Returns:
        numpy.ndarray: The formulated initial guess of a complex transmittance image.
    """
    if ref_probe is None:
        ref_probe = np.ones_like(y_meas[0]).astype(np.complex64)

    patch_rms = [[np.sqrt(np.linalg.norm(y_meas[j]) / np.linalg.norm(ref_probe))] * np.ones_like(ref_probe)
                 for j in range(len(y_meas))]
    img_wgt = patch2img(np.ones_like(y_meas), coords, img_sz=img_sz)
    init_obj = patch2img(patch_rms, coords, img_sz=img_sz, norm_wgt=img_wgt)
    init_obj[init_obj == 0] = np.median(init_obj)
    
    # Apply LPF to remove high frequencies
    output = gaussian_filter(np.abs(init_obj), sigma=lpf_sigma)

    return output.astype(np.complex64)


def gen_init_probe(y_meas, coords, ref_obj, fres_propagation=False, sampling_interval=None,
                   source_wl=0.140891, propagation_dist=4e2, lpf_sigma=2):
    """Formulate an initial complex probe from the initialized object and data.

    Args:
        y_meas (numpy.ndarray): Pre-processed diffraction patterns.
        coords (numpy.ndarray): Coordinates of projections.
        ref_obj (numpy.ndarray): Ground truth complex object or reference images.
        fres_propagation (bool): Option to Fresnel propagate the initialized probe.
        sampling_interval (float): Sampling interval at the source plane.
        source_wl (float): Illumination wavelength.
        propagation_dist (float): Propagation distance.
        lpf_sigma (float): Standard deviation of the Gaussian kernel for removing high frequencies.

    Returns:
        numpy.ndarray: The formulated initial guess of a complex probe.
    """
    if sampling_interval is None:
        sampling_interval = 2 * source_wl
        
    # Formulate init probe
    patch = img2patch(ref_obj, coords, y_meas.shape)
    tmp = [compute_ift(y_meas[j]) / patch[j] for j in range(len(y_meas))]
    init_probe = np.average(tmp, axis=0)
        
    # Fresnel propagation initialized probe
    # TODO: double-check default parameters of Fresnel propagation
    if fres_propagation:
        m, n = init_probe.shape
        fres_op = op.FresnelPropagator(tuple([m, n]), dx=sampling_interval, k0=2 * np.pi / source_wl, z=propagation_dist)
        # fres_op = op.FresnelPropagator(tuple([m, n]), dx=7.33808, k0=2 * np.pi / 0.140891, z=3e5, pad_factor=1, jit=True)
        output = fres_op(init_probe)
    else:
        output = init_probe
        
    # Apply LPF to remove high frequencies
    output = gaussian_filter(np.real(output), sigma=lpf_sigma) + 1j * gaussian_filter(np.imag(output), sigma=lpf_sigma)

    return output.astype(np.complex64)


def patch2img(img_patches, patch_coords, img_sz, norm_wgt=None):
    """Project image patches to a full-sized image with weights.

    Args:
        img_patches (list of numpy.ndarray): Projected image patches.
        patch_coords (numpy.ndarray): Coordinates of projections.
        img_sz (tuple): Size of the full image (rows, columns).
        norm_wgt (numpy.ndarray): Normalization weight.

    Returns:
        numpy.ndarray: The full-sized complex image.
    """
    # Initialization
    if norm_wgt is None:
        norm_wgt = np.ones(img_sz, dtype=np.complex64)
    full_img = np.zeros(img_sz, dtype=np.complex64)

    # Back projection
    for j in range(len(img_patches)):
        patch, coords = img_patches[j], patch_coords[j]
        row_start, row_end, col_start, col_end = coords
        full_img[row_start:row_end, col_start:col_end] += patch

    # Normalization
    output = divide_cmplx_numbers(full_img, norm_wgt)

    return output


def img2patch(full_img, patch_coords, patch_sz):
    """Extract image patches from full-sized image.

    Args:
        img (numpy.ndarray): Full-sized image.
        coords (numpy.ndarray): Coordinates of projections.
        patch_sz (tuple): Size of the output patches (rows, columns).

    Returns:
        list of numpy.ndarray: Projected image patches.
    """
    # Initialization
    num_patches = len(patch_coords)
    output = []
    
    # Extract patches from the full image
    for j in range(num_patches):
        row_start, row_end, col_start, col_end = patch_coords[j]
        patch = full_img[row_start:row_end, col_start:col_end]
        output.append(patch)
        
    return np.asarray(output)


def compute_ft(input_array, threads=1):
    """Compute the 2D Discrete Fourier Transform (DFT) of an input array.
    
    Args:
        input_array (numpy.ndarray): The input 2D array for DFT computation.
        threads(int): number of threads for performing DFT using pyfftw.
        
    Returns:
        numpy.ndarray: The result of the 2D DFT.
    """
    # # DFT using numpy
    # a = np.fft.fftshift(input_array.astype(np.complex64), axes=(-2, -1))
    # b = np.fft.fft2(a, s=None, axes=(-2, -1), norm='ortho')
    # output = np.fft.ifftshift(b, axes=(-2, -1))
    
    # DFT using pyfftw
    if threads is None:
        threads = mp.cpu_count()
    
    a = np.fft.fftshift(input_array.astype(np.complex64), axes=(-2, -1))
    b = np.zeros_like(a)
    fft_object = pyfftw.FFTW(a, b, axes=(-2, -1), normalise_idft=False, ortho=True, direction='FFTW_FORWARD', threads=threads)
    output = np.fft.ifftshift(fft_object(), axes=(-2, -1))
    
    return output.astype(np.complex64)


def compute_ift(input_array, threads=1):
    """Compute the 2D Inverse Discrete Fourier Transform (IDFT) of an input array.
    
    Args:
        input_array (numpy.ndarray): The input 2D array for IDFT computation.
        threads(int): number of threads for performing DFT using pyfftw.
                
    Returns:
        numpy.ndarray: The result of the 2D IDFT.
    """
    # # DFT using numpy
    # a = np.fft.fftshift(input_array.astype(np.complex64), axes=(-2, -1))
    # b = np.fft.ifft2(a, s=None, axes=(-2, -1), norm='ortho')
    # output = np.fft.ifftshift(b, axes=(-2, -1))
    
    # DFT using pyfftw
    if threads is None:
        threads = mp.cpu_count()

    a = np.fft.fftshift(input_array.astype(np.complex64), axes=(-2, -1))
    b = np.zeros_like(a)
    ifft_object = pyfftw.FFTW(a, b, axes=(-2, -1), normalise_idft=False, ortho=True, direction='FFTW_BACKWARD', threads=threads)
    output = np.fft.ifftshift(ifft_object(), axes=(-2, -1))
    
    return output.astype(np.complex64)


def scale(input_obj, out_range):
    """Scale the input into a specified range.

    Args:
        input_obj (numpy.ndarray): Object to be scaled.
        out_range (tuple): Scale range (min, max).

    Returns:
        numpy.ndarray: Scaled input.
    """
    in_range = np.amin(input_obj), np.amax(input_obj)
    if (in_range[1] - in_range[0]) == 0:
        return input_obj
    else:
        y = (input_obj - (in_range[1] + in_range[0]) / 2) / (in_range[1] - in_range[0])
        output = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

    return output
                              

def divide_cmplx_numbers(cmplx_num, cmplx_denom):
    """Perform element-wise division with complex numbers, handling division by zero.

    Args:
        cmplx_num (numpy.ndarray): Complex numerator.
        cmplx_denom (numpy.ndarray): Complex denominator.

    Returns:
        ndarray: Result of the division.
    """
    # Use epsilon to avoid divsion by zero
    # epsilon = 1e-6 * LA.norm(cmplx_denom, ord='fro') / np.sqrt(cmplx_denom.size)
    fro_norm = np.sqrt(np.sum(np.square(np.abs(cmplx_denom))))
    epsilon = 1e-6 * fro_norm / np.sqrt(cmplx_denom.size)

    # Calculate the inverse of the denominator, considering epsilon
    denom_inv = np.conj(cmplx_denom) / (cmplx_denom * np.conj(cmplx_denom) + epsilon)

    # Perform element-wise operation, handling division by zero
    output = cmplx_num * denom_inv

    return output

                              
def save_tiff(cmplx_img, save_dir):
    """Save provided complex image to specified directory.

    Args:
        cmplx_img (numpy.ndarray): Complex image.
        save_dir (str): Specified directory for saving the image.
    """
    # Save reconstruction results
    img = np.asarray(cmplx_img)
    img_array = [np.real(img), np.imag(img), np.abs(img), np.angle(img)]
    tiff.imwrite(save_dir, np.asarray(img_array))


def save_array(arr, save_dir):
    """Save an array or list to a specified directory.

    Args:
        arr (numpy.ndarray or list): Numpy array or list to be saved.
        save_dir (str): Directory for saving the array.
    """
    f = open(save_dir, "wb")
    np.save(f, arr)
    f.close
    
                              
def get_proj_coords_from_data(scan_loc, y_meas):
    """Calculate projection coordinates from scan points.

    Args:
        scan_loc (numpy.ndarray): Scan locations.
        y_meas (numpy.ndarray): Pre-processed measurements.

    Returns:
        numpy.ndarray: Scan coordinates.
    """
    num_pts, m, n = y_meas.shape
    rounded_scan_loc = np.round(scan_loc)
    
    projection_coords = np.zeros((num_pts, 4), dtype=int)
    projection_coords[:, 0], projection_coords[:, 1] = rounded_scan_loc[:, 1] - m // 2, rounded_scan_loc[:, 1] + m // 2
    projection_coords[:, 2], projection_coords[:, 3] = rounded_scan_loc[:, 0] - n // 2, rounded_scan_loc[:, 0] + n // 2

    return projection_coords


def gen_tukey_2D_window(init_win, shape_param=0.5):
    """Generate a 2D Tukey window.

    Args:
        init_win (numpy.ndarray): Initialized output window.
        shape_param (float, optional): Shape parameter. Default is 0.5.

    Returns:
        numpy.ndarray: 2D Tukey window with a maximum value of 1.
    """
    # Initialization
    output = np.zeros_like(init_win)
    win_width = np.amin(init_win.shape)

    # Generate 1D Tukey window
    tukey_1d_win = signal.tukey(win_width, shape_param)
    tukey_1d_win_half = tukey_1d_win[int(len(tukey_1d_win) / 2) - 1:]
    x_coords = np.linspace(-win_width / 2, win_width / 2, win_width)
    y_coords = np.linspace(-win_width / 2, win_width / 2, win_width)

    # Generate 2D Tukey window from 1D Tukey window
    for x_idx in range(0, win_width):
        for y_idx in range(0, win_width):
            dist = int(np.sqrt(x_coords[x_idx] ** 2 + y_coords[y_idx] ** 2))
            if dist <= win_width / 2:
                output[x_idx, y_idx] = tukey_1d_win_half[dist]

    return output


def drop_line(y_meas, scan_pts):
    """Reduce scan points and measurements by skipping lines with a negative slope.

    Args:
        y_meas (numpy.ndarray): Pre-processed measurements.
        scan_pts (numpy.ndarray): Scan points.

    Returns:
        (numpy.ndarray, numpy.ndarray): Reduced scan points and associated measurements.
    """
    # Function to calculate the slope between two points
    def slope_between_pts(x, y):
        output = 0 if x[0] < y[0] else -1
        return output

    # Drop line with negative slopes
    pt_idx = np.arange(len(scan_pts))
    for idx in range(1, len(scan_pts)):
        slope = slope_between_pts(scan_pts[idx - 1], scan_pts[idx])
        if slope < 0:
            pt_idx = np.delete(pt_idx, np.where(pt_idx == idx))

    return y_meas[np.sort(pt_idx)], scan_pts[np.sort(pt_idx)]


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
        # Download the data from URL.
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


def find_center_offset(cmplx_img):
    """Find the unknown center of a symmetric pattern in an image.

    Args:
        cmplx_img (numpy.ndarray): Image with a symmetric pattern (such as a complex-valued probe function).

    Returns:
        list: Offset between the true image center and symmetric pattern center.
    """
    # Find the center of the given image
    c_0, c_1 = int(np.shape(cmplx_img)[0] / 2), int(np.shape(cmplx_img)[1] / 2)
    
    # Calculate peak and mean value of the magnitude image
    mag_img = np.abs(cmplx_img)
    peak_mag, mean_mag = np.amax(mag_img), np.mean(mag_img)
    
    # Find a group of points above the mean value
    pts = np.asarray(list(zip(*np.where(np.logical_and(mag_img >= mean_mag, mag_img <= peak_mag)))))
    
    # Find the unknown shifted center by averaging the group of points
    curr_center = np.mean(pts, axis=0)
    
    # Compute the offset between the unknown shifted center and the true center of the image
    center_offset = [int(c_0 - np.around(curr_center[0])), int(c_1 - np.around(curr_center[1]))]

    return center_offset


def correct_img_center(shifted_img, ref_img=None):
    """Shift a symmetric pattern to the center of an image.

    Args:
        shifted_img (numpy.ndarray): Image with unknown offsets.
        ref_img (numpy.ndarray, optional): Reference image for finding the unknown offsets.

    Returns:
        numpy.ndarray: Corrected image with the proper center location.
    """
    # Check reference image
    if ref_img is None:
        ref_img = np.copy(shifted_img)
        
    # Ensure the input image shape matches with the reference image
    try:
        shifted_img.shape == ref_img.shape
    except:
        print('Error: image shapes don\'t match')
        
    # Compute center offset using the reference image
    offset = find_center_offset(ref_img)
    
    # Shift the image back to the correct location
    output = np.roll(shifted_img, (offset[0], offset[1]), axis=(0, 1))

    return output