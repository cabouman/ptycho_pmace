import sys
import os
import re
import glob
import tarfile
import urllib.request
import multiprocessing as mp

import pyfftw
import tifffile as tiff
import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy import signal
from scipy.ndimage import gaussian_filter
import h5py


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
        Complex-valued number.
    """
    output = arg.astype(np.complex64) if isinstance(arg, float) else arg
    
    return output


def load_img(img_dir):
    """Load a complex image from a TIFF file.

    Args:
        img_dir (str): Directory of the TIFF image.

    Returns:
        numpy.ndarray: Complex image array.
    """
    # Read a TIFF image
    img = tiff.imread(img_dir)
    
    # Separate real and imaginary parts
    real, imag = img[0], img[1]
    
    # Create a complex image array
    cmplx_img = real + 1j * imag
                 
    return cmplx_img


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

    # Read the measurements
    work_dir = os.listdir(fpath)
    
    # Remove system files
    if '.DS_Store' in work_dir:
        work_dir.remove('.DS_Store')
    
    # Sort files based on their numeric indices
    work_dir.sort(key=key_func)
    
    # Initialize empty list to store measurements
    meas_ls = []
    
    # Loop through each measurement
    for fname in work_dir:
        # Load measurements
        meas = tiff.imread(os.path.join(fpath, fname))
        # Remove negtive values
        meas[meas < 0] = 0      
        # Take square root of the non-negative values
        meas_ls.append(np.sqrt(int2float(meas)))

    # Stack the measurements
    output = np.asarray(meas_ls)
        
    return output


def traverse_cxi_file(f_dir):
    """Traverse an HDF5 CXI file and print information.

    Args:
        f_dir (str): Path to the CXI file.
    """
    with h5py.File(f_dir, 'r') as cxi_file:
        
        # Function to recursively traverse the HDF5 file
        def traverse(group, current_path):
            for name, item in group.items():
                # Print the current directory path
                print(os.path.join(current_path, name))            
                
                # Check if the item is a directory
                if isinstance(item, h5py.Group):
                    # Recursively traverse the subdirectory
                    traverse(item, os.path.join(current_path, name))
                else:
                    # print(f"  - {os.path.join(current_path, name)}")
                    dataset_path = os.path.join(current_path, name)
                    print(f"  - {dataset_path}")
                    
                    # Check if the dataset has attributes (keys)
                    keys = list(cxi_file[dataset_path].attrs.keys())
                    if keys:
                        print(f"  - {dataset_path}")
                        print(f"    Keys: {keys}")
                        print(f"    Values: {cxi_file[dataset_path][:]}")
                        print("=" * 60)
                    else:
                        # If no keys, assume it's a simple array dataset
                        array = np.array(cxi_file[dataset_path])
                        print(f"    Values: {array}")
                        print("=" * 60)

        # Start traversal from the root
        traverse(cxi_file, '/')
        

def gen_scan_loc(cmplx_obj, cmplx_probe, num_pt, probe_spacing, randomization=True, max_offset=5):
    """Simulate scan locations.

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
    # Get image dimensions
    x, y = cmplx_obj.shape
    m, n = cmplx_probe.shape
    
    # Calculate number of points along each dimension
    num_pt_x, num_pt_y = int(np.sqrt(num_pt)), int(np.sqrt(num_pt))

    # Generate scan points in raster order
    scan_pt = [((i - num_pt_x / 2 + 1 / 2) * probe_spacing + x / 2, 
                (j - num_pt_y / 2 + 1 / 2) * probe_spacing + y / 2)
               for j in range(num_pt_x)
               for i in range(num_pt_y)]

    # Add random offsets to each scan point
    if randomization:
        offset = np.random.uniform(low=-max_offset, high=(max_offset + 1), size=(num_pt, 2))
        scan_pt = np.asarray(scan_pt + offset)

    # Check if scanning points are beyond valid region
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
    # Expand the dimensions of complex-valued probe
    if cmplx_probe is not None:
        probe_modes = np.expand_dims(cmplx_probe, axis=0) if np.array(cmplx_probe).ndim == 2 else cmplx_probe
    else:
        raise ValueError("Invalid probe")
        
    # Get image dimensions
    m, n = probe_modes[0].shape
    num_pts = len(patch_bounds)
    
    # Extract patches from full-sized object
    projected_patches = img2patch(cmplx_obj, patch_bounds)

    # Initialize data array
    noiseless_data = np.zeros_like(projected_patches, dtype=np.float32)

    # Warmup
    _ = compute_ft(probe_modes[0] * projected_patches[0])
    
    # Take 2D DFT and generate noiseless measurements
    for probe_mode in probe_modes:
        noiseless_data += np.abs(compute_ft(probe_mode * projected_patches)) ** 2

    # Introduce photon noise
    if add_noise:
        # Get peak signal value
        peak_signal_val = np.amax(noiseless_data)
        # Calculate expected photon rate given peak signal value and peak photon rate
        expected_photon_rate = noiseless_data * peak_photon_rate / peak_signal_val
        # Poisson random values realization
        photon_count = np.random.poisson(expected_photon_rate, (num_pts, m, n))
        # Add dark current noise
        noisy_data = photon_count + np.random.poisson(lam=shot_noise_pm, size=(num_pts, m, n))
        # Return the noisy data
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


def gen_init_obj(y_meas, patch_crds, img_sz, ref_probe, lpf_sigma=1):
    """Formulate an initial guess of a complex object for reconstruction.

    Args:
        y_meas (numpy.ndarray): Pre-processed intensity measurements.
        patch_crds (numpy.ndarray): Coordinates of projections.
        img_sz (tuple): Size of the full complex image (rows, columns).
        ref_probe (numpy.ndarray): Known or estimated complex probe function.
        lpf_sigma (float): Standard deviation of the Gaussian kernel for low-pass filtering the initialized guess.

    Returns:
        numpy.ndarray: The formulated initial guess of object transmittance image.
    """
    # Calculate RMS of patches
    patch_rms = np.sqrt(np.linalg.norm(y_meas, axis=tuple([-2, -1])) / np.linalg.norm(ref_probe))

    # Construct array of patches
    patch_arr = np.tile(patch_rms, (ref_probe.shape[0], ref_probe.shape[1], 1))
    
    # Convert dimensions of array to (num_patch, m, n)
    img_patch = np.transpose(patch_arr, (2, 0, 1))

    # Project patches to compose full-sized image with proper weights
    img_wgt = patch2img(np.ones_like(y_meas), patch_crds, img_sz=img_sz)
    init_obj = patch2img(img_patch, patch_crds, img_sz=img_sz, norm_wgt=img_wgt)
    
    # Apply LPF to remove high frequencies
    init_obj[init_obj == 0] = np.median(init_obj)
    output = gaussian_filter(np.abs(init_obj), sigma=lpf_sigma)

    return output.astype(np.complex64)


def fresnel_propagation(field, wavelength, distance, dx):
    """Perform Fresnel propagation of a wavefront from a source plane to an observation plane.

    Args:
        field (numpy.ndarray): The complex wavefront at the source plane, represented as a 2D array.
        wavelength (float): The wavelength of the wave in the same units as the distance and dx.
        distance (float): The propagation distance from the source plane to the observation plane.
        dx (float): The sampling interval in the source plane, i.e., the distance between adjacent points.

    Returns:
        numpy.ndarray: A 2D array representing the complex wavefront at the observation plane.
    """
    # Number of points in each dimension
    N = field.shape[0]

    # Spatial frequency coordinates
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    # Quadratic phase factor for Fresnel propagation (Fresnel kernel in the frequency domain)
    H = np.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))

    # Perform Fourier transform of the source field, apply the Fresnel kernel, and then inverse Fourier transform
    output = np.fft.ifft2(np.fft.fft2(field) * H)

    return output


def gen_init_probe(y_meas, patch_crds, ref_obj, lpf_sigma=1, 
                   fres_propagation=False, sampling_interval=None, source_wl=None, propagation_dist=None):
    """Formulate an initial complex probe from the initialized object and data.

    Args:
        y_meas (numpy.ndarray): Pre-processed diffraction patterns.
        patch_crds (numpy.ndarray): Coordinates of projections.
        ref_obj (numpy.ndarray): Ground truth complex object or reference images.
        lpf_sigma (float): Standard deviation of the Gaussian kernel for removing high frequencies.
        fres_propagation (bool): Option to Fresnel propagate the initialized probe.
        sampling_interval (float): Sampling interval at the source plane.
        source_wl (float): Illumination wavelength.
        propagation_dist (float): Propagation distance.
        
    Returns:
        numpy.ndarray: The formulated initial guess of a complex probe.
    """
    # Ensure wavelength, distance, and dx are provided for Fresnel propagation
    if fres_propagation and (source_wl is None or propagation_dist is None):
        raise ValueError("Wavelength and distance must be provided for Fresnel propagation.")
    
    # Initialization
    k0  = 2 * np.pi / source_wl
    Nx, Ny = y_meas.shape[-2], y_meas.shape[-1]
    
    if sampling_interval is None:
        sampling_interval = np.sqrt(2 * np.pi * propagation_dist / (k0 * Nx))
        
    # Formulate init probe
    patch = img2patch(ref_obj, patch_crds)
    init_probe = np.mean(divide_cmplx_numbers(compute_ift(y_meas), patch), axis=0)
    
    # Fresnel propagation initialized probe
    if fres_propagation:
        init_probe = fresnel_propagation(init_probe, source_wl, propagation_dist, sampling_interval)

    # Apply Gaussian Low Pass Filter to both real and imaginary parts
    filtered_real = gaussian_filter(np.real(init_probe), sigma=lpf_sigma)
    filtered_imag = gaussian_filter(np.imag(init_probe), sigma=lpf_sigma)
    output = filtered_real + 1j * filtered_imag

    return output.astype(np.complex64)


def generate_initial_guesses(y_meas, patch_crds, config, ref_object=None):
    """Generate initial guesses for object and probe reconstruction.
    
    Args:
        y_meas (numpy.ndarray): The measurements.
        patch_crds (numpy.ndarray): Coordinates of the projections.
        config (dict): Configuration parameters.
        ref_object (numpy.ndarray): Reference object image.
        
    Returns:
        init_obj (numpy.ndarray): Initial guess for the object.
        init_probe (numpy.ndarray): Initial guess for the probe.
    """
    # Retrieve initialization parameters from configuration file
    fresnel_propagation = config['initialization']['fresnel_propagation']
    source_wl = float(config['initialization']['source_wavelength'])
    propagation_dist = float(config['initialization']['propagation_distance'])
    sampling_interval = float(config['initialization']['sampling_interval'])

    if ref_object is not None:
        init_obj = np.ones_like(ref_object, dtype=ref_object.dtype)
    # else:

    # Initialize probe
    init_probe = gen_init_probe(y_meas, patch_crds, init_obj, 
                                fres_propagation=fresnel_propagation, sampling_interval=sampling_interval,
                                source_wl=source_wl, propagation_dist=propagation_dist)
    
    # Initialize object
    init_obj = gen_init_obj(y_meas, patch_crds, init_obj.shape, ref_probe=init_probe)

    return init_obj, init_probe


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


def img2patch(full_img, patch_coords):
    """Extract image patches from full-sized image.

    Args:
        img (numpy.ndarray): Full-sized image.
        coords (numpy.ndarray): Coordinates of projections.

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


def compute_ft(input_array, threads=None):
    """Compute the 2D Discrete Fourier Transform (DFT) of an input array.
    
    Args:
        input_array (numpy.ndarray): The input 2D array for DFT computation.
        threads(int): number of threads for performing DFT using pyfftw.
        
    Returns:
        numpy.ndarray: The result of the 2D DFT.
    """
    if threads is None:
        threads = mp.cpu_count()

    a = np.fft.fftshift(input_array.astype(np.complex64), axes=(-2, -1))
    b = np.zeros_like(a)
    fft_object = pyfftw.FFTW(a, b, axes=(-2, -1), normalise_idft=False, ortho=True, direction='FFTW_FORWARD', threads=threads)
    output = np.fft.ifftshift(fft_object(), axes=(-2, -1))

    return output


def compute_ift(input_array, threads=None):
    """Compute the 2D Inverse Discrete Fourier Transform (IDFT) of an input array.
    
    Args:
        input_array (numpy.ndarray): The input 2D array for IDFT computation.
        threads(int): number of threads for performing DFT using pyfftw.
                
    Returns:
        numpy.ndarray: The result of the 2D IDFT.
    """
    if threads is None:
        threads = mp.cpu_count()

    a = np.fft.fftshift(input_array.astype(np.complex64), axes=(-2, -1))
    b = np.zeros_like(a)
    ifft_object = pyfftw.FFTW(a, b, axes=(-2, -1), normalise_idft=False, ortho=True, direction='FFTW_BACKWARD', threads=threads)
    output = np.fft.ifftshift(ifft_object(), axes=(-2, -1))
    
    return output


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

        numpy.ndarray: Result of the division.
    """
    # Use epsilon to avoid divsion by zero
    fro_norm = np.sqrt(np.sum(np.abs(cmplx_denom) ** 2))
    epsilon = 1e-6 * fro_norm / np.sqrt(np.array(cmplx_denom).size)

    # Calculate the inverse of the denominator, considering epsilon
    denom_inv = np.conj(cmplx_denom) / (np.abs(cmplx_denom) ** 2 + epsilon)

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