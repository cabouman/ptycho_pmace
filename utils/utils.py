import tifffile as tiff
from scipy.ndimage import gaussian_filter
import sys, os, pyfftw, torch, re
import scico.linop.optics as op
from .display import *
from .nrmse import *
import pandas as pd
import numpy as np
import multiprocessing as mp
from scipy import signal


def int2float(arg):
    """
    Convert int argument to floating numbers.
    Args:
        arg: int argument.
    Returns:
        floating numbers.
    """
    output = arg.astype(np.float64) if isinstance(arg, int) else arg

    return output


def float2cmplx(arg):
    """
    Convert float argument to complex.
    Args:
        arg: float argument.
    Returns:
        complex numbers.
    """
    output = arg.astype(np.complex64) if isinstance(arg, float) else arg

    return output


def load_img(img_dir):
    """
    Read image from file path.
    Args:
        img_dir: directory of the image.
    Returns:
        complex image array.
    """
    # read tiff image
    image = tiff.imread(img_dir)
    real, imag, mag, pha = image[0], image[1], image[2], image[3]
    cmplx_img = real + 1j * imag

    return cmplx_img


def gen_scan_loc(obj, probe, num_pt, probe_spacing, randomization=True, max_offset=5):
    """
    Function to generate scan locations.
    Args:
        obj: complex sample image to be scanned.
        probe: complex probe.
        num_pts: number of scan points.
        probe_spacing: probe spacing between neighboring scan positions.
        randomization: option to add random offsets to each scan point.
        max_offset: maximum offsets to be added to scan points along each dimension.
    Returns:
        generated scan points.
    """
    # initialization
    x, y = obj.shape
    m, n = probe.shape
    num_pt_x, num_pt_y = int(np.sqrt(num_pt)), int(np.sqrt(num_pt))

    # generate scan point in raster order
    scan_pt = [((i - num_pt_x / 2 + 1 / 2) * probe_spacing + x / 2, (j - num_pt_y / 2 + 1 / 2) * probe_spacing + y / 2)
               for j in range(num_pt_x)
               for i in range(num_pt_y)]

    # add random offsets to each scan point
    if randomization:
        offset = np.random.uniform(low=-max_offset, high=(max_offset + 1), size=(num_pt, 2))
        scan_pt = np.asarray(scan_pt + offset)

    if ((int(np.amin(scan_pt) - m / 2) < 0) or (int(np.amax(scan_pt) + n / 2 ) >= np.max([x, y]))):
        print('Warning: Scanning beyond valid region! Please extend image or reduce probe spacing. ')

    return scan_pt



def gen_syn_data(obj, probe, patch_bounds, add_noise=True, photon_rate=1e5, shot_noise_pm=0.5, save_dir=None):
    """
    Function to simulate the ptychographic intensity measurements (diffraction pattern in far-field plane).
    Args:
        obj: complex object.
        probe: complex probe.
        patch_bounds: scan coordinates of projections.
        add_noise: option to add noise to data.
        photon_rate: rate of photon detection at detector.
        shot_noise_pm: expected number of Poisson distributed dark current noise.
        save_dir: directory for saving generated diffraction patterns.
    Returns:
        simualted ptychographic data.
    """
    # initialization
    m, n = probe.shape
    num_pts = len(patch_bounds)
    # extract patches x_j from full-sized object
    projected_patches = img2patch(obj, patch_bounds, (num_pts, m, n))
    # take 2D DFT and generate noiseless measurements
    noiseless_data = np.abs(compute_ft(probe * projected_patches)) ** 2
    # introduce photon noise
    if add_noise:
        # get peak signal value
        peak_signal_val = np.amax(noiseless_data)
        # calculate expected photon rate given peak signal value and peak photon rate
        expected_photon_rate = noiseless_data * photon_rate / peak_signal_val
        # poisson random values realization
        meas_in_photon_ct = np.random.poisson(expected_photon_rate, (num_pts, m, n))
        # add dark current noise
        noisy_data = meas_in_photon_ct + np.random.poisson(lam=shot_noise_pm, size=(num_pts, m, n))
        # return the numbers
        output = np.asarray(noisy_data, dtype=int)
    else:
        # return the floating numbers without poisson random variable realization
        output = np.asarray(noiseless_data)

    # check directories and save simulated data
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for j in range(num_pts):
            tiff.imwrite(save_dir + 'frame_data_{}.tiff'.format(j), output[j])

    return output


def load_measurement(fpath):
    """
    Function to read measurements from path and pre-process data.
    Args:
        fpath: file directory.
    Returns:
        pre-processed measurement (square root of non-negative data).
    """
    # specify the order of measurement
    def key_func(fname):
        non_digits = re.compile("\D")
        output = int(non_digits.sub("", fname))
        return output

    # read the measurements and remove negative values
    meas_ls = []
    for fname in sorted(os.listdir(fpath), key=key_func):
        y_meas = tiff.imread(os.path.join(fpath, fname))
        y_meas[y_meas < 0] = 0
        meas_ls.append(y_meas)
    meas_ls = int2float(meas_ls)
    
    # take square root of non-negative diffraction data
    output = np.sqrt(np.asarray(meas_ls))

    return output


def gen_init_obj(y_meas, coords, img_sz, ref_probe=None, lpf_sigma=10):
    """
    Function to formulate initial guess of complex object for reconstruction.
    Args:
        y_meas: pre-processed diffraction patterns.
        coords: coordinates of projections.
        img_sz: size of full complex image.
        ref_probe: known or estimated complex probe function.
        lpf_sigma: standard deviation of Gaussian kernel for low-pass filtering the initialized guess.
    Returns:
        formulated initial guess of complex transmittance image.
    """
    if ref_probe is None:
        ref_probe = np.ones_like(y_meas[0]).astype(np.complex64)

    patch_rms = [[np.sqrt(np.linalg.norm(y_meas[j]) / np.linalg.norm(ref_probe))] * np.ones_like(ref_probe)
                 for j in range(len(y_meas))]
    img_wgt = patch2img(np.ones_like(y_meas), coords, img_sz=img_sz)
    init_obj = patch2img(patch_rms, coords, img_sz=img_sz, norm=img_wgt)
    init_obj[init_obj == 0] = np.median(init_obj)
    # apply LPF to remove high frequencies
    output = gaussian_filter(np.abs(init_obj), sigma=lpf_sigma)

    return output.astype(np.complex64)


def gen_init_probe(y_meas, coords, ref_obj, fres_propagation=False, sampling_interval=None,
                   source_wl=0.140891, propagation_dist=4e2, lpf_sigma=2):
    """
    Function to formulate initial guess of complex probe for joint reconstruction on ptychographic data.
    Args:
        y_meas: pre-processed diffraction patterns.
        coords: coordinates of projections.
        ref_obj: ground truth complex object or reference images.
        fres_propagation: option to fresnel propagate initialized probe.
        sampling_interval: sampling interval at source plane.
        source_wl: illumination wavelength.
        propagation_dist:propagation distance.
        lpf_sigma: standard deviation of Guassian kernel for removing high frequencies.
    Returns:
        formualted initial guess of complex probe.
    """
    if sampling_interval is None:
        sampling_interval = 2 * source_wl
    # formulate init probe
    patch = img2patch(ref_obj, coords, y_meas.shape)
    tmp = [compute_ift(y_meas[j]) / patch[j] for j in range(len(y_meas))]
    init_probe = np.average(tmp, axis=0)
    if fres_propagation:
        m, n = init_probe.shape
        fres_op = op.FresnelPropagator(tuple([m, n]), dx=sampling_interval, k0=2 * np.pi / source_wl, z=propagation_dist)
    	# fres_op = op.FresnelPropagator(tuple([m, n]), dx=7.33808, k0=2 * np.pi / 0.140891, z=3e5, pad_factor=1, jit=True)
        output = fres_op(init_probe)
    else:
        output = init_probe
    # apply LPF to remove high frequencies
    output = gaussian_filter(np.real(output), sigma=lpf_sigma) + 1j * gaussian_filter(np.imag(output), sigma=lpf_sigma)

    return output.astype(np.complex64)


def patch2img(img_patch, coords, img_sz, norm=None):
    """
    Function to project image patch back to full-sized image with weights.
    Args:
        img_patch: projected image patches.
        coords: coordinates of projections.
        img_sz: size of full image.
        norm: normalization weight.
    Returns:
        full-sized complex image.
    """
    # initialization
    if norm is None:
        norm = np.ones(img_sz, dtype=np.complex64)
    img = np.zeros(img_sz, dtype=np.complex64)

    # back projection
    for j in range(len(img_patch)):
        img[coords[j, 0]:coords[j, 1], coords[j, 2]:coords[j, 3]] += img_patch[j]

    # normalization
    output = divide_cmplx_numbers(img, norm)

    return output


def img2patch(img, coords, patch_sz):
    """
    Function to extract image patches from full-sized image.
    Args:
        img: full-sized image.
        coords: coordinates of projections.
        patch_sz: size of output patches.
    Returns:
        projected image patches.
    """
    # initialization
    output = np.zeros(patch_sz, dtype=np.complex64)

    # take projections
    for j in range(len(output)):
        output[j, :, :] = img[coords[j, 0]:coords[j, 1], coords[j, 2]:coords[j, 3]]

    return output


def compute_ft(input_array, threads=None):
    """
    Function to take 2D DFT of input using pyfftw.
    Args:
        input_array: input.
        threads: number of threads for performing DFT using pyfftw.
    Returns:
        result of 2D DFT.
    """
    if threads is None:
        threads = mp.cpu_count()
    # DFT    
    a = np.fft.fftshift(input_array.astype(np.complex64), axes=(-2, -1))
    b = np.zeros_like(a)
    fft_object = pyfftw.FFTW(a, b, axes=(-2, -1), normalise_idft=False, ortho=True, direction='FFTW_FORWARD', threads=threads)
    output = np.fft.ifftshift(fft_object(), axes=(-2, -1))

    return output.astype(np.complex64)


def compute_ift(input_array, threads=None):
    """
    Function to take 2D inverse DFT of input using pyfftw.
    Args:
        input: input.
        threads: number of threads for performing IDFT using pyfftw.
    Returns:
        results of 2D inverse DFT.
    """
    if threads is None:
        threads = mp.cpu_count()
    # IDFT
    a = np.fft.fftshift(input_array.astype(np.complex64), axes=(-2, -1))
    b = np.zeros_like(a)
    ifft_object = pyfftw.FFTW(a, b, axes=(-2, -1), normalise_idft=False, ortho=True, direction='FFTW_BACKWARD', threads=threads)
    output = np.fft.ifftshift(ifft_object(), axes=(-2, -1))

    return output.astype(np.complex64)


def scale(input_obj, out_range):
    """
    This function scales input  into the given range.
    Args:
        input_obj: object to be scaled.
        out_range: scale range.
    Returns:
        scaled input.
    """
    in_range = np.amin(input_obj), np.amax(input_obj)
    if (in_range[1] - in_range[0]) == 0:
        return input_obj
    else:
        y = (input_obj - (in_range[1] + in_range[0]) / 2) / (in_range[1] - in_range[0])
        output = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

    return output


def divide_cmplx_numbers(cmplx_num, cmplx_denom):
    """
    Take division regarding complex numbers.
    Args:
        cmplx_num: complex numerator.
        cmplx_denom: complex denominator.
    Returns:
        result.
    """
    tol = np.amax([1e-3, np.amax(np.abs(cmplx_denom)) * 1e-6])
    cmplx_denom[np.abs(cmplx_denom) < tol] = tol
    output = cmplx_num / cmplx_denom
    #output = cmplx_num * np.conj(cmplx_denom) / (np.abs(cmplx_denom) ** 2 + 1e-15)

    return output


def save_tiff(cmplx_img, save_dir):
    """
    Function to save complex image to given path.
    Args:
        input: complex image.
        save_dir: save .tiff image to specific directory.
    """
    # save recon results
    img = np.asarray(cmplx_img)
    img_array = [np.real(img), np.imag(img), np.abs(img), np.angle(img)]
    tiff.imwrite(save_dir, np.asarray(img_array))


def save_array(arr, save_dir):
    """
    Function to save array or list to given directory.
    Args:
        arr: numpy array or list.
        save_dir: directory for saving array.
    """
    f = open(save_dir, "wb")
    np.save(f, arr)
    f.close


def get_proj_coords_from_data(scan_loc, y_meas):
    """
    Function to obtain projection coordinates from scan points.
    Args:
        scan_loc: scan points.
        y_meas: diffraction patterns.
    Returns:
        scan coordinates.
    """
    num_pts, m, n = y_meas.shape
    rounded_scan_loc = np.round(scan_loc)
    projection_coords = np.zeros((num_pts, 4), dtype=int)
    projection_coords[:, 0], projection_coords[:, 1] = rounded_scan_loc[:, 1] - m / 2, rounded_scan_loc[:, 1] + m / 2
    projection_coords[:, 2], projection_coords[:, 3] = rounded_scan_loc[:, 0] - n / 2, rounded_scan_loc[:, 0] + n / 2

    return projection_coords


def gen_tukey_2D_window(init_win, shape_param=0.5):
    """
    Funtion to generate a 2D Tukey window.
    Args:
        init_window: initialized output window.
        shape_param: shape parameter.
    Returns:
        2D Tukey window with maximum value 1.
    """
    # initialization
    output = np.zeros_like(init_win)
    win_width = np.amin(init_win.shape)

    # generate 1D Tukey window
    tukey_1d_win = signal.tukey(win_width, shape_param)
    tukey_1d_win_half = tukey_1d_win[int(len(tukey_1d_win) / 2) - 1:]
    x_coords = np.linspace(-win_width/2, win_width/2, win_width)
    y_coords = np.linspace(-win_width/2, win_width/2, win_width)
    # generate 2D Tukey window from 1D Tukey window
    for x_idx in range(0, win_width):
        for y_idx in range(0, win_width):
            dist = int(np.sqrt(x_coords[x_idx]**2 + y_coords[y_idx]**2))
            if dist <= win_width/2:
                output[x_idx, y_idx] = tukey_1d_win_half[dist]

    return output


def drop_line(y_meas, scan_pts):
    """
    Function to reduce scan points and measurements by skipping lines with negative slope.
    Args:
        y_meas: diffraction patterns.
        scan_pts: scan points.
    Returns:
        reduced scan points and corresponding diffraction patterns.
    """
    # slope between two points
    def slope_between_pts(x, y):
        output = 0 if x[0] < y[0] else -1
        return output

    # drop line with negative slopes
    pt_idx = np.arange(len(scan_pts))
    for idx in range(1, len(scan_pts)):
        slope = slope_between_pts(scan_pts[idx - 1], scan_pts[idx])
        if slope < 0:
            pt_idx = np.delete(pt_idx, np.where(pt_idx == idx))
    return y_meas[np.sort(pt_idx)], scan_pts[np.sort(pt_idx)]
