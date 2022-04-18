import tifffile as tiff
from scipy.ndimage import gaussian_filter
import sys, os, pyfftw, torch, re, scico, random
import scico.linop.optics as op
from utils.nrmse import *
from utils.display import *
import pandas as pd
import numpy as np
from scipy import signal


def int2float(arg):
    """
    Convert int argument to floating numbers.
    :param arg: int argument.
    :return: floating numbers.
    """
    if isinstance(arg, int):
        output = arg.astype(np.float64)
    else:
        output = arg
    return output


def float2cmplx(arg):
    """
    Convert float argument to complex.
    :param arg: float argument.
    :return: complex
    """
    if isinstance(arg, float):
        output = arg.astype(np.complex128)
    else:
        output = arg
    return output


def load_img(img_dir):
    """
    Read image from file path.
    :param img_dir: path of the file.
    :return: complex image array
    """
    # read tiff image
    image = tiff.imread(img_dir)
    real, imag, mag, pha = image[0], image[1], image[2], image[3]
    cmplx_img = real + 1j * imag

    return cmplx_img


def gen_scan_loc(obj, probe, num_pt, probe_spacing, randomization=True, max_offset=5, display=False, save_dir=None):
    """
    Function to generate the locations for scanning object.
    :param obj: complex sample image to be illuminated.
    :param probe: complex probe function.
    :param num_pt: number of scan points.
    :param probe_spacing: probe spacing between neighboring scan positions.
    :param randomization: add random offsets to generated scan points.
    :param display: option to display the scan pattern.
    :param save_dir: save generated scan points to given directory.
    :return: scan points for projections.
    """
    # check directories for saving files
    if save_dir is None:
        save_data = False
        save_fname = None
    else:
        save_data = True
        save_fname = save_dir + 'scan_pattern'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # initialization
    x, y = np.shape(obj)
    m, n = np.shape(probe)
    x_num = int(np.sqrt(num_pt))
    y_num = int(np.sqrt(num_pt))
    # generate scan point in raster order
    scan_pt = [((i - x_num / 2 + 1 / 2) * probe_spacing + x / 2, (j - y_num / 2 + 1 / 2) * probe_spacing + y / 2)
               for j in range(x_num)
               for i in range(y_num)]

    # generate random offset for each scan point
    if randomization:
        offset = np.random.uniform(low=-max_offset, high=(max_offset + 1), size=(num_pt, 2))
    else:
        offset = np.zeros((num_pt, 2), dtype=int)
    randomized_scan_pt = np.asarray(scan_pt + offset)

    if ((int(np.amin(randomized_scan_pt) - m / 2) < 0) or (int(np.amax(randomized_scan_pt) + n / 2 ) >= np.max([x, y]))):
        print('Exceeds the Boundary! Please extend the image or reduce probe spacing. ')

    if display:
        figure(num=None, figsize=(9, 4), dpi=60, facecolor='w', edgecolor='k')
        plt.subplot(121)
        plt.plot(np.asarray(scan_pt)[:, 0], np.asarray(scan_pt)[:, 1], 'o-')
        plt.title('scan points (d = {} px)'.format(probe_spacing))
        plt.axis([0, x, y, 0])
        plt.subplot(122)
        plt.plot(randomized_scan_pt[:, 0], randomized_scan_pt[:, 1], 'o-')
        plt.title('randomized points (max offset = {} px)'.format(max_offset))
        plt.axis([0, x, y, 0])
        plt.show()
        plt.clf()

    if save_data:
        df = pd.DataFrame({'FCx': randomized_scan_pt[:, 0], 'FCy': randomized_scan_pt[:, 1]})
        df.to_csv(save_dir + 'Translations.tsv.txt')
        figure(num=None, figsize=(9, 4), dpi=60, facecolor='w', edgecolor='k')
        plt.subplot(121)
        plt.plot(np.asarray(scan_pt)[:, 0], np.asarray(scan_pt)[:, 1], 'o-')
        plt.title('scan points (d = {} px)'.format(probe_spacing))
        plt.axis([0, x, y, 0])
        plt.subplot(122)
        plt.plot(randomized_scan_pt[:, 0], randomized_scan_pt[:, 1], 'o-')
        plt.title('randomized points (max offset = {} px)'.format(max_offset))
        plt.axis([0, x, y, 0])
        plt.savefig('{}.png'.format(save_fname))
        plt.clf()

    return randomized_scan_pt


def gen_syn_data(obj, probe, scan_coords, num_agts, add_noise=True, photon_rate=1e5, shot_noise_pm=0.5, fft_threads=1,
                 display=False, save_dir=None):
    """
    Function to simulate the ptychographic intensity measurements (diffraction pattern in far-field plane).
    :param obj: complex sample image.
    :param probe: complex probe function.
    :param scan_coords: coordinates of projections.
    :param num_agts: number of scan points.
    :param add_noise: option to add Poisson distributed noise including detector response and dark current noise.
    :param photon_rate: detector photon detection rate.
    :param shot_noise_pm: rate parameter of Poisson distributed dark current noise.
    :param fft_threads: number of threads for FFT.
    :param display: option to display the simulated diffraction pattern.
    :param save_dir: save generated diffraction patterns to given directory.
    :return: simulated intensity data.
    """

    # initialization
    m, n = probe.shape
    # extract patches x_j from full-sized object
    projected_patch = img2patch(obj, scan_coords, (num_agts, m, n))
    # take 2D DFT and generate noiseless measurements
    noiseless_data = np.abs(compute_ft(probe * projected_patch, threads=1)) ** 2

    # introduce photon noise
    if add_noise:
        # get peak signal value
        peak_signal_val = np.amax(noiseless_data)
        # calculate expected photon rate given peak signal value and peak photon rate
        expected_photon_rate = noiseless_data * photon_rate / peak_signal_val
        # poisson random values realization
        diffr_in_photon_ct = np.random.poisson(expected_photon_rate, (num_agts, m, n))
        # # convert to floating numbers
        # noisy_data = diffr_in_photon_ct * peak_signal_val / photon_rate
        # add dark current noise
        noisy_data = diffr_in_photon_ct + np.random.poisson(lam=shot_noise_pm, size=(num_agts, m, n))
        # return the numbers
        output = np.asarray(noisy_data, dtype=int)
    else:
        # return the floating numbers without poisson random variable realization
        output = np.asarray(noiseless_data)

    # check directories and save simulated data
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for j in range(num_agts):
            tiff.imwrite(save_dir + 'frame_data_{}.tiff'.format(j), output[j])

    # plot the simulated diffraction patterns
    if display:
        if add_noise:
            figure(num=None, figsize=(15, 10), dpi=100, facecolor='w', edgecolor='k')
            plt.subplot(221)
            plt.imshow(diffr_in_photon_ct[0] - expected_photon_rate[0], cmap='gray')
            plt.title('photon/detector noise')
            plt.axis('off')
            plt.colorbar()
            plt.subplot(222)
            plt.imshow(np.random.poisson(lam=shot_noise_pm, size=(m, n)), cmap='gray')
            plt.title('short/dark current noise')
            plt.axis('off')
            plt.colorbar()
            plt.subplot(223)
            plt.imshow(output[0], cmap='gray')
            plt.title('simulated noisy data')
            plt.colorbar()
            plt.axis('off')
            diffr_dbscale = 10 * np.log10(output + 1e-16)
            plt.subplot(224)
            plt.imshow(diffr_dbscale[0], cmap='gray', vmin=0)
            plt.title('simulated data (in decibel)')
            plt.axis('off')
            plt.colorbar()
            plt.show()
        else:
            figure(num=None, figsize=(15, 5), dpi=100, facecolor='w', edgecolor='k')
            plt.subplot(121)
            plt.imshow(output[0], cmap='gray')
            plt.title('simulated noiseless data')
            plt.colorbar()
            plt.axis('off')
            diffr_dbscale = 10 * np.log10(output + 1e-16)
            plt.subplot(122)
            plt.imshow(diffr_dbscale[0], cmap='gray', vmin=0)
            plt.title('simulated data (in decibel)')
            plt.axis('off')
            plt.colorbar()
            plt.show()

    return output


def load_measurement(fpath, display=False):
    """
    Function to read measurements from local path.
    :param fpath: data directory.
    :param display: option to display the data.
    :return: pre-processed measurements (square root of the recorded diffraction pattern).
    """
    # specify the order of measurement
    def key_func(fname):
        non_digits = re.compile("\D")
        output = int(non_digits.sub("", fname))
        return output

    # read the measurements
    meas_ls = []
    for fname in sorted(os.listdir(fpath), key=key_func):
        intensity_data = tiff.imread(os.path.join(fpath, fname))
        intensity_data[intensity_data < 0] = 0
        meas_ls.append(intensity_data)
    meas_ls = int2float(meas_ls)
    if display:
        figure(num=None, figsize=(15, 5), dpi=100, facecolor='w', edgecolor='k')
        plt.subplot(121)
        plt.imshow(meas_ls[0], cmap='gray')
        plt.title('measurement')
        plt.colorbar()
        plt.axis('off')
        meas_dbscale = 10 * np.log10(np.asarray(meas_ls) + 1e-16)
        plt.subplot(122)
        plt.imshow(meas_dbscale[0], cmap='gray', vmin=0)
        plt.title('measurement (in decibel)')
        plt.axis('off')
        plt.colorbar()
        plt.show()
    # take square root of non-negative diffraction data
    output = np.sqrt(np.asarray(meas_ls))

    return output


def gen_init_obj(img_ref, probe_ref, coords, measurement, formation, gauss_kernel_std=10, display=False):
    """
    Function to generate initial guess of complex transmittance for reconstruction.
    :param img_ref: ground truth complex object or reference images.
    :param probe_ref: known or estimated complex probe function.
    :param coords: coordinates of projections.
    :param measurement: intensity-only measurements (recorded diffraction patterns).
    :param formation: approach to generate initial guess.
    :param gauss_kernel_std: standard deviation of Gaussian kernel for low-pass filtering the initialized guess.
    :param display: option to display the initial guess of complex image.
    :return: formulated initial guess of complex transmittance of complex object.
    """
    # initialization
    num_agts, m, n = measurement.shape
    if formation == 'groundtruth' or formation == 'ground truth':
        output = img_ref
    elif formation == 'formulated':
        Ny = float(m * n)              # num of pixels in each measurement
        Nd = float(m * n)              # num of pixels in probe function
        Nx = float(m * n)              # num of pixels in each projection
        Drms = np.sqrt(np.linalg.norm(probe_ref) / Nd)
        patch_est = np.zeros(measurement.shape, dtype=np.complex128)      # estimate of each patch
        for j in range(num_agts):
            Yrms = np.sqrt(np.linalg.norm(measurement[j]) / Ny)
            patch_est[j] = np.sqrt(Ny / Nx) * Yrms / Drms
        norm = patch2img(np.ones(measurement.shape), coords, img_sz=img_ref.shape)
        obj_est = patch2img(patch_est, coords, img_sz=img_ref.shape, norm=norm)
        obj_est[obj_est == 0] = np.median(obj_est)
        # apply LPF to remove high frequencies
        output = gaussian_filter(np.abs(obj_est), sigma=gauss_kernel_std).astype(np.complex128)
    else:
        print('Error: The approach for creating initial guess does not exist!')
    if display:
        figure(num=None, figsize=(6.8, 2.4), dpi=100, facecolor='w', edgecolor='k')
        plt.subplot(121)
        plt.imshow(np.abs(output), cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('amplitude of init guess')
        plt.subplot(122)
        plt.imshow(np.angle(output), cmap='gray', vmax=np.pi, vmin=-np.pi)
        plt.colorbar()
        plt.axis('off')
        plt.title('phase of init guess (gauss_kernel_std = {})'.format(gauss_kernel_std))
        plt.show()
        plt.clf()

    return output


def gen_init_probe(obj_ref, probe_ref, coords, diffr, formation, sampling_interval=None,
                   source_wavelength=0.140891, propagation_dist=4e2, gauss_kernel_std=2, display=False):
    """
    Function to generate initial guess for reconstruction.
    :param obj_ref: complex object.
    :param probe_ref: reference probe function.
    :param coords: coordinates of projections.
    :param diffr: phaseless measurements (diffraction patterns).
    :param formation: approach to generate initial guess.
    :param sampling_interval: sampling interval at source plane.
    :param source_wavelength: illumination wavelength.
    :param propagation_dist:propagation distance.
    :param gauss_kernel_std: standard deviation of Guassian kernel for removing high frequencies.
    :param display: option to display the initial guess
    :return:initialized probe function
    """

    if sampling_interval is None:
        sampling_interval = 2 * source_wavelength
    if formation == 'groundtruth' or formation == 'ground truth':
        output = probe_ref
    elif formation == 'formulated':
        # considering init guess of object
        temp = np.zeros(diffr.shape, dtype=np.complex128)
        patch = img2patch(obj_ref, coords, diffr.shape)
        for i in range(len(diffr)):
            temp[i] = compute_ift(diffr[i]) / patch[i]
        probe_guess = np.average(temp, 0)
        num_agts, m, n = diffr.shape
        fres_op = op.FresnelPropagator(tuple([m, n]), dx=sampling_interval, k0=2 * np.pi / source_wavelength,
                                       z=propagation_dist, pad_factor=1, jit=True)
        # fres_op = op.FresnelPropagator(tuple([m, n]), dx=7.33808, k0=2 * np.pi / 0.140891, z=3e5, pad_factor=1,
        #                                jit=True)
        output = fres_op(probe_guess)
        output = gaussian_filter(np.real(output), sigma=gauss_kernel_std) + 1j * gaussian_filter(np.imag(output), sigma=gauss_kernel_std)
    if display:
        figure(num=None, figsize=(6.8, 2.4), dpi=100, facecolor='w', edgecolor='k')
        plt.subplot(121)
        plt.imshow(np.abs(output), cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('amplitude of init probe')
        plt.subplot(122)
        plt.imshow(np.angle(output), cmap='gray', vmax=np.pi, vmin=-np.pi)
        plt.colorbar()
        plt.axis('off')
        plt.title('phase of init probe')
        plt.show()
        plt.clf()

    return output


def patch2img(img_patch, coords, img_sz, norm=None):
    """
    Function to project image patch back to full-sized image with weights.
    :param img_patch: projected image patches.
    :param coords: coordinates of projections.
    :param img_sz: size of image.
    :param norm: weights for back projection.
    :return: full-sized complex image.
    """
    # initialization
    if norm is None:
        norm = np.ones(img_sz, dtype=np.complex128)
    img = np.zeros(img_sz, dtype=np.complex128)
    # back projection
    for j in range(len(img_patch)):
        img[coords[j, 0]:coords[j, 1], coords[j, 2]:coords[j, 3]] += img_patch[j]
    # normalization
    output = divide_cmplx_numbers(img, norm)

    return output


def img2patch(img, coords, patch_sz):
    """
    Function to extract image patches from full-sized image.
    :param img: the full-sized image.
    :param coords: coordinates of projections.
    :param patch_sz: size of each patch.
    :return: projected image patches.
    """
    num_agts, m, n = patch_sz
    output = np.zeros(patch_sz, dtype=np.complex128)
    for j in range(num_agts):
        output[j, :, :] = img[coords[j, 0]:coords[j, 1], coords[j, 2]:coords[j, 3]]

    return output


def compute_ft(input, threads=1):
    """
    Function to take 2D DFT of input using pyfftw.
    :param input: input (image).
    :param threads: number of threads for performing DFT using pyfftw..
    :return: 2D DFT of input (spectrum).
    """
    pyfftw_input = np.fft.fftshift(input, axes=(-2, -1))
    freq = pyfftw.interfaces.numpy_fft.fft2(pyfftw_input, s=None, axes=(-2, -1), norm='ortho', overwrite_input=False,
                                            planner_effort='FFTW_MEASURE', threads=threads, auto_align_input=True,
                                            auto_contiguous=True)
    output = np.fft.ifftshift(freq, axes=(-2, -1))

    return output


def compute_ift(input, threads=1):
    """
    Function to take 2D inverse DFT of input using pyfftw.
    :param input: input (spectrum).
    :param threads: number of threads for performing IDFT using pyfftw.
    :return: IFT of input (image).
    """
    pyfftw_input = np.fft.fftshift(input, axes=(-2, -1))
    image = pyfftw.interfaces.numpy_fft.ifft2(pyfftw_input, s=None, axes=(-2, -1), norm='ortho', overwrite_input=False,
                                              planner_effort='FFTW_MEASURE', threads=threads, auto_align_input=True,
                                              auto_contiguous=True)
    output = np.fft.ifftshift(image, axes=(-2, -1))

    return output


def scale(input, out_range):
    """
    This function scales input x into the given range.
    :param input: object.
    :param out_range: given range.
    :return: the scaled object.
    """
    in_range = np.amin(input), np.amax(input)
    if (in_range[1] - in_range[0]) == 0:
        return input
    else:
        y = (input - (in_range[1] + in_range[0]) / 2) / (in_range[1] - in_range[0])
        output = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

    return output


def divide_cmplx_numbers(cmplx_num, cmplx_denom):
    """
    Avoid error when dividing with complex numbers.
    :param cmplx_num: complex numerator.
    :param cmplx_denom: complex denominator.
    :return: result.
    """
    output = cmplx_num * np.conj(cmplx_denom) / (np.abs(cmplx_denom) ** 2 + 1e-15)

    return output


def save_tiff(input, save_dir):
    """
    Function to save complex image to given path.
    :param input: complex image.
    :param save_dir: provided directory.
    :return: N/A.
    """
    # save recon results
    img = np.asarray(input)
    cmplx_img_array = [np.real(img), np.imag(img), np.abs(img), np.angle(img)]
    tiff.imwrite(save_dir, np.asarray(cmplx_img_array))


def save_array(arr, save_dir):
    """
    Function to save array or list to given directory.
    :param arr: numpy array or list.
    :param save_dir: provided directory.
    :return: N/A.
    """
    f0 = open(save_dir, "wb")
    np.save(f0, arr)
    f0.close


def get_proj_coords_from_data(scan_loc, diffraction_data):
    num_agts, m, n = diffraction_data.shape
    rounded_scan_loc = np.round(scan_loc)
    projection_coords = np.zeros((num_agts, 4), dtype=int)
    projection_coords[:, 0], projection_coords[:, 1] = rounded_scan_loc[:, 1] - m / 2, rounded_scan_loc[:, 1] + m / 2
    projection_coords[:, 2], projection_coords[:, 3] = rounded_scan_loc[:, 0] - n / 2, rounded_scan_loc[:, 0] + n / 2

    return projection_coords


def gen_tukey_2D_window(init_window, alpha=0.5):
    """
    Funtion to generate a 2D Tukey window.
    :param init_window: init of output window.
    :param alpha: shape parameter of the Tukey window.
    :return: 2D Tukey window with the maximum value normalized to 1
    """
    # initialization
    output = np.zeros(init_window.shape)
    window_width = np.amin(init_window.shape)
    tukey_1d = signal.tukey(window_width, alpha)
    tukey_1d_half_window = tukey_1d[int(len(tukey_1d)/2)-1:]
    x_coords = np.linspace(-window_width/2, window_width/2, window_width)
    y_coords = np.linspace(-window_width/2, window_width/2, window_width)
    # generate 2D Tukey window from 1D Tukey window
    for x_idx in range(0, window_width):
        for y_idx in range(0, window_width):
            dist = int(np.sqrt(x_coords[x_idx]**2 + y_coords[y_idx]**2))
            if dist <= window_width/2:
                output[x_idx, y_idx] = tukey_1d_half_window[dist]

    return output

