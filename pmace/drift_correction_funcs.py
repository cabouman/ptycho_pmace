import numpy as np
from .utils import *


def find_center_offset(cmplx_img):
    """
    Find the offset between the center of the input complex image and the true center of the image.

    Args:
        cmplx_img (numpy.ndarray): Complex-valued input image.

    Returns:
        list: A list containing the offset in the x and y directions, respectively.
    """
    # Find the center of the given image
    c_0, c_1 = np.shape(cmplx_img)[0] // 2, np.shape(cmplx_img)[1] // 2
    
    # Calculate peak and mean value of the magnitude image
    mag_img = np.abs(cmplx_img)
    peak_mag, mean_mag = np.amax(mag_img), np.mean(mag_img)
    
    # Find a group of points above the mean value
    pts = np.argwhere(np.logical_and(mag_img >= mean_mag, mag_img <= peak_mag))
    
    # Find the unknown shifted center by averaging the group of points
    curr_center = np.mean(pts, axis=0)
    
    # Compute the offset between the unknown shifted center and the true center of the image
    offset = [int(c_0 - np.around(curr_center[0])), int(c_1 - np.around(curr_center[1]))]

    return offset


def correct_img_center(shifted_img, ref_img=None):
    """
    Correct the center of an image by aligning it with a reference image.

    Args:
        shifted_img (numpy.ndarray): The image to be aligned/corrected.
        ref_img (numpy.ndarray): The reference image. If not provided, the input image is used as the reference.

    Returns:
        ndarray: The corrected image with the center aligned to the reference.
    """
    # Check the reference image
    if ref_img is None:
        ref_img = np.copy(shifted_img)
        
    # Compute center offset using the reference image
    offset = find_center_offset(ref_img)
    
    # Shift the image back to the correct location
    output = np.roll(shifted_img, (offset[0], offset[1]), axis=(0, 1))

    return output


def center_img_with_main_mode(cmplx_img, cmplx_probe_modes):
    """
    Center a complex image and a set of complex probe modes with respect to the first probe mode.

    Args:
      cmplx_img (numpy.ndarray): Complex input image.
      cmplx_probe_modes (list): Complex probe modes.

    Returns:
      ndarray: The centered complex image.
      list: A list containing the centered probe modes.
    """
    # Correct the center of the complex input image with respect to the first probe mode
    output_img = correct_img_center(cmplx_img, ref_img=cmplx_probe_modes[0])
    
    # Initialize an array to store the centered probe modes
    output_probe_modes = np.zeros_like(cmplx_probe_modes, dtype=np.complex64)
    
    # Loop through each probe mode and correct its center w.r.t the first probe mode
    for mode_idx in range(len(cmplx_probe_modes) - 1, -1, -1):
        output_probe_modes[mode_idx] = correct_img_center(cmplx_probe_modes[mode_idx], ref_img=cmplx_probe_modes[0])

    return output_img, output_probe_modes