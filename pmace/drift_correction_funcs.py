from .utils import *
from scipy.ndimage import center_of_mass


def find_center_offset(cmplx_img):
    """Find the offset between the center of complex-valued input image and the true center of the image.
    
    Args:
        cmplx_img (numpy.ndarray): Complex-valued input image.
    
    Returns:
        list: A list containing the offset in the x and y directions, respectively.
    """
    # Compute the center of the given image
    center_of_mass_coords = center_of_mass(np.abs(cmplx_img))

    # Determine the true center of the image
    true_center = (np.array(cmplx_img.shape) - 1) / 2

    # Calculate the offset
    offset = center_of_mass_coords - true_center

    # Round the offset to the nearest integer
    integer_offset = np.round(offset).astype(int)

    return integer_offset


def correct_img_center(input_img, integer_shift=False, ref_img=None):
    """Correct the center of an image by aligning it with a reference image.

    Args:
        shifted_img (numpy.ndarray): The image to be aligned/corrected.
        ref_img (numpy.ndarray): The reference image. If not provided, the input image is used as the reference.

    Returns:
        ndarray: The corrected image with the center aligned to the reference.
    """
    # Check the reference image
    if ref_img is None:
        ref_img = np.copy(input_img)
        
    # Compute center offset using the reference image
    offset = find_center_offset(ref_img)
    
    # Shift the image back to the correct location
    if integer_shift:
        output = np.roll(input_img, (int(np.round(offset[0])), int(np.round(offset[1]))), axis=(0, 1))
    else:
        output = shift_complex_array(input_img, offset[0], offset[1])

    return output


def shift_complex_array(arr, delta_y, delta_x):
    """
    Shifts a 2D complex array by non-integer displacement amounts using the Fourier shift theorem.
    
    Args:
        arr (numpy.ndarray): 2D complex array to be shifted
        delta_y (float): Shift amount in y-direction (rows)
        delta_x (float): Shift amount in x-direction (columns)
    
    Returns:
        numpy.ndarray: Shifted complex array with same shape as input
    """
    # Get original array dimensions
    ny, nx = arr.shape
    
    # Calculate padding width to avoid edge effects
    # Use 1.5 times the maximum shift or minimum of 20 pixels
    pad_width = max(int(1.5 * abs(delta_x)), int(1.5 * abs(delta_y)), 20)
    
    # Pad the array with zeros to avoid edge effects during shifting
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    
    # Compute 2D FFT of the padded array
    arr_fft = np.fft.fft2(padded_arr)
    
    # Get dimensions of padded array
    ly, lx = padded_arr.shape
    
    # Generate frequency grids for x and y directions
    x = np.fft.fftfreq(lx)
    y = np.fft.fftfreq(ly)
    xv, yv = np.meshgrid(x, y)
    
    # Create the shift kernel using the Fourier shift theorem
    shift_kernel = np.exp(-2j * np.pi * (xv * delta_x + yv * delta_y))
    
    # Apply the shift in frequency domain by multiplying with the shift kernel
    arr_fft_shifted = arr_fft * shift_kernel
    
    # Transform back to spatial domain using inverse FFT
    arr_shifted_padded = np.fft.ifft2(arr_fft_shifted)
    
    # Extract the original region from the padded shifted array
    startx = pad_width
    starty = pad_width
    arr_shifted = arr_shifted_padded[starty:starty + ny, startx:startx + nx]
    
    return arr_shifted
    
    
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