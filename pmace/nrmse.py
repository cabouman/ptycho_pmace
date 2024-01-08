import numpy as np
import matplotlib.pyplot as plt
from math import *
from pylab import *


def compute_mse(input_img, ref_img):
    """Compute Mean Squared Error (MSE) between two images.

    Args:
        input_img (numpy.ndarray): Complex-valued image for comparison.
        ref_img (numpy.ndarray): Reference image for comparison.

    Returns:
        float: The computed MSE between two images.
    """
    if input_img.shape != ref_img.shape:
        raise ValueError("Input and reference image must have the same shape for MSE calculation.")
        
    # Compute Mean Square Error (MSE) for all elements
    mse = np.mean(np.abs(input_img - ref_img) ** 2)

    return mse


def compute_nrmse(input_img, ref_img, cstr=None):
    """Compute Normalized Root-Mean-Square-Error (NRMSE) between two images.

    This function calculates the NRMSE between the provided image and reference image.

    Args:
        input_img (numpy.ndarray): Complex-valued image for comparison.
        ref_img (numpy.ndarray): Reference image for comparison.
        cstr (numpy.ndarray, optional): Area for comparison. If provided, only this region will be considered.

    Returns:
        float: The computed NRMSE between two images.
    """
    # Assign a region for calculating the error
    img_rgn = input_img if (cstr is None) else cstr * input_img
    ref_rgn = ref_img if (cstr is None) else cstr * ref_img

    # Compute Mean Square Error (MSE) for all elements
    num_px = np.sum(np.abs(cstr)) if (cstr is not None) else ref_img.size
    mse = np.sum(np.abs(img_rgn - ref_rgn) ** 2) / num_px

    # Compute the NRMSE
    rmse = np.sqrt(mse)
    nrmse = rmse / np.sqrt((np.sum(np.abs(ref_rgn) ** 2)) / num_px)

    return nrmse


def pha_err(img, ref_img):
    """Calculate the phase error between complex images.

    The phase error is determined as the minimum of | angle(img) - angle(ref_img) - 2*k*pi| where k belongs to {-1, 0, 1}.

    Args:
        img (numpy.ndarray): Complex-valued image for comparison.
        ref_img (numpy.ndarray): Reference image for comparison.

    Returns:
        numpy.ndarray: Phase error between two images.
    """
    ang_diff = np.angle(ref_img) - np.angle(img)
    ang_diff[ang_diff > pi] -= pi
    ang_diff[ang_diff < -pi] += pi
    pha_err = np.minimum(ang_diff, ang_diff + 2 * pi)

    return pha_err


def phase_norm(img, ref_img, cstr=None):
    """Perform phase normalization on reconstructed image.

    Since reconstruction is blind to the absolute phase of the ground truth image, this function applies a phase shift
    to the reconstruction results given the known ground truth image.

    Args:
        img (numpy.ndarray): The reconstruction that needs phase normalization.
        ref_img (numpy.ndarray): The known ground truth image or reference image.
        cstr (numpy.ndarray, optional): Preconditioning window. If provided, only this region will be considered.

    Returns:
        numpy.ndarray: The phase-normalized reconstruction as a complex image (dtype: np.complex64).
    """
    # Assign a region for phase normalization
    img_rgn = img if (cstr is None) else cstr * img
    ref_rgn = ref_img if (cstr is None) else cstr * ref_img

    # Phase normalization
    cmplx_scaler = np.sum(np.conj(img_rgn) * ref_rgn) / (np.linalg.norm(img_rgn) ** 2)
    output = cmplx_scaler * img_rgn

    return output.astype(np.complex64)