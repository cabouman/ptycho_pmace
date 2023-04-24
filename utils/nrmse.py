import numpy as np
import matplotlib.pyplot as plt
from math import *
from pylab import *


def compute_nrmse(img, ref_img, cstr=None):
    """Normalized Root-Mean-Square-Error (NRMSE) between two images.
    
    Args:
        img: complex image.
        ref_img: reference image.
        cstr: area for comparison.
        
    Returns:
        NRMSE between two images.
    """
    # assign region for calculating error
    img_adj = img if (cstr is None) else cstr * img
    ref_img_adj = ref_img if (cstr is None) else cstr * ref_img

    # compute MSE
    num_px = float(np.sum(np.abs(cstr))) if (cstr is not None) else float(img.shape[0] * img.shape[1])
    rmse = np.sqrt(np.sum(np.abs(img_adj - ref_img_adj) ** 2) / num_px)
    output = rmse / np.sqrt(np.sum(np.abs(ref_img_adj) ** 2) / num_px)

    return output


def pha_err(img, ref_img):
    """Phase error.
    
    The phase error is given by min | angle(img) - angle(gt) - 2*k*pi| where k \in {-1, 0, 1}
    
    Args:
        img: complex image.
        ref_img: complex reference image.
        
    Returns:
        phase error between complex images.
    """
    ang_diff = np.angle(ref_img) - np.angle(img)
    ang_diff[ang_diff > pi] -= pi
    ang_diff[ang_diff < -pi] += pi
    pha_err = np.minimum(ang_diff, ang_diff + 2*pi)

    return pha_err


def phase_norm(img, ref_img, cstr=None):
    """Phase normalization.
    
    The reconstruction is blind to absolute phase of ground truth image, so need to make
    phase shift to the reconstruction results given the known ground truth image.
    
    Args:
        img: the reconstruction needs phase normalization.
        ref_img: the known ground truth image or reference image.
        cstr: preconditioning window.
        
    Returns:
        phase normalized reconstruction.
    """
    # assign region for phase normalization
    img_adj = img if (cstr is None) else cstr * img
    ref_img_adj = ref_img if (cstr is None) else cstr * ref_img

    # phase normalization
    cmplx_scaler = np.sum(np.conj(img_adj) * ref_img_adj) / (np.linalg.norm(img_adj) ** 2)
    output = cmplx_scaler * img_adj

    return output.astype(np.complex64)

