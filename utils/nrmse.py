import numpy as np
import matplotlib.pyplot as plt
from math import *
from pylab import *


def compute_nrmse(img, ref_img, cstr=None):
    """
    Function to calculate the Normalized Root-Mean-Square-Error (NRMSE) between two images x and y.
    :param img: complex image.
    :param ref_img: reference image.
    :param cstr: area for comparison.
    :return: NRMSE between two images.
    """
    if cstr is None:
        cstr = np.ones(img.shape)

    num_px = float(np.sum(cstr))
    ref_img = cstr * ref_img
    img = cstr * img
    # compute MSE
    mse = np.sum(np.abs(img - ref_img) ** 2) / num_px
    # rmse = np.sqrt(mse)
    temp = np.sqrt(np.sum(np.abs(ref_img) ** 2) / num_px)
    output = np.divide(np.sqrt(mse), temp, where=(temp != 0))

    return output


def pha_err(gt, img):
    """
    The phase error = min | angle(img) - angle(gt) - 2*k*pi| where k \in {-1, 0, 1}
    :param gt: the ground truth image.
    :param img: the recovered image.
    :return: the phase error between the ground truth image and recovered image.
    """
    ang_diff = np.angle(gt) - np.angle(img)
    ang_diff[ang_diff > pi] -= pi
    ang_diff[ang_diff < -pi] += pi
    pha_error = np.minimum(ang_diff, ang_diff + 2*pi)

    return pha_error


def phase_norm(img, img_ref):
    """
    The reconstruction is blind to absolute phase of ground truth image, so need to make
    phase shift to the reconstruction results given the known ground truth image.
    :param img: the reconstruction needs phase normalization.
    :param img_ref: the known ground truth image or reference image.
    :return: the phase normalized reconstruction.
    """
    # phase normalization
    cmplx_scaler = np.sum(np.conj(img) * img_ref) / (np.linalg.norm(img) ** 2)
    output = cmplx_scaler * img

    return output

