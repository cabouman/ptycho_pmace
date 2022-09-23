from utils.utils import *


def plot_goldball_img(cmplx_img, display_win=None, display=False, save_dir=None):
    """ Function to plot reconstruction results in this experiment. """
    # check directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    # initialize window and determine area for showing and comparing images
    if display_win is None:
        display_win = np.ones_like(cmplx_img, dtype=np.complex64)
    non_zero_idx = np.nonzero(display_win)
    a, b = np.maximum(0, np.amin(non_zero_idx[0])), np.minimum(cmplx_img.shape[0], np.amax(non_zero_idx[0]) + 1)
    c, d = np.maximum(0, np.amin(non_zero_idx[1])), np.minimum(cmplx_img.shape[1], np.amax(non_zero_idx[1]) + 1)
    win_img = cmplx_img[a:b, c:d]
    # phase normalization
    norm_img = np.abs(win_img) * np.exp(1j * (np.angle(win_img) - np.mean(np.angle(win_img))))
    # plot real part of complex image
    real_plot = plt.imshow(np.real(norm_img), cmap='gray', vmax=140, vmin=70)
    plt.colorbar()
    plt.axis('off')
    if save_dir is not None:
        plt.savefig(save_dir + 'real_img', bbox_inches='tight', pad_inches=0, dpi=160)
    if display:
        plt.show()
    plt.clf()
    # plot imaginary part of complex image
    imag_plot = plt.imshow(np.imag(norm_img), cmap='gray', vmax=60, vmin=-30)
    plt.colorbar()
    plt.axis('off')
    if save_dir is not None:
        plt.savefig(save_dir + 'imag_img', bbox_inches='tight', pad_inches=0, dpi=160)
    if display:
        plt.show()
    plt.clf()