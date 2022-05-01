from .nrmse import *
import matplotlib.pyplot as plt


def plot_diffr_data(diffr, fig_sz=[12, 8]):
    """
    The function to plot diffraction pattern in original scale and dbscales.
    :param diffr: diffraction pattern.
    :param fig_sz: size of figure.
    :return: visualization of diffraction pattern.
    """
    plt.figure(num=None, figsize=(fig_sz[0], fig_sz[1]), dpi=400, facecolor='w', edgecolor='k')
    plt.subplot(221)
    plt.imshow(diffr, cmap='gray')
    plt.title('phaseless data (diffraction pattern)')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(222)
    diffr_dbscale = 10 * np.log10(diffr + 1e-16)
    plt.imshow(diffr_dbscale, cmap='gray', vmin=0)
    plt.title('diffraction pattern in decibel')
    plt.axis('off')
    plt.colorbar()

    sqrt_diffr = np.sqrt(np.asarray(diffr))
    plt.subplot(223)
    plt.imshow(sqrt_diffr, cmap='gray')
    plt.title('sqrt of diffraction pattern')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(224)
    sqrt_diffr_db = 10 * np.log10(sqrt_diffr + 1e-16)
    plt.imshow(sqrt_diffr_db, cmap='gray', vmin=0)
    plt.title('sqrt of diffraction pattern in decibel')
    plt.axis('off')
    plt.colorbar()
    plt.show()
    plt.clf()


def plot_nrmse(nrmse_ls, title, label, abscissa=None, step_sz=15, fig_sz=[10, 4.8], display=False, save_fname=None):
    """ Plot the nrmse vs number of iterations"""
    xlabel, ylabel = label[0], label[1]
    plt.figure(num=None, figsize=(fig_sz[0], fig_sz[1]), dpi=100, facecolor='w', edgecolor='k')
    if np.all(abscissa == None):
        if isinstance(nrmse_ls, dict):
            length = 0
            for line_label, line in nrmse_ls.items():
                length = np.maximum(length, len(line))
                plt.semilogy(line, label=line_label)
        else:
            line = np.asarray(nrmse_ls)
            length = len(line)
            line_label = label[2]
            plt.semilogy(line, label=line_label)
        plt.xticks(np.arange(length, step=step_sz), np.arange(1, length + 1, step=step_sz))
        plt.legend(loc='best')
        plt.legend(loc='best')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.grid(True)
        if save_fname is not None:
            plt.savefig('{}.png'.format(save_fname))
        if display:
            plt.show()
        plt.clf()
    else:
        # fig, ax = plt.subplots()
        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        if isinstance(nrmse_ls, dict):
            idx = 0
            # legend = []
            xmin = 1
            xmax = 0
            ymax = 0
            ymin = 1
            for line_label, line in nrmse_ls.items():
                xmin = np.minimum(xmin, np.amin(np.asarray(abscissa[idx])))
                xmax = np.maximum(xmax, np.amax(np.asarray(abscissa[idx])))
                ymin = np.mininum(ymin, np.amin(np.asarray(line)))
                ymax = np.maximum(ymax, np.amax(np.asarray(line)))
                plt.semilogy(line, abscissa[idx], label=line_label)
                idx = idx + 1
        else:
            line = np.asarray(nrmse_ls)
            plt.semilogy(line, abscissa, label=label[2])
            xmin, xmax = np.amin(abscissa), np.max(abscissa)
            ymin, ymax = np.amin(line), np.amax(line)
        plt.xlim([np.maximum(xmin, 0), xmax+1e-2])
        plt.ylim([ymin-1e-2, ymax+1e-2])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if save_fname is not None:
            plt.savefig('{}.png'.format(save_fname))
        if display:
            plt.show()
        plt.clf()


def plot_cmplx_img(cmplx_img, img_title='img', ref_img=None, display_win=None, display=False, save_fname=None,
                   fig_sz=[8, 3], mag_vmax=1, mag_vmin=0, phase_vmax=np.pi, phase_vmin=-np.pi,
                   real_vmax=1, real_vmin=0, imag_vmax=0, imag_vmin=-1):
    """
    Function to plot the complex object images and error images compared with reference image.
    :param cmplx_img: complex image.
    :param img_title: title of complex image.
    :param ref_img: reference image.
    :param display_win: pre-defined window for showing images.
    :param display: option to show images.
    :param save_fname: save images to designated file directory.
    :param fig_sz: size of image plots.
    :param mag_vmax: max value for showing image magnitude.
    :param mag_vmin: min value for showing image magnitude.
    :param phase_vmax: max value for showing image phase.
    :param phase_vmin: max value for showing image phase.
    :param real_vmax: max value for showing real part of image.
    :param real_vmin: max value for showing real part of image
    :param imag_vmax: max value for showing imaginary part of image
    :param imag_vmin: max value for showing imaginary magnitude.
    :return:
    """

    # plot error images if reference image is provided
    show_err_img = False if (ref_img is None) or (np.linalg.norm(cmplx_img - ref_img) < 1e-9) else True

    # initialize window and determine area for showing and comparing images
    if display_win is None:
        display_win = np.ones_like(cmplx_img, dtype=np.complex128)
    non_zero_idx = np.nonzero(display_win)
    blk_idx = [np.amin(non_zero_idx[0]), np.amax(non_zero_idx[0])+1, np.amin(non_zero_idx[1]), np.amax(non_zero_idx[1])+1]
    cmplx_img_rgn = cmplx_img[blk_idx[0]:blk_idx[1], blk_idx[2]:blk_idx[3]]
    ref_img_rgn = ref_img[blk_idx[0]:blk_idx[1], blk_idx[2]:blk_idx[3]]

    # display the amplitude and phase images
    plt.figure(num=None, figsize=(fig_sz[0], fig_sz[1]), dpi=100, facecolor='w', edgecolor='k')
    # mag of reconstructed complex image
    plt.subplot(2, 4, 1)
    plt.imshow(np.abs(cmplx_img_rgn), cmap='gray', vmax=mag_vmax, vmin=mag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'mag of {}'.format(img_title))
    # phase of reconstructed complex image
    plt.subplot(2, 4, 2)
    plt.imshow(np.angle(cmplx_img_rgn), cmap='gray', vmax=phase_vmax, vmin=phase_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'phase of {}'.format(img_title))
    # real part of reconstructed complex image
    plt.subplot(2, 4, 3)
    plt.imshow(np.real(cmplx_img_rgn), cmap='gray', vmax=real_vmax, vmin=real_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('real of {}'.format(img_title))
    # imag part of reconstructed complex image
    plt.subplot(2, 4, 4)
    plt.imshow(np.imag(cmplx_img_rgn), cmap='gray', vmax=imag_vmax, vmin=imag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('imag of {}'.format(img_title))

    if show_err_img:
        # amplitude of difference between complex reconstruction and ground truth image
        plt.subplot(2, 4, 5)
        plt.imshow(np.abs(cmplx_img_rgn - ref_img_rgn), cmap='gray', vmax=0.2, vmin=0)
        plt.title(r'error - amp')
        plt.colorbar()
        plt.axis('off')
        # phase difference between complex reconstruction and ground truth image
        ang_err = pha_err(ref_img_rgn, cmplx_img_rgn)
        plt.subplot(2, 4, 6)
        plt.imshow(ang_err, cmap='gray', vmax=np.pi/2, vmin=-np.pi/2)
        plt.colorbar()
        plt.axis('off')
        plt.title(r'phase error')
        # real part of error image between complex reconstruction and ground truth image
        err = cmplx_img_rgn - ref_img_rgn
        plt.subplot(2, 4, 7)
        plt.imshow(np.real(err), cmap='gray', vmax=0.2, vmin=-0.2)
        plt.colorbar()
        plt.axis('off')
        plt.title('err - real')
        # image part of error between complex reconstruction and ground truth image
        plt.subplot(2, 4, 8)
        plt.imshow(np.imag(err), cmap='gray', vmax=0.2, vmin=-0.2)
        plt.colorbar()
        plt.axis('off')
        plt.title('err - imag')

    if save_fname is not None:
        plt.savefig('{}.png'.format(save_fname))
    if display:
        plt.show()
    plt.clf()

