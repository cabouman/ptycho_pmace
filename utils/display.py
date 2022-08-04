from .nrmse import *
import matplotlib.pyplot as plt


def plot_scan_pt(scan_pt):
    """ 
    Function to plot scan points.
    Args:
        scan_pt: scan points.
    """
    plt.plot(np.asarray(scan_pt)[:, 0], np.asarray(scan_pt)[:, 1], 'o-')
    plt.title('scan points')
    plt.show()
    plt.clf()


def plot_diffr_data(y_meas):
    """
    Function to plot diffraction pattern in original scale and dbscales.
    Args:
        y_meas: one diffraction pattern.
    """
    plt.subplot(211)
    plt.imshow(y_meas, cmap='gray')
    plt.title('diffraction data')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(212)
    meas_dbscale = 10 * np.log10(y_meas + 1e-16)
    plt.imshow(meas_dbscale, cmap='gray', vmin=0)
    plt.title('data in decibel')
    plt.axis('off')
    plt.colorbar()

    plt.show()
    plt.clf()


def plot_nrmse(nrmse_ls, title, label, abscissa=None, step_sz=15, fig_sz=[10, 4.8], display=False, save_fname=None):
    """
    Function to plot the nrmse versus number of iterations.
    Args:
        nrmse_ls: nrmse lists.
        title: corresponding title for each nrmse list.
        label: axis labels.
    """
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
        if isinstance(nrmse_ls, dict):
            idx = 0
            xmin, xmax, ymin, ymax = 1, 0, 1, 0
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
    Args:
        cmplx_img: complex image.
        img_title: title of complex image.
        ref_img: reference image.
        display_win: pre-defined window for showing images.
        display: option to show images.
        save_fname: save images to designated file directory.
        fig_sz: size of image plots.
        mag_vmax: max value for showing image magnitude.
        mag_vmin: min value for showing image magnitude.
        phase_vmax: max value for showing image phase.
        phase_vmin: max value for showing image phase.
        real_vmax: max value for showing real part of image.
        real_vmin: max value for showing real part of image
        imag_vmax: max value for showing imaginary part of image
        imag_vmin: max value for showing imaginary magnitude.
    """
    # plot error images if reference image is provided
    show_err_img = False if (ref_img is None) or (np.linalg.norm(cmplx_img - ref_img) < 1e-9) else True

    # initialize window and determine area for showing and comparing images
    if display_win is None:
        display_win = np.ones_like(cmplx_img, dtype=np.complex128)
    non_zero_idx = np.nonzero(display_win)
    blk_idx = [np.amin(non_zero_idx[0]), np.amax(non_zero_idx[0])+1, np.amin(non_zero_idx[1]), np.amax(non_zero_idx[1])+1]
    cmplx_img_rgn = cmplx_img[blk_idx[0]:blk_idx[1], blk_idx[2]:blk_idx[3]]
    if ref_img is not None:
        ref_img_rgn = ref_img[blk_idx[0]:blk_idx[1], blk_idx[2]:blk_idx[3]]

    # display the amplitude and phase images
    plt.figure(num=None, figsize=(fig_sz[0], fig_sz[1]), dpi=400, facecolor='w', edgecolor='k')
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
        ang_err = pha_err(cmplx_img_rgn, ref_img_rgn)
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
