import sys
from pathlib import Path
root_dir = Path(__file__).parent.absolute().parent.absolute().parent.absolute()
sys.path.append(str(root_dir))
print(root_dir)
sys.path.append(str(root_dir.parent.absolute()))
import argparse, yaml
import datetime as dt
from shutil import copyfile
from utils.utils import *
from ptycho import *
import h5py


'''
This file demonstrates the reconstruction of complex transmittance image and complex probe 
by processing the GoldBalls data. 
'''


def build_parser():
    parser = argparse.ArgumentParser(description='Ptychographic image reconstruction on real CuFoam data.')
    parser.add_argument('config_dir', type=str, help='Configuration file.', nargs='?', const='GoldBall_joint_recon.yaml',
                        default=os.path.join(root_dir, 'experiment/real_data_experiment/config/GoldBall_joint_recon.yaml'))
    return parser


def read_cxi_data(fpath,display=False):
    
    # load cxi file
    f = h5py.File(fpath, 'r')
    with f:
        cxi_version = np.array(f["cxi_version"])
        print('cxi_version = {}'.format(cxi_version))

        data = np.array(f["entry_1/data_1/data"])
        print('shape of measurements:', np.shape(data))

        dark_data = np.array(f["entry_1/instrument_1/detector_1/data_dark"])
        print('shape of dark data:', dark_data.shape)

        trans = np.array(f["entry_1/data_1/translation"])
        print('shape of translation:', trans.shape)

        distance = np.array(f["entry_1/instrument_1/detector_1/distance"])
        print('distance between detector and sample:', distance, 'm')

        corner_position = np.array(f["entry_1/instrument_1/detector_1/corner_position"])
        print('corner position:', corner_position)

        x_pixel_sz = np.array(f["entry_1/instrument_1/detector_1/x_pixel_size"])
        y_pixel_sz = np.array(f["entry_1/instrument_1/detector_1/y_pixel_size"])
        print('x_px_sz = {} m, y_px_sz = {} m'.format(x_pixel_sz, y_pixel_sz))

        name = np.array(f["entry_1/instrument_1/name"])
        print('instrument name:', name)

        # read source name
        name = np.array(f["entry_1/instrument_1/source_1/name"])
        print('source name:', name)

        # read source information
        energy_J = np.array(f["entry_1/instrument_1/source_1/energy"])
        energy_eV = energy_J * 6.241509e18
        print('source energy:', energy_J, 'J', 'or equivalently', energy_eV, 'eV')

    # compute source wavelength
    source_wavelength = 1239.84193 / energy_eV
    print('source wavelength:', source_wavelength, 'nm', 'or equivalently', source_wavelength * 1e-9, 'm')

    # compute image piexel size
    img_pixel_sz = source_wavelength * 1e-9 * distance / (data.shape[1] * x_pixel_sz)
    print('image pixel size =', img_pixel_sz, 'm')

    # convert translations from units of meters to units of pixels
    shifted_trans = np.copy(trans)
    shifted_trans[: ,1] += np.abs(np.amin(trans[:, 1]))
    trans_px = shifted_trans / img_pixel_sz + data.shape[1] / 2 + 12
    if display:
        plt.subplot(121)
        plt.plot(trans[:, 0], trans[:, 1], 'r.')
        plt.plot(trans[:, 0], trans[:, 1], 'b-')
        ax=plt.gca()
        ax.invert_yaxis()
        plt.title('translation (in meters)')
        plt.subplot(122)
        plt.plot(trans_px[:, 0], trans_px[:, 1], 'r.')
        plt.plot(trans_px[:, 0], trans_px[:, 1], 'b-')
        ax=plt.gca()
        ax.invert_yaxis()
        plt.title('translation (in pixels)')
        plt.show()
        plt.clf()
   
    # subtract dark data
    avg_dark_data = np.average(dark_data, axis=0)
    subtracted_data = data - avg_dark_data
    if display:
        plt.subplot(221)
        plt.imshow(dark_data[0], cmap='gray', vmax=400)
        plt.colorbar()
        plt.axis('off')
        plt.title('dark data')
        plt.subplot(222)
        plt.imshow(avg_dark_data, cmap='gray', vmax=400)
        plt.colorbar()
        plt.axis('off')
        plt.title('average of dark data')
        plt.subplot(223)
        plt.imshow(data[0], cmap='gray', vmax=6400)
        plt.colorbar()
        plt.axis('off')
        plt.title('diffraction pattern')
        plt.subplot(224)
        plt.imshow(subtracted_data[0], cmap='gray', vmax=6400)
        plt.colorbar()
        plt.axis('off')
        plt.title('subtracted data')
        plt.show()
        plt.clf()

    # center each diffraction pattern
    shifted_data = np.copy(subtracted_data)
    shifted_data = np.roll(shifted_data, [1, 15], axis=(1, 2))
    if display: 
        plt.subplot(121)
        plt.imshow(subtracted_data[0], cmap='gray', vmax=400)
        plt.colorbar()
        plt.title('subtracted data')
        plt.subplot(122)
        plt.imshow(shifted_data[0], cmap='gray', vmax=400)
        plt.colorbar()
        plt.title('shifted data')
        plt.show()
        plt.clf()

    return shifted_data, trans_px


def main():
    # arguments
    parser = build_parser()
    args = parser.parse_args()

    # load config file
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)

    # read data from config file
    data_dir = os.path.join(root_dir, config['data']['data_dir'])
    display = config['data']['display']
    out_dir = os.path.join(root_dir, config['output']['out_dir'])

    # Determine time stamp
    today_date = dt.date.today()
    date_time = dt.datetime.strftime(today_date, '%Y-%m-%d_%H_%M/')

    # Path with time stamp
    save_dir = os.path.join(out_dir, date_time)

    # Create the directory
    try:
        os.makedirs(save_dir, exist_ok=True)
        print("Output directory '%s' created successfully" % save_dir)
    except OSError as error:
        print("Output directory '%s' can not be created" % save_dir)

    # load data from file
    y_dir = os.path.join(data_dir, 'frame_data/')
    if os.path.isfile(y_dir):
        y_meas = load_measurement(y_dir, display=False)
    trans_dir = os.path.join(data_dir, 'Translation.tsv.txt')
    if os.path.isfile(trans_dir):
        trans_px = pd.read_csv(trans_dir, sep=None, engine='python', header=0)
    else:
        print('Error. Call function read_cxi_data().')

    # Default parameters
    rand_seed = 0
    np.random.seed(rand_seed)

    # calculate the coordinates of projections
    projection_coords = get_proj_coords_from_data(trans_px, diffract_data)
    
    # Generate formulated initial guess for reconstruction
    init_obj = np.ones((750, 750), dtype=np.complex128)
    init_probe = gen_init_probe(diffract_data, projection_coords, obj_ref=init_obj, display=True)
    
    # reconstruction arguments
    args = dict(init_obj=init_obj, init_probe=init_probe, 
               num_iter=config['recon']['num_iter'], joint_recon=config['recon']['joint_recon'])

    # ePIE recon
    obj_ss = config['ePIE']['obj_step_sz']
    probe_ss = config['ePIE']['probe_step_sz']
    epie_dir = save_dir + 'ePIE/obj_ss_{}_probe_ss_{}/'.format(obj_ss, probe_ss)
    epie_result = pie.epie_recon(y_meas, projection_coords, obj_step_sz=obj_ss, probe_step_sz=probe_ss, save_dir=epie_dir, **args)

    # PMACE recon
    obj_alpha = config['PMACE']['obj_alpha']
    probe_alpha = config['PMACE']['probe_alpha']
    pmace_dir = save_dir + 'PMACE/obj_alpha_{}_probe_alpha_{}/'.format(obj_alpha, probe_alpha)
    pmace_result = pmace.pmace_recon(y_meas, projection_coords, obj_pm=obj_alpha, probe_pm=probe_alpha, save_dir=pmace_dir)


#def plot_GoldBall_img(cmplx_img, img_title, display_win=None, display=False, save_dir=None):
#    """ Function to plot reconstruction results in this experiment. """
#    save_fname = None if (save_dir is None) else save_dir + 'recon_cmplx_img'
#    plot_cmplx_img(cmplx_img, img_title=img_title, ref_img=None,
#                   display_win=display_win, display=display, save_fname=save_fname,
#                   fig_sz=[8, 3], mag_vmax=2, mag_vmin=0, real_vmax=1.9, real_vmin=-1.3, imag_vmax=1.3, imag_vmin=-1.9)


if __name__ == '__main__':
    main()