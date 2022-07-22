import sys, os, argparse, yaml
from pathlib import Path
root_dir = Path(__file__).parent.absolute().parent.absolute()
sys.path.append(str(root_dir))
from shutil import copyfile
from utils.utils import *
from ptycho.oo_pmace import *


'''
This file demonstrates reconstruction of complex transmittance image using PMACE. 
'''


def build_parser():
    parser = argparse.ArgumentParser(description='Ptychographic image reconstruction using PMACE.')
    parser.add_argument('config_dir', type=str, help='Path to config file.', nargs='?', const='demo_pmace.yaml',
                        default='configs/demo_pmace.yaml')
    return parser


def main():
    # load config file and pass arguments
    parser = build_parser()
    args = parser.parse_args()

    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)

    obj_dir = os.path.join(root_dir, config['data']['obj_dir'])
    probe_dir = os.path.join(root_dir, config['data']['probe_dir'])
    data_dir = os.path.join(root_dir, config['data']['data_dir'])
    window_coords = config['data']['window_coords']
    save_dir = os.path.join(root_dir, config['recon']['out_dir'])

    # check directory
    try:
        os.makedirs(save_dir, exist_ok=True)
        print("Output directory '%s' created successfully" % save_dir)
    except OSError as error:
        print("Output directory '%s' can not be created" % save_dir)

    # load reference images from file
    ref_obj = load_img(obj_dir)
    ref_probe = load_img(probe_dir)

    # load measurements (diffraction patterns) from file and pre-process data
    y_meas = load_measurement(data_dir + 'frame_data/')

    # load scan positions
    scan_loc_file = pd.read_csv(data_dir + 'Translations.tsv.txt', sep=None, engine='python', header=0)
    scan_loc = scan_loc_file[['FCx', 'FCy']].to_numpy()

    # calculate coordinates of projections from scan positions
    patch_crds = get_proj_coords_from_data(scan_loc, y_meas)

    # formulate initial guess of complx object for reconstruction
    init_obj = gen_init_obj(y_meas, patch_crds, ref_obj.shape, ref_probe=ref_probe)

    # pre-define reconstruction region
    if window_coords is not None:
        xmin, xmax, ymin, ymax = window_coords[0], window_coords[1], window_coords[2], window_coords[3]
        recon_win = np.zeros(init_obj.shape)
        recon_win[xmin:xmax, ymin:ymax] = 1
    else:
        recon_win = None

    # reconstruction parameters
    num_iter = config['recon']['num_iter']
    joint_recon = config['recon']['joint_recon']
    alpha = config['recon']['data_fit_param']
    rho = config['recon']['rho']
    probe_exp = config['recon']['probe_exp']
    sigma = config['recon']['denoising_param']
    fig_args = dict(display_win=recon_win, save_dir=save_dir)

    # use class named PMACE to create object
    pmace_obj = PMACE(y_meas, patch_crds, 
                      init_obj, ref_obj=ref_obj, ref_probe=ref_probe, 
                      recon_win=recon_win, save_dir=save_dir, probe_exp=probe_exp)
    
    # PMACE recon
    pmace_result = pmace_obj.recon(num_iter=num_iter, joint_recon=joint_recon, 
                                   obj_data_fit_param=alpha, rho=rho, use_reg=False)
    plot_synthetic_img(pmace_result['object'], img_title='PMACE', **fig_args)
    
    # reg-PMACE recon
    pmace_obj.reset()
    reg_pmace_result = pmace_obj.recon(num_iter=num_iter, joint_recon=joint_recon, 
                                       obj_data_fit_param=alpha, rho=rho, use_reg=True, sigma=sigma)
    plot_synthetic_img(reg_pmace_result['object'], img_title='reg-PMACE', **fig_args)
    

def plot_synthetic_img(cmplx_img, img_title, display_win=None, save_dir=None):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    save_fname = None if (save_dir is None) else save_dir + '{}_recon_cmplx_img'.format(img_title)

    # plot complex image
    plot_cmplx_img(cmplx_img, img_title=img_title, display_win=display_win, save_fname=save_fname,
                   mag_vmax=1, mag_vmin=0.5, phase_vmax=0, phase_vmin=-np.pi/4,
                   real_vmax=1.1, real_vmin=0.8, imag_vmax=0, imag_vmin=-0.6)


if __name__ == '__main__':
    main()
