import sys
from pathlib import Path
root_dir = Path(__file__).parent.absolute().parent.absolute().parent.absolute()
sys.path.append(str(root_dir))
import argparse, yaml
import datetime as dt
from shutil import copyfile
from ptycho_pmace.utils.utils import *
from ptycho_pmace.ptycho import *


'''
This file demonstrates the reconstruction of complex transmittance image by processing the synthetic data. 
'''


def build_parser():
    parser = argparse.ArgumentParser(description='Ptychographic image reconstruction on real CuFoam data.')
    parser.add_argument('config_dir', type=str, help='Configuration file.', nargs='?', const='CuFoam_data.yaml',
                        default=os.path.join(root_dir, 'experiment/real_data_experiment/config/CuFoam_data.yaml'))
    return parser


def plot_CuFoam_img(cmplx_img, img_title, display_win=None, display=False, save_dir=None):
    """ Function to plot reconstruction results in this experiment. """
    save_fname = None if (save_dir is None) else save_dir + 'recon_cmplx_img'
    plot_cmplx_img(cmplx_img, img_title=img_title, ref_img=None,
                   display_win=display_win, display=display, save_fname=save_fname,
                   fig_sz=[8, 3], mag_vmax=2, mag_vmin=0, real_vmax=1.9, real_vmin=-1.3, imag_vmax=1.3, imag_vmin=-1.9)


def main():
    # Arguments
    parser = build_parser()
    args = parser.parse_args()

    # Load config file
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)

    # Read data from config file
    obj_dir = os.path.join(root_dir, config['data']['obj_dir'])
    probe_dir = os.path.join(root_dir, config['data']['probe_dir'])
    data_dir = os.path.join(root_dir, config['data']['data_dir'])
    display = config['data']['display']
    window_coords = config['data']['window_coords']
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

    # Default parameters
    rand_seed = 0
    np.random.seed(rand_seed)

    # Load ground truth images from file
    obj_ref = load_img(obj_dir)
    probe_ref = load_img(probe_dir)

    # Load intensity only measurements(data) from file and pre-process the data
    diffraction_data = load_measurement(data_dir + 'frame_data/', display=display)

    # Load scan points
    scan_loc_data = pd.read_csv(data_dir + 'Translations.tsv.txt', sep=None, engine='python', header=0)
    scan_loc = scan_loc_data[['FCx', 'FCy']].to_numpy()

    # calculate the coordinates of projections
    projection_coords = get_proj_coords_from_data(scan_loc, diffraction_data)

    # Generate formulated initial guess for reconstruction
    init_obj = gen_init_obj(diffraction_data, projection_coords, obj_ref=obj_ref, probe_ref=probe_ref)

    # Produce the cover/window for comparison
    if window_coords is not None:
        xmin, xmax, ymin, ymax = window_coords[0], window_coords[1], window_coords[2], window_coords[3]
        cstr_win = np.zeros(init_obj.shape)
        cstr_win[xmin:xmax, ymin:ymax] = 1
    else:
        cstr_win = None

    # Reconstruction parameters
    num_iter = config['recon']['num_iter']
    joint_recon = config['recon']['joint_recon']

    # ePIE recon
    obj_step_sz = config['ePIE']['obj_step_sz']
    epie_dir = save_dir + 'ePIE/obj_step_sz_{}/'.format(obj_step_sz)
    epie_result = pie.epie_recon(diffraction_data, projection_coords, init_obj=init_obj,
                                 obj_ref=obj_ref, probe_ref=probe_ref, num_iter=num_iter, obj_step_sz=obj_step_sz,
                                 joint_recon=joint_recon, cstr_win=cstr_win, save_dir=epie_dir)
    # Plot reconstructed image
    plot_CuFoam_img(epie_result['obj_revy'], img_title='ePIE', display_win=cstr_win, display=display, save_dir=epie_dir)

    # # Accelerated Wirtinger Flow (AWF) recon
    awf_dir = save_dir + 'AWF/'
    awf_result = wf.wf_recon(diffraction_data, projection_coords, init_obj=init_obj, obj_ref=obj_ref, probe_ref=probe_ref,
                             accel=True, num_iter=num_iter, joint_recon=joint_recon, cstr_win=cstr_win, save_dir=awf_dir)
    # Plot reconstructed image
    plot_CuFoam_img(awf_result['obj_revy'], img_title='AWF', display_win=cstr_win, display=display, save_dir=awf_dir)

    # SHARP recon
    relax_pm = config['SHARP']['relax_pm']
    sharp_dir = save_dir + 'SHARP/relax_pm_{}/'.format(relax_pm)
    sharp_result = sharp.sharp_recon(diffraction_data, projection_coords, init_obj, obj_ref=obj_ref, probe_ref=probe_ref,
                                     num_iter=num_iter, relax_pm=relax_pm, joint_recon=joint_recon, cstr_win=cstr_win,
                                     save_dir=sharp_dir)
    # Plot reconstructed image
    plot_CuFoam_img(sharp_result['obj_revy'], img_title='SHARP', display_win=cstr_win, display=display, save_dir=sharp_dir)

    # SHARP+ recon
    sharp_plus_pm = config['SHARP_plus']['relax_pm']
    sharp_plus_dir = save_dir + 'SHARP_plus/relax_pm_{}/'.format(sharp_plus_pm)
    sharp_plus_result = sharp.sharp_plus_recon(diffraction_data, projection_coords, init_obj, obj_ref=obj_ref,
                                               probe_ref=probe_ref, num_iter=num_iter, relax_pm=sharp_plus_pm,
                                               joint_recon=joint_recon, cstr_win=cstr_win, save_dir=sharp_plus_dir)
    # Plot reconstructed image
    plot_CuFoam_img(sharp_plus_result['obj_revy'], img_title='SHARP', display_win=cstr_win, display=display, save_dir=sharp_plus_dir)

    # PMACE recon
    alpha = config['PMACE']['alpha']                   # noise-to-signal ratio
    rho = config['PMACE']['rho']                       # Mann averaging parameter
    probe_exp = config['PMACE']['probe_exponent']      # probe exponent
    pmace_dir = save_dir + 'PMACE/alpha_{}_rho_{}_probe_exp_{}/'.format(alpha, rho, probe_exp)
    pmace_result = pmace.pmace_recon(diffraction_data, projection_coords, init_obj, obj_ref=obj_ref, probe_ref=probe_ref,
                                     num_iter=num_iter, obj_pm=alpha, rho=rho, probe_exp=probe_exp, add_reg=False,
                                     joint_recon=joint_recon, cstr_win=cstr_win, save_dir=pmace_dir)
    # Plot reconstructed image
    plot_CuFoam_img(pmace_result['obj_revy'], img_title='PMACE', display_win=cstr_win, display=display,
                    save_dir=pmace_dir)

    # reg-PMACE recon
    alpha = config['reg-PMACE']['alpha']
    rho = config['reg-PMACE']['rho']
    probe_exp = config['reg-PMACE']['probe_exponent']
    reg_wgt = config['reg-PMACE']['reg_wgt']            # regularization weight
    noise_std = config['reg-PMACE']['noise_std']        # denoising parameter
    prior = config['reg-PMACE']['prior']                # prior model, eg. bm3d or DnCNN
    reg_pmace_dir = save_dir + 'reg_PMACE/reg_wgt_{}_noise_std_{}/'.format(reg_wgt, noise_std)
    reg_pmace_result = pmace.pmace_recon(diffraction_data, projection_coords, init_obj, obj_ref=obj_ref, probe_ref=probe_ref,
                                         num_iter=num_iter, obj_pm=alpha, rho=rho, probe_exp=probe_exp,
                                         add_reg=True, reg_wgt=reg_wgt, noise_std=noise_std, prior=prior,
                                         joint_recon=joint_recon, cstr_win=cstr_win, save_dir=reg_pmace_dir)
    # Plot reconstructed image
    plot_CuFoam_img(reg_pmace_result['obj_revy'], img_title='reg-PMACE', display_win=cstr_win, display=display,
                    save_dir=reg_pmace_dir)

    # Save config file to output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile(args.config_dir, save_dir + 'config.yaml')

    # Convergence plots
    xlabel, ylabel = 'Number of iteration', 'NRMSE value in log scale'
    line_label = 'nrmse'
    nrmse = {'ePIE': epie_result['obj_err'],
             'AWF': awf_result['obj_err'],
             'SHARP': sharp_result['obj_err'],
             'SHARP+': sharp_plus_result['obj_err'],
             'PMACE': pmace_result['obj_err'],
             'reg-PMACE': reg_pmace_result['obj_err']}
    plot_nrmse(nrmse, title='Convergence plots of PMACE', label=[xlabel, ylabel, line_label],
               step_sz=10, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_plot')


if __name__ == '__main__':
    main()
