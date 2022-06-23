import sys
from pathlib import Path
root_dir = Path(__file__).parent.absolute().parent.absolute()
sys.path.append(str(root_dir))
import os, argparse, yaml
import datetime as dt
from shutil import copyfile
from utils.utils import *
from ptycho import *


'''
This file demonstrates the reconstruction of complex transmittance image by processing the synthetic data. 
'''


def build_parser():
    parser = argparse.ArgumentParser(description='Ptychographic image reconstruction using various approaches.')

    parser.add_argument('config_dir', type=str, help='Path to config file.', nargs='?', const='configs/demo_pmace.yaml',
                        default=os.path.join(root_dir, 'configs/demo_pmace.yaml'))
    return parser


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
    save_dir = os.path.join(root_dir, config['output']['out_dir'])

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
    init_obj = gen_init_obj(diffraction_data, projection_coords, obj_ref=obj_ref, probe_ref=probe_ref, display=display)

    # Produce the cover/window for comparison
    if window_coords is not None:
        xmin, xmax, ymin, ymax = window_coords[0], window_coords[1], window_coords[2], window_coords[3]
        display_win = np.zeros(init_obj.shape)
        display_win[xmin:xmax, ymin:ymax] = 1
    else:
        display_win = None

    def plot_synthetic_img(cmplx_img, img_title, ref_img, display_win=None, display=False, save_dir=None):
        """ Function to plot reconstruction results in this experiment. """
        save_fname = None if (save_dir is None) else save_dir + 'recon_cmplx_img'
        plot_cmplx_img(cmplx_img, img_title=img_title, ref_img=ref_img,
                       display_win=display_win, display=display, save_fname=save_fname,
                       fig_sz=[8, 3], mag_vmax=1, mag_vmin=0.5, phase_vmax=0, phase_vmin=-np.pi/4,
                       real_vmax=1.1, real_vmin=0.8, imag_vmax=0, imag_vmin=-0.6)

    # Reconstruction parameters
    num_iter = config['recon']['num_iter']

    # ePIE recon
    obj_step_sz = config['ePIE']['obj_step_sz']
    epie_dir = save_dir + config['ePIE']['out_dir'] + 'obj_step_sz_{}/'.format(obj_step_sz)
    epie_result = pie.epie_recon(diffraction_data, projection_coords, init_obj=init_obj,
                                 obj_ref=obj_ref, probe_ref=probe_ref, num_iter=num_iter, obj_step_sz=obj_step_sz,
                                 joint_recon=False, cstr_win=display_win, save_dir=epie_dir)
    # Plot reconstructed image
    plot_synthetic_img(epie_result['obj_revy'], img_title='ePIE', ref_img=obj_ref,
                       display_win=display_win, display=display, save_dir=epie_dir)

    # Wirtinger Flow (WF) recon
    wf_dir = save_dir + config['WF']['out_dir']
    wf_result = wf.wf_recon(diffraction_data, projection_coords, init_obj, obj_ref=obj_ref, probe_ref=probe_ref,
                            accel=False, num_iter=num_iter, joint_recon=False, cstr_win=display_win, save_dir=wf_dir)
    # Plot reconstructed image
    plot_synthetic_img(wf_result['obj_revy'], img_title='WF', ref_img=obj_ref,
                       display_win=display_win, display=display, save_dir=wf_dir)

    # Acclerated Wirtinger Flow (AWF) recon
    awf_dir = save_dir + config['AWF']['out_dir']
    awf_result = wf.wf_recon(diffraction_data, projection_coords, init_obj, obj_ref=obj_ref, probe_ref=probe_ref,
                             accel=True, num_iter=num_iter, joint_recon=False, cstr_win=display_win, save_dir=awf_dir)
    # Plot reconstructed image
    plot_synthetic_img(awf_result['obj_revy'], img_title='AWF', ref_img=obj_ref,
                       display_win=display_win, display=display, save_dir=awf_dir)

    # SHARP recon
    relax_pm = config['SHARP']['relax_pm']
    sharp_dir = save_dir + config['SHARP']['out_dir'] + 'relax_pm_{}/'.format(relax_pm)
    sharp_result = sharp.sharp_recon(diffraction_data, projection_coords, init_obj, obj_ref=obj_ref, probe_ref=probe_ref,
                                     num_iter=num_iter, relax_pm=relax_pm, joint_recon=False, cstr_win=display_win, save_dir=sharp_dir)
    # Plot reconstructed image
    plot_synthetic_img(sharp_result['obj_revy'], img_title='SHARP', ref_img=obj_ref,
                       display_win=display_win, display=display, save_dir=sharp_dir)

    # SHARP+ recon
    relax_pm = config['SHARP_plus']['relax_pm']
    sharp_plus_dir = save_dir + config['SHARP_plus']['out_dir'] + 'relax_pm_{}/'.format(relax_pm)
    sharp_plus_result = sharp.sharp_plus_recon(diffraction_data, projection_coords, init_obj, obj_ref=obj_ref,
                                               probe_ref=probe_ref, num_iter=num_iter, relax_pm=relax_pm,
                                               joint_recon=False, cstr_win=display_win, save_dir=sharp_plus_dir)
    # Plot reconstructed image
    plot_synthetic_img(sharp_plus_result['obj_revy'], img_title='SHARP+', ref_img=obj_ref,
                       display_win=display_win, display=display, save_dir=sharp_plus_dir)

    # PMACE recon
    alpha = config['PMACE']['alpha']                   
    rho = config['PMACE']['rho']                       # Mann averaging parameter
    probe_exp = config['PMACE']['probe_exponent']      # probe exponent
    pmace_dir = save_dir + config['PMACE']['out_dir'] + 'alpha_{}_rho_{}_probe_exp_{}/'.format(alpha, rho, probe_exp)
    pmace_result = pmace.pmace_recon(diffraction_data, projection_coords, init_obj, obj_ref=obj_ref, probe_ref=probe_ref,
                                     num_iter=num_iter, obj_pm=alpha, rho=rho, probe_exp=probe_exp, add_reg=False,
                                     joint_recon=False, cstr_win=display_win, save_dir=pmace_dir)
    # Plot reconstructed image
    plot_synthetic_img(pmace_result['obj_revy'], img_title='PMACE', ref_img=obj_ref,
                       display_win=display_win, display=display, save_dir=pmace_dir)

    # reg-PMACE recon
    alpha = config['reg-PMACE']['alpha']
    rho = config['reg-PMACE']['rho']
    probe_exp = config['reg-PMACE']['probe_exponent']
    bm3d_psd = config['reg-PMACE']['bm3d_psd']      
    reg_pmace_dir = save_dir + config['reg-PMACE']['out_dir'] + 'reg_PMACE/sigma_{}/'.format(bm3d_psd)
    reg_pmace_result = pmace.pmace_recon(diffraction_data, projection_coords, init_obj, obj_ref=obj_ref,
                                         probe_ref=probe_ref, num_iter=num_iter, obj_pm=alpha, rho=rho, 
                                         probe_exp=probe_exp, add_reg=True, sigma=bm3d_psd, 
                                         joint_recon=False, cstr_win=display_win, save_dir=reg_pmace_dir)
    # Plot reconstructed image
    plot_synthetic_img(reg_pmace_result['obj_revy'], img_title='reg-PMACE', ref_img=obj_ref,
                       display_win=display_win, display=display, save_dir=reg_pmace_dir)

    # Save config file to output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile(args.config_dir, save_dir + 'config.yaml')

    # Convergence plots
    xlabel, ylabel = 'Number of iteration', 'NRMSE value in log scale'
    line_label = 'nrmse'
    nrmse = {'ePIE': epie_result['obj_err'], 'WF': wf_result['obj_err'], 'AWF': awf_result['obj_err'],
             'SHARP': sharp_result['obj_err'], 'SHARP+': sharp_plus_result['obj_err'],
             'PMACE': pmace_result['obj_err'], 'reg-PMACE': reg_pmace_result['obj_err']}
    plot_nrmse(nrmse, title='Convergence plots of PMACE', label=[xlabel, ylabel, line_label],
               step_sz=10, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_plot')


if __name__ == '__main__':
    main()
