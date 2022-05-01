import sys
from pathlib import Path
root_dir = Path(__file__).parent.absolute().parent.absolute()
sys.path.append(str(root_dir))
import os, argparse, yaml
import datetime as dt
from shutil import copyfile

from ptycho_pmace.utils.utils import *
from ptycho_pmace.ptycho import *


'''
This file demonstrates the joint reconstruction of complex object and probe function 
from ptychographic data using WF, SHARP, ePIE and PMACE.
'''


def build_parser():
    parser = argparse.ArgumentParser(description='Ptychographic joint reconstruction using various approaches.')

    parser.add_argument('config_dir', type=str, help='Path to config file.', nargs='?', const='configs/joint_recon.yaml',
                        default=os.path.join(root_dir, 'configs/joint_recon.yaml'))
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
    init_obj_form = config['data']['init_obj_form']
    init_probe_form = config['data']['init_probe_form']
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
    diffract_data = load_measurement(data_dir + 'frame_data/', display=display)

    # Load scan points
    scan_loc_data = pd.read_csv(data_dir + 'Translations.tsv.txt', sep=None, engine='python', header=0)
    scan_loc = scan_loc_data[['FCx', 'FCy']].to_numpy()

    # calculate the coordinates of projections
    projection_coords = get_proj_coords_from_data(scan_loc, diffract_data)

    # # Generate formulated initial guess for reconstruction
    init_obj = np.ones(obj_ref.shape, dtype=np.complex128)
    init_probe = gen_init_probe(init_obj, probe_ref, projection_coords, diffract_data, formation=init_probe_form, display=display)
    init_obj = gen_init_obj(init_obj, init_probe, projection_coords, diffract_data, formation=init_obj_form, display=display)

    # display ground truth images and initial guesses for reconstruction.
    plot_cmplx_obj(obj_ref, obj_ref, img_title='gt obj', display_win=None, display=display)
    plot_cmplx_probe(probe_ref, probe_ref, img_title='gt probe', display=display)
    plot_cmplx_obj(init_obj, init_obj, img_title='init obj', display_win=None, display=display)
    plot_cmplx_probe(init_probe, init_probe, img_title='init probe', display=display)

    # Produce the cover/window for comparison
    if window_coords is not None:
        xmin, xmax, ymin, ymax = window_coords[0], window_coords[1], window_coords[2], window_coords[3]
        display_win = np.zeros(init_obj.shape)
        display_win[xmin:xmax, ymin:ymax] = 1
    else:
        display_win = None

    # Reconstruction parameters
    num_iter = config['recon']['num_iter']

    # ePIE recon
    obj_step_sz = config['ePIE']['obj_step_sz']
    probe_step_sz = config['ePIE']['probe_step_sz']
    epie_dir = save_dir + config['ePIE']['out_dir']
    epie_result = pie.epie_joint_recon(init_obj, init_probe, diffract_data, projection_coords, obj_ref, probe_ref,
                                       num_iter=num_iter, obj_step_sz=obj_step_sz, probe_step_sz=probe_step_sz,
                                       cstr_win=display_win, save_dir=epie_dir)

    # WF recon
    wf_dir = save_dir + config['WF']['out_dir']
    wf_result = wf.wf_joint_recon(init_obj, init_probe, diffract_data, projection_coords, obj_ref, probe_ref,
                                  num_iter=num_iter, accel=False, cstr_win=display_win, save_dir=wf_dir)

    # AWF
    awf_dir = save_dir + config['AWF']['out_dir']
    awf_result = wf.wf_joint_recon(init_obj, init_probe, diffract_data, projection_coords, obj_ref, probe_ref,
                                   num_iter=num_iter, accel=True, cstr_win=display_win, save_dir=awf_dir)

    # SHARP
    relax_pm = config['SHARP']['relax_pm']
    sharp_dir = save_dir + config['SHARP']['out_dir']
    sharp_result = sharp.sharp_joint_recon(init_obj, init_probe, diffract_data, projection_coords, obj_ref, probe_ref,
                                           num_iter=num_iter, relax_pm=relax_pm, cstr_win=display_win, save_dir=sharp_dir)

    # SHARP_plus
    relax_pm = config['SHARP_plus']['relax_pm']
    srp_plus_dir = save_dir + config['SHARP_plus']['out_dir']
    sharp_plus_result = sharp.sharp_plus_joint_recon(init_obj, init_probe, diffract_data, projection_coords, obj_ref, probe_ref,
                                                     num_iter=num_iter, relax_pm=relax_pm, cstr_win=display_win, save_dir=srp_plus_dir)

    # PMACE
    obj_param = config['PMACE']['obj_noisetosignal_ratio']
    probe_param = config['PMACE']['probe_noisetosignal_ratio']
    rho = config['PMACE']['rho']
    probe_exp = config['PMACE']['probe_exponent']
    obj_exp = config['PMACE']['obj_exponent']
    pmace_dir = save_dir + config['PMACE']['out_dir']
    pmace_result = pmace.pmace_joint_recon(init_obj, init_probe, diffract_data, projection_coords, obj_ref, probe_ref,
                                           num_iter=num_iter, obj_param=obj_param, probe_param=probe_param, rho=rho,
                                           probe_exp=probe_exp, obj_exp=obj_exp, cstr_win=display_win, save_dir=pmace_dir)
    #
    # PMACE + serial regularization
    obj_param = config['reg-PMACE']['obj_noisetosignal_ratio']
    probe_param = config['reg-PMACE']['probe_noisetosignal_ratio']
    rho = config['reg-PMACE']['rho']
    probe_exp = config['reg-PMACE']['probe_exp']
    obj_exp = config['reg-PMACE']['obj_exp']
    reg_wgt = config['reg-PMACE']['reg_wgt']
    noise_std = config['reg-PMACE']['noise_std']
    prior_model = config['reg-PMACE']['prior_model']
    reg_dir = save_dir + config['reg-PMACE']['out_dir']
    reg_pmace_result = pmace.reg_pmace_joint_recon(init_obj, init_probe, diffract_data, projection_coords, obj_ref, probe_ref,
                                                   num_iter=num_iter, obj_param=obj_param, probe_param=probe_param, rho=rho,
                                                   probe_exp=probe_exp, obj_exp=obj_exp, reg_wgt=reg_wgt, noise_std=noise_std,
                                                   prior_model=prior_model, cstr_win=display_win, save_dir=reg_dir)
    # Save config file to output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile(args.config_dir, save_dir + 'config.yaml')

    # Plot reconstructed images and compare with ground truth complex image
    plot_cmplx_img(pmace_result['obj_revy'], img_title='PMACE', ref_img=obj_ref,
                   display_win=display_win, display=display, save_fname=reg_dir+'reconstructed_cmplx_img',
                   fig_sz=[8, 3], mag_vmax=1.05, mag_vmin=.85, real_vmax=1.1, real_vmin=.8, imag_vmax=0, imag_vmin=-0.6)
    plot_cmplx_img(reg_pmace_result['probe_revy'], img_title='reg-PMACE', ref_img=probe_ref,
                   display_win=display_win, display=display, save_fname=reg_dir+'reconstructed_cmplx_probe',
                   fig_sz=[8, 3], mag_vmax=100, mag_vmin=0, real_vmax=30, real_vmin=-70, imag_vmax=30, imag_vmin=-70)

    # Convergence plots
    xlabel, ylabel = 'Number of iteration', 'NRMSE value in log scale'
    line_label = 'obj_nrmse'
    obj_nrmse = {'ePIE': epie_result['obj_err'], 'WF': wf_result['obj_err'], 'AWF': awf_result['obj_err'],
                 'SHARP': sharp_result['obj_err'], 'SHARP+': sharp_plus_result['obj_err'],
                 'PMACE': pmace_result['obj_err'], 'reg-PMACE': reg_pmace_result['obj_err']}
    plot_nrmse(obj_nrmse, title='Convergence plots', label=[xlabel, ylabel, line_label],
               step_sz=10, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_plot')
    
    line_label = 'probe_nrmse'
    probe_nrmse = {'ePIE': epie_result['probe_err'], 'WF': wf_result['probe_err'], 'AWF': awf_result['probe_err'],
                   'SHARP': sharp_result['probe_err'], 'SHARP+': sharp_plus_result['probe_err'],
                   'PMACE': pmace_result['probe_err'], 'reg-PMACE': reg_pmace_result['probe_err']}
    plot_nrmse(probe_nrmse, title='Convergence plots', label=[xlabel, ylabel, line_label],
               step_sz=10, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_plot')


if __name__ == '__main__':
    main()
