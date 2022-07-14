import sys
from pathlib import Path
root_dir = Path(__file__).parent.absolute().parent.absolute()
sys.path.append(str(root_dir))
root_root_dir = Path(__file__).parent.absolute().parent.absolute().parent.absolute()
sys.path.append(str(root_root_dir))
import os, argparse, yaml
import datetime as dt
from shutil import copyfile
from utils.utils import *
from ptycho import *


'''
This file demonstrates the joint reconstruction of complex object and probe function 
from ptychographic data using WF, SHARP, ePIE and PMACE.
'''


def build_parser():
    parser = argparse.ArgumentParser(description='Ptychographic joint reconstruction using various approaches.')
    parser.add_argument('config_dir', type=str, help='Path to config file.', nargs='?', const='joint_recon.yaml',
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
    ref_obj = load_img(obj_dir)
    ref_probe = load_img(probe_dir)

    # display ground truth images
    plot_cmplx_img(ref_obj, img_title='GT obj', ref_img=ref_obj, display=display,
                   mag_vmax=1, mag_vmin=.5, phase_vmax=0, phase_vmin=-np.pi/4,
                   real_vmax=1.1, real_vmin=.8, imag_vmax=0, imag_vmin=-0.6)
    plot_cmplx_img(ref_probe, img_title='GT probe', ref_img=ref_probe, display=display,
                   mag_vmax=100, mag_vmin=0, real_vmax=30, real_vmin=-70, imag_vmax=30, imag_vmin=-70)

    # Load intensity only measurements(data) from file and pre-process the data
    y_meas = load_measurement(data_dir + 'frame_data/', display=display)

    # Load scan points
    scan_file = pd.read_csv(data_dir + 'Translations.tsv.txt', sep=None, engine='python', header=0)
    scan_loc = scan_file[['FCx', 'FCy']].to_numpy()

    # calculate the coordinates of projections
    patch_bounds = get_proj_coords_from_data(scan_loc, y_meas)

    # # Generate formulated initial guess for reconstruction
    init_obj = np.ones(ref_obj.shape, dtype=np.complex64)
    init_probe = gen_init_probe(y_meas, patch_bounds, obj_ref=init_obj, display=display)
    init_obj = gen_init_obj(y_meas, patch_bounds, obj_ref=init_obj, probe_ref=init_probe, display=display)

    # Produce the cover/window for comparison
    if window_coords is not None:
        xmin, xmax, ymin, ymax = window_coords[0], window_coords[1], window_coords[2], window_coords[3]
        recon_win = np.zeros(init_obj.shape)
        recon_win[xmin:xmax, ymin:ymax] = 1
    else:
        recon_win = None

    # Reconstruction parameters
    num_iter = config['recon']['num_iter']# Reconstruction parameters and arguments
    args = dict(init_obj=init_obj, init_probe=init_probe, ref_obj=ref_obj, ref_probe=ref_probe, 
                num_iter=config['recon']['num_iter'], joint_recon=True, recon_win=recon_win)

    # ePIE recon
    obj_ss = config['ePIE']['obj_step_sz']
    probe_ss = config['ePIE']['probe_step_sz']
    epie_dir = save_dir + config['ePIE']['out_dir']
    epie_result = pie.epie_recon(y_meas, patch_bounds, obj_step_sz=obj_ss, probe_step_sz=probe_ss, save_dir=epie_dir, **args)

    # WF recon
    wf_dir = save_dir + config['WF']['out_dir']
    wf_result = wf.wf_recon(y_meas, patch_bounds, accel=False, save_dir=wf_dir, **args)

    # AWF recon
    awf_dir = save_dir + config['AWF']['out_dir']
    awf_result = wf.wf_recon(y_meas, patch_bounds, accel=True, save_dir=awf_dir, **args)

    # SHARP recon
    relax_pm = config['SHARP']['relax_prm']
    sharp_dir = save_dir + config['SHARP']['out_dir']
    sharp_result = sharp.sharp_recon(y_meas, patch_bounds, relax_pm=relax_pm, save_dir=sharp_dir, **args)


    # SHARP_plus
    relax_prm = config['SHARP_plus']['relax_prm']
    srp_plus_dir = save_dir + config['SHARP_plus']['out_dir']
    sharp_plus_result = sharp.sharp_plus_recon(y_meas, patch_bounds, relax_pm=relax_prm,
                                               save_dir=srp_plus_dir, **args)

    # PMACE
    obj_prm = config['PMACE']['obj_prm']
    probe_prm = config['PMACE']['probe_prm']
    probe_exp = config['PMACE']['probe_exp']
    obj_exp = config['PMACE']['obj_exp']
    pmace_dir = save_dir + config['PMACE']['out_dir']
    pmace_result = pmace.pmace_recon(y_meas, patch_bounds, obj_data_fit_prm=obj_prm, probe_data_fit_prm=probe_prm, 
                                     probe_exp=probe_exp, obj_exp=obj_exp, add_reg=False, save_dir=pmace_dir, **args)

    
    # PMACE + serial regularization
    obj_prm = config['reg-PMACE']['obj_prm']
    probe_prm = config['reg-PMACE']['probe_prm']
    probe_exp = config['reg-PMACE']['probe_exp']
    obj_exp = config['reg-PMACE']['obj_exp']
    bm3d_psd = config['reg-PMACE']['bm3d_psd']
    reg_pmace_dir = save_dir + config['reg-PMACE']['out_dir']
    reg_pmace_result = pmace.pmace_recon(y_meas, patch_bounds, obj_data_fit_prm=obj_prm, probe_data_fit_prm=probe_prm, 
                                         probe_exp=probe_exp, obj_exp=obj_exp, add_reg=True, sigma=bm3d_psd, 
                                         save_dir=reg_pmace_dir, **args)

    # Save config file to output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile(args.config_dir, save_dir + 'config.yaml')

    # Plot reconstructed images and compare with ground truth complex image
    plot_cmplx_img(reg_pmace_result['object'], img_title='reg-PMACE', ref_img=obj_ref,
                   display_win=display_win, display=display, save_fname=reg_pmace_dir + 'reconstructed_cmplx_img',
                   mag_vmax=1, mag_vmin=.5, phase_vmax=0, phase_vmin=-np.pi/4,
                   real_vmax=1.1, real_vmin=.8, imag_vmax=0, imag_vmin=-0.6)
    plot_cmplx_img(reg_pmace_result['probe'], img_title='reg-PMACE', ref_img=probe_ref, display=display,
                   save_fname=reg_pmace_dir + 'reconstructed_cmplx_probe',
                   mag_vmax=100, mag_vmin=0, real_vmax=30, real_vmin=-70, imag_vmax=30, imag_vmin=-70)

    # Convergence plots
    xlabel, ylabel = 'Number of iteration', 'NRMSE value in log scale'
    line_label = 'obj_nrmse'
    obj_nrmse = {'ePIE': epie_result['err_obj'], 'WF': wf_result['err_obj'], 'AWF': awf_result['err_obj'],
                 'SHARP': sharp_result['err_obj'], 'SHARP+': sharp_plus_result['err_obj'],
                 'PMACE': pmace_result['err_obj'], 'reg-PMACE': reg_pmace_result['err_obj']}
    plot_nrmse(obj_nrmse, title='Convergence plots', label=[xlabel, ylabel, line_label],
               step_sz=10, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_plot')
    
    line_label = 'probe_nrmse'
    probe_nrmse = {'ePIE': epie_result['err_probe'], 'WF': wf_result['err_probe'], 'AWF': awf_result['err_probe'],
                   'SHARP': sharp_result['err_probe'], 'SHARP+': sharp_plus_result['err_probe'],
                   'PMACE': pmace_result['err_probe'], 'reg-PMACE': reg_pmace_result['err_probe']}
    plot_nrmse(probe_nrmse, title='Convergence plots', label=[xlabel, ylabel, line_label],
               step_sz=10, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_plot')


if __name__ == '__main__':
    main()
