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
This file demonstrates the reconstruction of complex transmittance image by processing the synthetic data with known probe function. 
'''


def build_parser():
    parser = argparse.ArgumentParser(description='Ptychographic image reconstruction using various approaches.')
    parser.add_argument('config_dir', type=str, help='Path to config file.', nargs='?', const='demo_pmace.yaml',
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
    ref_obj = load_img(obj_dir)
    ref_probe = load_img(probe_dir)

    # Load intensity only measurements(data) from file and pre-process the data
    y_meas = load_measurement(data_dir + 'frame_data/', display=display)

    # Load scan points
    scan_file = pd.read_csv(data_dir + 'Translations.tsv.txt', sep=None, engine='python', header=0)
    scan_loc = scan_file[['FCx', 'FCy']].to_numpy()

    # calculate the coordinates of projections
    patch_bounds = get_proj_coords_from_data(scan_loc, y_meas)

    # Generate formulated initial guess for reconstruction
    init_obj = gen_init_obj(y_meas, patch_bounds, obj_ref=ref_obj, probe_ref=ref_probe, display=display)

    # Produce the cover/window for comparison
    if window_coords is not None:
        xmin, xmax, ymin, ymax = window_coords[0], window_coords[1], window_coords[2], window_coords[3]
        recon_win = np.zeros(init_obj.shape)
        recon_win[xmin:xmax, ymin:ymax] = 1
    else:
        recon_win = None

    # Reconstruction parameters and arguments
    recon_args = dict(init_obj=init_obj, ref_obj=ref_obj, ref_probe=ref_probe, recon_win=recon_win,num_iter=config['recon']['num_iter'], joint_recon=False)
    fig_args = dict(ref_img=ref_obj, display_win=recon_win, display=display)

    # ePIE recon
    step_sz = config['ePIE']['step_sz']
    epie_dir = save_dir + config['ePIE']['out_dir']
    epie_result = pie.epie_recon(y_meas, patch_bounds, obj_step_sz=step_sz, save_dir=epie_dir, **recon_args)
    # Plot reconstructed image
    plot_synthetic_img(epie_result['object'], img_title='ePIE', save_dir=epie_dir, **fig_args)

    # Wirtinger Flow (WF) recon
    wf_dir = save_dir + config['WF']['out_dir']
    wf_result = wf.wf_recon(y_meas, patch_bounds, accel=False, save_dir=wf_dir, **recon_args)
    plot_synthetic_img(wf_result['object'], img_title='WF', save_dir=wf_dir, **fig_args)

    # Acclerated Wirtinger Flow (AWF) recon
    awf_dir = save_dir + config['AWF']['out_dir']
    awf_result = wf.wf_recon(y_meas, patch_bounds, accel=True, save_dir=awf_dir, **recon_args)
    plot_synthetic_img(awf_result['object'], img_title='AWF', save_dir=awf_dir, **fig_args)

    # SHARP recon
    relax_pm = config['SHARP']['relax_pm']
    sharp_dir = save_dir + config['SHARP']['out_dir']
    sharp_result = sharp.sharp_recon(y_meas, patch_bounds, relax_pm=relax_pm, save_dir=sharp_dir, **recon_args)
    plot_synthetic_img(sharp_result['object'], img_title='SHARP', save_dir=sharp_dir, **fig_args)

    # SHARP+ recon
    relax_pm = config['SHARP_plus']['relax_pm']
    sharp_plus_dir = save_dir + config['SHARP_plus']['out_dir']
    sharp_plus_result = sharp.sharp_plus_recon(y_meas, patch_bounds, relax_pm=relax_pm, save_dir=sharp_plus_dir, **recon_args)
    plot_synthetic_img(sharp_plus_result['object'], img_title='SHARP+', save_dir=sharp_plus_dir, **fig_args)

    # PMACE recon
    alpha = config['PMACE']['alpha']                   
    pmace_dir = save_dir + config['PMACE']['out_dir']
    pmace_result = pmace.pmace_recon(y_meas, patch_bounds, obj_data_fit_prm=alpha, add_reg=False, save_dir=pmace_dir, **recon_args)
    plot_synthetic_img(pmace_result['object'], img_title='PMACE', save_dir=pmace_dir, **fig_args)

    # reg-PMACE recon
    alpha = config['reg-PMACE']['alpha']
    bm3d_psd = config['reg-PMACE']['bm3d_psd']      
    reg_pmace_dir = save_dir + config['reg-PMACE']['out_dir']
    reg_pmace_result = pmace.pmace_recon(y_meas, patch_bounds, obj_data_fit_prm=alpha, add_reg=True, sigma=bm3d_psd, save_dir=reg_pmace_dir, **recon_args)
    plot_synthetic_img(reg_pmace_result['object'], img_title='reg-PMACE', save_dir=reg_pmace_dir, **fig_args)

    # Save config file to output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile(args.config_dir, save_dir + 'config.yaml')

    # Convergence plots
    xlabel, ylabel = 'Number of iteration', 'NRMSE value in log scale'
    line_label = 'nrmse'
    nrmse = {'ePIE': epie_result['err_obj'], 'WF': wf_result['err_obj'], 'AWF': awf_result['err_obj'],
             'SHARP': sharp_result['err_obj'], 'SHARP+': sharp_plus_result['err_obj'],
             'PMACE': pmace_result['err_obj'], 'reg-PMACE': reg_pmace_result['err_obj']}
    plot_nrmse(nrmse, title='Convergence plots of PMACE', label=[xlabel, ylabel, line_label],
               step_sz=10, fig_sz=[8, 4.8], display=display, save_fname=save_dir + 'convergence_plot')

def plot_synthetic_img(cmplx_img, img_title, ref_img, display_win=None, display=False, save_dir=None):
    """ Function to plot reconstruction results in this experiment. """
    save_fname = None if (save_dir is None) else save_dir + 'recon_cmplx_img'
    plot_cmplx_img(cmplx_img, img_title=img_title, ref_img=ref_img, 
                   display_win=display_win, display=display, save_fname=save_fname,
                   fig_sz=[8, 3], mag_vmax=1, mag_vmin=0.5, phase_vmax=0, phase_vmin=-np.pi/4,
                   real_vmax=1.1, real_vmin=0.8, imag_vmax=0, imag_vmin=-0.6)


if __name__ == '__main__':
    main()
