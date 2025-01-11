import argparse
import os
import yaml
import numpy as np
import pandas as pd
from pmace.utils import *
from pmace.pmace import *
import demo_utils
from pmace.display import *


'''
This script demonstrates reconstruction of complex transmittance image along with single/multiple probe modes using BM-PMACE. 
Demo functionality includes:
 * Loading reference object transmittance image and reference probe profile function;
 * Loading scan locations, simulated measurements, and reconstruction parameters;
 * Computing a reconstruction from the loaded data using PMACE;
 * Displaying and saving the results.
'''
print('This script demonstrates reconstruction of complex transmittance image along with single/multiple probe modes using BM-PMACE.\
\n\t Demo functionality includes: \
\n\t * Loading reference object transmittance image and reference probe profile function; \
\n\t * Loading scan locations, simulated measurements, and reconstruction parameters; \
\n\t * Computing a reconstruction from the loaded data using BM-PMACE; \
\n\t * Displaying and saving the results.\n')


def build_parser():
    parser = argparse.ArgumentParser(description='BM-PMACE demo for ptychographic image reconstruction.')
    parser.add_argument('config_dir', type=str, help='Path to config file.', nargs='?', 
                        const='demo_blind_multi_mode_pmace.yaml',
                        default='config/demo_blind_multi_mode_pmace.yaml')
    
    parser.add_argument('-r', '--random_seed', type=int, help='Random seed for reproducibility.', default=0)
    parser.add_argument('-d', '--display', action='store_true', help='Display the intermediate results.')
    parser.add_argument('-t', '--timestamp', action='store_true', help='Add timestamp to the output directory.')
    
    return parser


def plot_synthetic_img(cmplx_img, img_title, display=True, display_win=None, save_dir=None):
    """
    Display and save a demo result.

    Args:
        cmplx_img: complex image array.
        img_title: title of the plot image.
        display: display the image.
        display_win: window to plot the image.
        save_dir: directory for saving the image.
    
    Example:
    plot_synthetic_img(cmplx_img, img_title="demo_img", display_win=None, save_dir='/path/to/save/directory/')
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
    save_fname = None if (save_dir is None) else os.path.join(save_dir, f'{img_title}_recon_cmplx_img')
    # Plot complex image with customized parameters
    plot_cmplx_img(cmplx_img, 
                   img_title=img_title, 
                   display_win=display_win, 
                   display=display,
                   save_fname=save_fname,
                   mag_vmax=1, 
                   mag_vmin=0.9, 
                   phase_vmax=-0.2, 
                   phase_vmin=-1.0,
                   real_vmax=1.1, 
                   real_vmin=0.8, 
                   imag_vmax=0, 
                   imag_vmin=-1.0)


def main():
    # Load config file and parse arguments
    parser = build_parser()
    args = parser.parse_args()
    print("Passing arguments ...")
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)

    # Read data from configuration file
    data_dir = config['data']['data_dir']
    ref_object_dir = os.path.join(data_dir, config['data']['ref_object_dir'])
    ref_probe_mode_0_dir = os.path.join(data_dir, config['data']['ref_probe_0_dir'])
    ref_probe_mode_1_dir = os.path.join(data_dir, config['data']['ref_probe_1_dir'])
    window_coords = config['data']['window_coords']
    save_dir = config['output']['out_dir']

    # Check and create the output directory
    print("Creating output directory '%s' ..." % save_dir)
    if args.timestamp:
        save_dir = setup_output_directory_with_timestamp(config['output']['out_dir'])
    else:
        save_dir = config['output']['out_dir']
        os.makedirs(save_dir, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(args.random_seed)

    # Load reference images
    print("Loading data ...")
    ref_object = load_img(ref_object_dir)
    ref_probe_mode_0 = load_img(ref_probe_mode_0_dir)
    ref_probe_mode_1 = load_img(ref_probe_mode_1_dir)

    # Load measurements (diffraction patterns) and pre-process data
    y_meas = load_measurement(os.path.join(data_dir, 'frame_data/'))

    # Load scan positions
    scan_loc_file = pd.read_csv(os.path.join(data_dir, 'Translations.tsv.txt'), sep=None, engine='python', header=0)
    scan_loc = scan_loc_file[['FCx', 'FCy']].to_numpy()

    # Calculate coordinates of projections from scan positions
    patch_crds = get_proj_coords_from_data(scan_loc, y_meas)

    # Generate initial guesses for reconstruction
    source_wl = float(config['initialization']['source_wavelength'])
    propagation_dist = float(config['initialization']['propagation_distance'])
    sampling_interval = float(config['initialization']['sampling_interval'])
    
    # Pre-define reconstruction region
    if window_coords is not None:
        xmin, xmax, ymin, ymax = window_coords
    else:
        xmin, xmax, ymin, ymax = np.amin(scan_loc[:, 0]), np.amax(scan_loc[:, 0]), np.amin(scan_loc[:, 1]), np.amax(scan_loc[:, 1])
    recon_win = np.zeros(ref_object.shape)
    recon_win[xmin:xmax, ymin:ymax] = 1
    
    # Initialize images
    init_obj = np.ones_like(ref_object, dtype=np.complex64)
    init_probe = gen_init_probe(y_meas, patch_crds, init_obj, fres_propagation=True, sampling_interval=sampling_interval,
                                source_wl=source_wl, propagation_dist=propagation_dist) 
    init_obj = gen_init_obj(y_meas, patch_crds, init_obj.shape, ref_probe=init_probe)
    
    # Save initial guesses
    save_tiff(init_obj, os.path.join(save_dir, 'init_obj.tiff'))
    save_tiff(init_probe, os.path.join(save_dir, 'init_probe.tiff')) 

    # Reconstruction parameters -- single mode
    single_mode_recon_args = dict(init_obj=init_obj, init_probe=init_probe, recon_win=recon_win,
                                  ref_obj=ref_object, ref_probe=[ref_probe_mode_0, ref_probe_mode_1],
                                  num_iter=config['recon']['num_iter'], joint_recon=config['recon']['joint_recon'],
                                  obj_data_fit_prm=config['single-mode']['alpha1'], 
                                  probe_data_fit_prm=config['single-mode']['alpha2'], 
                                  probe_exp=config['single-mode']['kappa'],
                                  add_mode=[])
    
    # Reconstruction parameters -- multi mode
    multi_mode_recon_args = dict(init_obj=init_obj, init_probe=init_probe, recon_win=recon_win,
                                  ref_obj=ref_object, ref_probe=[ref_probe_mode_0, ref_probe_mode_1],
                                  num_iter=config['recon']['num_iter'], joint_recon=config['recon']['joint_recon'],
                                  obj_data_fit_prm=config['multi-mode']['alpha1'], 
                                  probe_data_fit_prm=config['multi-mode']['alpha2'], 
                                  probe_exp=config['multi-mode']['kappa'],
                                  add_mode=config['recon']['add_mode'], 
                                  energy_ratio=float(config['recon']['energy_ratio']),
                                  wavelength=float(config['recon']['source_wavelength']),
                                  propagation_dist=float(config['recon']['propagation_distance']),
                                  img_px_sz=float(config['recon']['sampling_interval']))
    
    fig_args = dict(display_win=recon_win, save_dir=save_dir, display=args.display)
    
    # PMACE reconstruction -- single mode
    single_mode_recon_dir = os.path.join(save_dir, 'single_mode_reconstruction_result/')
    pmace_single_mode_obj = PMACE(pmace_recon, y_meas, patch_crds, save_dir=single_mode_recon_dir, **single_mode_recon_args)
    pmace_single_mode_res = pmace_single_mode_obj()
    plot_synthetic_img(pmace_single_mode_res['object'], img_title='single_mode', **fig_args)

    # PMACE reconstruction -- two modes
    multi_mode_recon_dir = os.path.join(save_dir, 'multi_mode_reconstruction_result/')
    pmace_multi_mode_obj = PMACE(pmace_recon, y_meas, patch_crds, save_dir=multi_mode_recon_dir, **multi_mode_recon_args)
    pmace_multi_mode_res = pmace_multi_mode_obj()
    plot_synthetic_img(pmace_multi_mode_res['object'], img_title='multi_mode', **fig_args)


if __name__ == '__main__':
    main()