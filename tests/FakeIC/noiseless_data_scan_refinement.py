import argparse
import yaml
import os
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from pmace.utils import *
from pmace.pmace import *
from exp_funcs import *
from shutil import copyfile


'''
This script demonstrates scan refinement along with PMACE reconstruction. Demo functionality includes:
 * Loading reference object transmittance image and reference probe profile function;
 * Loading scan locations, simulated measurements, and reconstruction parameters;
 * Computing a reconstruction from the loaded data using PMACE;
 * Displaying and saving the results.
'''
print('This script demonstrates reconstruction of complex transmittance image using PMACE. Demo functionality includes:'
      '\n\t * Loading reference object transmittance image and reference probe profile function;'
      '\n\t * Loading scan locations, simulated measurements, and reconstruction parameters;'
      '\n\t * Computing a reconstruction from the loaded data using PMACE;'
      '\n\t * Displaying and saving the results.\n')


def build_parser():
    # Build an argument parser for command line arguments.
    parser = argparse.ArgumentParser(description='FakeIC image reconstruction using PMACE.')
    parser.add_argument('config_dir', type=str, help='Path to config file.', nargs='?', 
                        const='noiseless_data_scan_refinement.yaml',
                        default='config/noiseless_data_scan_refinement.yaml')
    return parser


def main():
    # Load config file and pass arguments
    parser = build_parser()
    args = parser.parse_args()
    print("Passing arguments ...")
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)
        
    # Define paths to reference object, probe, and intensity measurements
    data_dir = config['data']['data_dir']
    obj_dir = data_dir + 'ref_object.tiff'
    probe_dir = data_dir + 'ref_probe.tiff'
    init_scan_loc_file_pth = data_dir + config['data']['init_scan_loc_file_pth']
    print(init_scan_loc_file_pth)

    # Define window coordinates for reconstruction region
    window_coords = None  
    out_dir = config['recon']['out_dir']

    # Determine the current timestamp
    today_date = dt.date.today()
    date_time = dt.datetime.strftime(today_date, '%Y-%m-%d_%H_%M/')

    # Create a path with the current timestamp
    save_dir = os.path.join(out_dir, date_time)

    # Check if the output directory exists, if not, create it
    print("Creating output directory '%s' ..." % save_dir)
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as error:
        print("Output directory cannot be created")

    # Load reference images from files
    print("Loading data ...")
    ref_obj = load_img(obj_dir)
    ref_probe = load_img(probe_dir)
    plot_FakeIC_img(ref_obj, img_title='GT', display_win=None, save_dir=save_dir)
    
    # Load measurements (diffraction patterns) from file and pre-process data
    y_meas = load_measurement(data_dir + 'frame_data/')

    # Load ground truth scan positions
    scan_loc_file = pd.read_csv(data_dir + 'Translations.tsv.txt', sep=None, engine='python', header=0)
    ground_truth_scan_loc = scan_loc_file[['FCx', 'FCy']].to_numpy()

    # Check if the init_scan_loc file exists
    if os.path.exists(init_scan_loc_file_pth):
        # Load randomized init scan locations
        init_scan_loc_file = pd.read_csv(init_scan_loc_file_pth, sep=None, engine='python', header=0)
        init_scan_loc = init_scan_loc_file[['FCx', 'FCy']].to_numpy()
    else:
        # Mean and standard deviation for the Gaussian distribution
        mean = 0  # Mean of the distribution
        std_dev = 4  # Standard deviation of the distribution

        # Generate random offsets following a Gaussian distribution
        offsets = np.clip(np.random.normal(mean, std_dev, (len(ground_truth_scan_loc), 2)), -10, 10)
        init_scan_loc = np.asarray(ground_truth_scan_loc + offsets)
        print(np.amax(offsets), np.amin(offsets))

        # Save initial scan locations (randomized) to output directory 
        df = pd.DataFrame({'FCx': init_scan_loc[:, 0], 'FCy': init_scan_loc[:, 1]})
        df.to_csv(init_scan_loc_file_pth)
    
    # Plot difference between init scan locations and ground truth scan locations
    # compare_scan_loc(init_scan_loc, ground_truth_scan_loc, save_dir=save_dir)
    compare_scan_loc(np.round(init_scan_loc), np.round(ground_truth_scan_loc), save_dir=save_dir)
    
    # Calculate coordinates of projections from scan positions
    ground_truth_patch_crds = get_proj_coords_from_data(ground_truth_scan_loc, y_meas)
    init_patch_crds = get_proj_coords_from_data(init_scan_loc, y_meas)
    
    # # calculate coordinates of projections from scan positions ===================================================
    scan_loc = ground_truth_scan_loc
    patch_crds = ground_truth_patch_crds
    # scan_loc = init_scan_loc
    # patch_crds = init_patch_crds
    
    # Formulate an initial guess of complex object for reconstruction
    init_obj = np.ones(ref_obj.shape, dtype=np.complex64)
    init_probe = gen_init_probe(y_meas, patch_crds, init_obj, fres_propagation=False)
    init_obj = gen_init_obj(y_meas, patch_crds, ref_obj.shape, ref_probe=init_probe)

    # Pre-define the reconstruction region
    if window_coords is not None:
        xmin, xmax, ymin, ymax = window_coords[0], window_coords[1], window_coords[2], window_coords[3]
    else:
        xmin, xmax, ymin, ymax = np.min(scan_loc[:, 0]), np.max(scan_loc[:, 0]), np.min(scan_loc[:, 1]), np.max(scan_loc[:, 1])
    recon_win = np.zeros(init_obj.shape)
    recon_win[int(xmin):int(xmax), int(ymin):int(ymax)] = 1

    # Save initialized images to the output file
    save_tiff(init_obj, save_dir + 'init_object.tiff')
    save_tiff(init_probe, save_dir + 'init_probe.tiff')  
    # plot_FakeIC_img(init_obj, img_title='init', display_win=None, save_dir=save_dir) 
    # plot_probe_img(init_probe, img_title='init', save_dir=save_dir)
    # compare_result_with_ground_truth_img(init_obj, ref_obj, display_win=recon_win, save_dir=save_dir)
    # compare_result_with_ground_truth_probe(init_probe, ref_probe, save_dir=save_dir)
    
    # Reconstruction parameters
    num_iter = config['recon']['num_iter']
    joint_recon = config['recon']['joint_recon']
    object_prm = config['recon']['object_data_fit_param']
    probe_prm = config['recon']['probe_data_fit_param']
    probe_exp = config['recon']['probe_exp']
    fig_args = dict(display_win=recon_win, save_dir=save_dir)
    
    for object_prm in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for probe_prm in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for probe_exp in [1, 1.5]:
                pmace_dir = save_dir + 'obj_data_fit_prm_{}_obj_data_fit_prm_{}_probe_exp_{}_gt_scan/'.format(object_prm, probe_prm, probe_exp)
                pmace_obj = PMACE(pmace_recon, y_meas, patch_crds, init_obj, init_probe=init_probe, ref_obj=ref_obj, ref_probe=ref_probe, 
                                  num_iter=num_iter, joint_recon=joint_recon, probe_exp=probe_exp,
                                  obj_data_fit_prm=object_prm, probe_data_fit_prm=probe_prm, recon_win=recon_win, save_dir=pmace_dir,
                                  scan_loc_refinement_iterations=[], gt_scan_loc=np.round(ground_truth_scan_loc))
                pmace_result = pmace_obj()
                plot_FakeIC_img(pmace_result['object'], img_title='PMACE', display_win=recon_win, save_dir=pmace_dir)  
                compare_result_with_ground_truth_img(pmace_result['object'], ref_obj, display_win=recon_win, save_dir=pmace_dir)
                plot_convergence_curve(n_iter=num_iter, init_err=0.2, err_obj=pmace_result['err_obj'], err_meas=pmace_result['err_meas'], save_dir=pmace_dir)
                if joint_recon:
                    recon_probe = pmace_result['probe'][0]
                    plot_probe_img(recon_probe, img_title='PMACE_recon', save_dir=pmace_dir)  
                    compare_result_with_ground_truth_probe(recon_probe, ref_probe, save_dir=pmace_dir)
                    plot_convergence_curve(n_iter=num_iter, init_err=0.2, err_obj=pmace_result['err_obj'], err_probe=pmace_result['err_probe'], err_meas=pmace_result['err_meas'], save_dir=pmace_dir)
                    print(object_prm, probe_prm, probe_exp, pmace_result['err_obj'][-1], pmace_result['err_probe'][-1], pmace_result['err_meas'][-1])
                             
#     # PMACE reconstruction
#     pmace_dir = save_dir + 'obj_data_fit_prm_{}_probe_exp_{}_init_scan_refinement/'.format(object_prm, probe_exp)
#     pmace_obj = PMACE(pmace_recon, y_meas, patch_crds, init_obj, init_probe=init_probe, ref_obj=ref_obj, ref_probe=ref_probe, 
#                       num_iter=num_iter, joint_recon=joint_recon, probe_exp=probe_exp,
#                       obj_data_fit_prm=object_prm, probe_data_fit_prm=probe_prm, recon_win=recon_win, save_dir=pmace_dir,
#                       scan_loc_refinement_iterations=range(10, 101, 1), gt_scan_loc=np.round(ground_truth_scan_loc))
#     pmace_result = pmace_obj()
#     plot_FakeIC_img(pmace_result['object'], img_title='PMACE', display_win=recon_win, save_dir=pmace_dir)  
#     compare_result_with_ground_truth_img(pmace_result['object'], ref_obj, display_win=recon_win, save_dir=pmace_dir)
#     plot_convergence_curve(n_iter=num_iter, init_err=0.2, err_obj=pmace_result['err_obj'], err_meas=pmace_result['err_meas'], save_dir=pmace_dir)
#     if joint_recon:
#         recon_probe = pmace_result['probe'][0]
#         plot_probe_img(recon_probe, img_title='PMACE_recon', save_dir=pmace_dir)  
#         compare_result_with_ground_truth_probe(recon_probe, ref_probe, save_dir=pmace_dir)
#         plot_convergence_curve(n_iter=num_iter, init_err=0.2, err_obj=pmace_result['err_obj'], err_probe=pmace_result['err_probe'], err_meas=pmace_result['err_meas'], save_dir=pmace_dir)
#         # print(object_prm, probe_prm, pmace_result['err_obj'][-1], pmace_result['err_probe'][-1], pmace_result['err_meas'][-1])
            
    # Save config file to the output directory
    if not os.path.exists(pmace_dir):
        os.makedirs(pmace_dir)
    copyfile(args.config_dir, pmace_dir + 'config.yaml')

    
if __name__ == '__main__':
    main()