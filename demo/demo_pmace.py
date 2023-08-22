import demo_utils
import argparse, yaml
import time
from pmace.utils import *
from pmace.pmace import *


'''
This script demonstrates reconstruction of complex transmittance image using PMACE. Demo functionality includes:
 * Downloading demo dataset from specified urls;
 * Loading reference object transmittance image and reference probe profile function;
 * Loading scan locations, simulated measurements, and reconstruction parameters;
 * Computing a reconstruction from the loaded data using PMACE;
 * Displaying and saving the results.
'''
print('This script demonstrates reconstruction of complex transmittance image using PMACE. Demo functionality includes:\
\n\t * Downloading demo dataset from specified urls; \
\n\t * Loading reference object transmittance image and reference probe profile function; \
\n\t * Loading scan locations, simulated measurements, and reconstruction parameters; \
\n\t * Computing a reconstruction from the loaded data using PMACE; \
\n\t * Displaying and saving the results.\n')


def build_parser():
    parser = argparse.ArgumentParser(description='Ptychographic image reconstruction using PMACE.')
    parser.add_argument('config_dir', type=str, help='Path to config file.', nargs='?', const='demo_pmace.yaml',
                        default='config/demo_pmace.yaml')
    return parser


def main():
    # load config file and pass arguments
    parser = build_parser()
    args = parser.parse_args()
    print("Passing arguments ...")
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)

    # url to dataset.
    dataset_url = config['dataset']['download_url']
    # destination path to download and extract the phantom and NN weight files.
    dataset_dir = config['dataset']['save_dir']  
    # path to reference object, probe and intensity measurements
    obj_dir = dataset_dir + config['data']['obj_dir']
    probe_dir = dataset_dir + config['data']['probe_dir']
    data_dir = dataset_dir + config['data']['data_dir']
    window_coords = config['data']['window_coords']
    save_dir = config['recon']['out_dir']
    
    # download dataset. 
    dataset_path = download_and_extract(dataset_url, dataset_dir)

    # check directory
    print("Creating output directory '%s' ..." % save_dir)
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as error:
        print("Output directory can not be created")

    # load reference images from file
    print("Loading data ...")
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
    else:
        xmin, xmax, ymin, ymax = np.amin(scan_loc[0]), np.amax(scan_loc[0]), np.amin(scan_loc[1]), np.amax(scan_loc[1])
    recon_win = np.zeros(init_obj.shape)
    recon_win[xmin:xmax, ymin:ymax] = 1

    # reconstruction parameters
    num_iter = config['recon']['num_iter']
    joint_recon = config['recon']['joint_recon']
    alpha = config['recon']['data_fit_param']
    fig_args = dict(display_win=recon_win, save_dir=save_dir)
                    
    # PMACE recon
    pmace_obj = PMACE(pmace_recon, y_meas, patch_crds, init_obj, ref_obj=ref_obj, ref_probe=ref_probe, 
                      num_iter=num_iter, obj_data_fit_prm=alpha, recon_win=recon_win, save_dir=save_dir)
    pmace_result = pmace_obj()
    demo_utils.plot_synthetic_img(pmace_result['object'], img_title='PMACE', **fig_args)
    
    # # recon with regularization
    # reg_pmace_obj = PMACE(pmace_recon, y_meas, patch_crds, init_obj, ref_obj=ref_obj, ref_probe=ref_probe, 
    #                       recon_win=recon_win, save_dir=save_dir, num_iter=num_iter, 
    #                       obj_data_fit_prm=alpha, add_reg=True, sigma=sigma)
    # reg_pmace_result = reg_pmace_obj()
    # demo_utils.plot_synthetic_img(reg_pmace_result['object'], img_title='reg-PMACE', **fig_args)
    
    
if __name__ == '__main__':
    main()
