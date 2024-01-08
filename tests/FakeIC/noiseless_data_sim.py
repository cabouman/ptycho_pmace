import argparse, yaml
from shutil import copyfile
from pmace.utils import *
import matplotlib.pyplot as plt
from exp_funcs import *
from matplotlib.colors import LogNorm


'''
This file simulates noiseless ptychograhic data. The functionality includes:
 * Loading reference object transmittance image and reference probe profile function;
 * Generating scan locations and simulating noiseless measurements;
 * Saving the simulated intensity data to specified location.
'''
print('This script simulates noiesless ptychographic data. Script functionality includes:\
\n\t * Loading reference object transmittance image and reference probe profile function; \
\n\t * Generating scan locations and simulating noiseless measurements; \
\n\t * Saving the simulated intensity data to specified location.\n')


def build_parser():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Noiseless Data Simulation')
    parser.add_argument('config_dir', type=str, help='config_dir', nargs='?', 
                        const='config/noiseless_data_sim.yaml',
                        default='config/noiseless_data_sim.yaml')
    
    return parser


def main():
    # Load configuration from the specified YAML file
    parser = build_parser()
    args = parser.parse_args()

    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters from the config
    obj_dir = config['data']['obj_dir']
    probe_dir = config['data']['probe_dir']
    num_meas = config['data']['num_meas']
    probe_spacing = config['data']['probe_spacing']
    max_scan_loc_offset = config['data']['max_scan_loc_offset']
    add_noise = config['data']['add_noise']
    peak_photon_rate = config['data']['peak_photon_rate']
    shot_noise_rate = config['data']['shot_noise_rate']
    data_dir = config['data']['data_dir'] + 'probe_dist_{}/'.format(probe_spacing)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # Load ground truth images from file
    print("Loading reference images ...")
    obj = load_img(obj_dir)
    probe = load_img(probe_dir)

    # Set the random seed for reproducibility
    rand_seed = 0
    np.random.seed(rand_seed)

    # Initialize an array to store simulated measurements
    print("Simulating intensity data ...")
    y_meas = np.zeros((num_meas, probe.shape[0], probe.shape[1]), dtype=np.float64)

    # Generate scan positions
    scan_loc = gen_scan_loc(obj, probe, num_meas, probe_spacing, randomization=True, max_offset=max_scan_loc_offset)
    df = pd.DataFrame({'FCx': scan_loc[:, 0], 'FCy': scan_loc[:, 1]})
    df.to_csv(data_dir + 'Translations.tsv.txt')

    # Calculate the coordinates of projections
    scan_coords = get_proj_coords_from_data(scan_loc, y_meas)    

    # Generate noisefree diffraction patterns
    noisy_data = gen_syn_data(obj, probe, scan_coords, add_noise=add_noise, peak_photon_rate=float(peak_photon_rate), 
                              shot_noise_pm=shot_noise_rate, save_dir=data_dir + 'frame_data/')
    
    # Save simulated diffraction pattern 
    plot_meas_img(noisy_data[0], save_dir=data_dir)
    
    # Plot ground truth object image
    plot_FakeIC_img(obj, img_title='GT', display_win=None, save_dir=data_dir)  
    
    # Save ground truth images to output file
    save_tiff(obj, data_dir + 'ref_object.tiff')
    save_tiff(probe, data_dir + 'ref_probe.tiff')

    # Save config file to output directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    copyfile(args.config_dir, data_dir + 'config.yaml')

    # Data simulation completed
    print("Simulated data saved to directory '%s'" % data_dir)
    

if __name__ == '__main__':
    main()