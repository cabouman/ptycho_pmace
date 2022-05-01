import sys, os
from pathlib import Path
root_dir = Path(__file__).parent.absolute().parent.absolute()
sys.path.append(str(root_dir))
from ptycho_pmace.utils.utils import *
from shutil import copyfile
import argparse, yaml


'''
This file demonstrates the simulation of ptychograhy data.
'''

# arguments
parser = argparse.ArgumentParser(description='Forward Model Simulation of Ptychography Experiment.')
parser.add_argument('config_dir', type=str, help='Path to config file.',
                    nargs='?', const='configs/gen_data.yaml', default=os.path.join(root_dir, 'configs/gen_data.yaml'))
args = parser.parse_args()

# load config file
with open(args.config_dir, 'r') as f:
    config = yaml.safe_load(f)

# hyper-params
obj_dir = os.path.join(root_dir, config['data']['obj_dir'])
probe_dir = os.path.join(root_dir, config['data']['probe_dir'])
num_agts = config['data']['num_agts']
probe_spacing = config['data']['probe_spacing']
max_scan_loc_offset = config['data']['max_scan_loc_offset']
add_noise = config['data']['add_noise']
photon_rate = config['data']['photon_rate']
shot_noise_rate = config['data']['shot_noise_rate']
data_dir = os.path.join(os.path.join(root_dir, config['data']['data_dir']), 'probe_spacing_{}/photon_rate_{}/'.format(probe_spacing, photon_rate))
# noiseless_data_dir = os.path.join(config['data']['noiseless_data_dir'], 'probe_spacing_{}/'.format(probe_spacing))
display = config['data']['display']

# load ground truth images from file
obj = load_img(obj_dir)
probe = load_img(probe_dir)
m, n = probe.shape

# default parameters
rand_seed = 0
np.random.seed(rand_seed)

# generate scan positions
scan_loc = gen_scan_loc(obj, probe, num_agts, probe_spacing, randomization=True, max_offset=max_scan_loc_offset,
                        display=display, save_dir=data_dir)

# calculate the coordinates of projections
rounded_scan_loc = np.round(scan_loc)
scan_coords = np.zeros((num_agts, 4), dtype=int)
scan_coords[:, 0], scan_coords[:, 1] = rounded_scan_loc[:, 1] - m / 2, rounded_scan_loc[:, 1] + m / 2
scan_coords[:, 2], scan_coords[:, 3] = rounded_scan_loc[:, 0] - n / 2, rounded_scan_loc[:, 0] + n / 2

# # generate noise-free diffraction patterns
# noiseless_data = gen_syn_data(obj, probe, scan_coords, num_agts, add_noise=False, fft_threads=1,
#                               display=display, save_dir=data_dir+'frame_data/')
# generate noisy diffraction patterns
noisy_data = gen_syn_data(obj, probe, scan_coords, num_agts,
                          add_noise=True, photon_rate=photon_rate, shot_noise_pm=shot_noise_rate,
                          fft_threads=1, display=display, save_dir=data_dir+'frame_data/')

# save config file to output directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
copyfile(args.config_dir, data_dir + 'config.yaml')
