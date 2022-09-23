import argparse, yaml
from utils.utils import *


'''
This file downsamples CuFoam data by reducing scan points. 
'''


def build_parser():
    parser = argparse.ArgumentParser(description='CuFoam data downsampling.')
    parser.add_argument('config_dir', type=str, help='Configuration file.', nargs='?',
                        const='config/data_downsampling.yaml', default='config/data_downsampling.yaml')
    return parser


def main():
    # Arguments
    parser = build_parser()
    args = parser.parse_args()

    # Load config file
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)

    # Read data from config file
    display = config['data']['display']
    data_dir = config['data']['data_dir']
    save_dir = config['output']['save_dir']

    # Create the directory
    os.makedirs(save_dir, exist_ok=True)

    # Load intensity only measurements(data) from file and pre-process the data
    y_meas = load_measurement(data_dir + 'frame_data/')

    # Load scan points
    scan_loc_file = pd.read_csv(data_dir + 'Translations.tsv.txt', sep=None, engine='python', header=0)
    scan_loc = scan_loc_file[['FCx', 'FCy']].to_numpy()

    # Reduce data via removing the points around turning corners
    reduced_scan_loc = scan_loc[(scan_loc[:, 0] >= config['data']['stop_pt_left']) & (scan_loc[:, 0] <= config['data']['stop_pt_right'])]
    meas_idx = []
    for idx in range(len(reduced_scan_loc)):
        curr_point = reduced_scan_loc[idx]
        meas_idx.append(np.where((scan_loc[:, 0] == curr_point[0]) & (scan_loc[:, 1] == curr_point[1]))[0][0])
    reduced_meas = y_meas[np.sort(meas_idx)]

    # Downsample the measurements by a factor of 2
    downsampling_factor = 2
    downsampled_idx = np.arange(0, len(reduced_scan_loc), downsampling_factor)
    downsampled_scan_loc = reduced_scan_loc[downsampled_idx]
    downsampled_meas = reduced_meas[downsampled_idx]

    # Skip lines to increase larger vertical probe spacings
    final_meas, final_scan_loc = drop_line(downsampled_meas, downsampled_scan_loc)
    for j in range(len(final_meas)):
        tiff.imwrite(data_dir + 'frame_data_{}.tiff'.format(j), np.asarray(final_meas[j]))

    df = pd.DataFrame({'FCx': final_scan_loc[:, 0], 'FCy': final_scan_loc[:, 1]})
    df.to_csv(save_dir + 'Translations.tsv.txt')

    if display:
        plt.plot(final_scan_loc[:, 0], final_scan_loc[:, 1], 'r.')
        plt.plot(final_scan_loc[:, 0], final_scan_loc[:, 1], 'b-')
        ax = plt.gca()
        ax.invert_yaxis()
        plt.title('downsampled scan positions')
        plt.show()
        plt.clf()


if __name__ == '__main__':
    main()
