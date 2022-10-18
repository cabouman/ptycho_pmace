import argparse, yaml
import datetime as dt
from shutil import copyfile
from utils.utils import *
from ptycho import *
from ptycho.pie import *
from real_experiment_funcs import *


'''
This file demonstrates the reconstruction of complex transmittance image on gold balls data [1]. 
[1] Marchesini, Stefano. Ptychography Gold Ball Example Dataset. United States: N. p., 2017. Web. doi:10.11577/1454414.
'''


def build_parser():
    parser = argparse.ArgumentParser(description='Ptychographic image reconstruction on gold balls data set.')
    parser.add_argument('config_dir', type=str, help='Configuration file.', nargs='?', const='GoldBalls_data.yaml',
                        default='config/GoldBalls_data.yaml')
    return parser


def main():
    # Arguments
    parser = build_parser()
    args = parser.parse_args()

    # Load config file
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)

    # Read data from config file
    probe_dir = config['data']['probe_dir']
    data_dir = config['data']['data_dir']
    display = config['data']['display']
    window_coords = config['data']['window_coords']
    out_dir = config['output']['out_dir']

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
    ref_probe = load_img(probe_dir)

    # Load intensity only measurements(data) from file and pre-process the data
    y_tmp = load_measurement(data_dir + 'frame_data/')
    tukey_win = gen_tukey_2D_window(np.zeros_like(y_tmp[0]))
    y_meas = y_tmp * tukey_win

    # Load scan points
    scan_loc_file = pd.read_csv(data_dir + 'Translations.tsv.txt', sep=None, engine='python', header=0)
    scan_loc = scan_loc_file[['FCx', 'FCy']].to_numpy()

    # calculate the coordinates of projections
    patch_bounds = get_proj_coords_from_data(scan_loc + y_meas.shape[1] / 2, y_meas)

    # Generate formulated initial guess for reconstruction
    img_sz = math.ceil(np.amax(scan_loc + np.maximum(y_meas.shape[1], y_meas.shape[2])))
    init_obj = np.ones((img_sz, img_sz), dtype=np.complex64)

    # Produce the cover/window for comparison
    if window_coords is not None:
        xmin, xmax, ymin, ymax = window_coords[0], window_coords[1], window_coords[2], window_coords[3]
        recon_win = np.zeros(init_obj.shape)
        recon_win[xmin:xmax, ymin:ymax] = 1
    else:
        recon_win = None

    # Reconstruction parameters
    recon_args = dict(init_obj=init_obj, ref_probe=ref_probe, recon_win=recon_win, 
                      num_iter=config['recon']['num_iter'], joint_recon=config['recon']['joint_recon'])
    fig_args = dict(display_win=recon_win, display=display)

    # ePIE recon
    obj_step_sz = config['ePIE']['obj_step_sz']
    epie_dir = save_dir + 'ePIE/'
    epie_result = epie_recon(y_meas, patch_bounds, obj_step_sz=obj_step_sz, save_dir=epie_dir, **recon_args)
    # Plot reconstructed image
    plot_goldball_img(epie_result['object'], save_dir=epie_dir, **fig_args)

    # Accelerated Wirtinger Flow (AWF) recon
    awf_dir = save_dir + 'AWF/'
    awf_result = wf.wf_recon(y_meas, patch_bounds, accel=True, save_dir=awf_dir, **recon_args)
    # Plot reconstructed image
    plot_goldball_img(awf_result['object'], save_dir=awf_dir, **fig_args)

    # SHARP recon
    relax_pm = config['SHARP']['relax_pm']
    sharp_dir = save_dir + 'SHARP/'
    sharp_result = sharp.sharp_recon(y_meas, patch_bounds, relax_pm=relax_pm, save_dir=sharp_dir,**recon_args)
    # Plot reconstructed image
    plot_goldball_img(sharp_result['object'], save_dir=sharp_dir, **fig_args)

    # # SHARP+ recon
    # sharp_plus_pm = config['SHARP_plus']['relax_pm']
    # sharp_plus_dir = save_dir + 'SHARP_plus/'
    # sharp_plus_result = sharp.sharp_plus_recon(y_meas, patch_bounds, relax_pm=sharp_plus_pm, save_dir=sharp_plus_dir, **recon_args)
    # # Plot reconstructed image
    # plot_goldball_img(sharp_plus_result['object'], save_dir=sharp_plus_dir, **fig_args)

    # PMACE recon
    alpha = config['PMACE']['alpha']                
    rho = config['PMACE']['rho']                       # Mann averaging parameter
    probe_exp = config['PMACE']['probe_exponent']      # probe exponent
    pmace_dir = save_dir + 'PMACE/'
    pmace_result = pmace.pmace_recon(y_meas, patch_bounds, obj_data_fit_prm=alpha, rho=rho, probe_exp=probe_exp, 
                                     add_reg=False, save_dir=pmace_dir, **recon_args)
    # Plot reconstructed image
    plot_goldball_img(pmace_result['object'], save_dir=pmace_dir, **fig_args)

    # Save config file to output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile(args.config_dir, save_dir + 'config.yaml')


if __name__ == '__main__':
    main()
