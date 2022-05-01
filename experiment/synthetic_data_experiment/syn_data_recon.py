import sys, os
from pathlib import Path
root_dir = Path(__file__).parent.absolute().parent.absolute().parent.absolute()
sys.path.append(str(root_dir))
from ptycho_pmace.utils.utils import *
from ptycho_pmace.ptycho import *
import datetime as dt
from shutil import copyfile
import argparse, yaml

print(root_dir)
'''
This file reconstructs complex transmittance image by processing the synthetic data with different probe spacing and 
compares results from reconstruction approaches. 
'''

def build_parser():
    parser = argparse.ArgumentParser(description='Ptychographic image reconstruction.')
    parser.add_argument('config_dir', type=str, help='Configuration file.', nargs='?', const='syn_data_recon.yaml',
                        default=os.path.join(root_dir, 'experiment/synthetic_data_experiment/config/syn_data_recon.yaml'))
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

    # Load ground truth images from file
    obj_ref = load_img(obj_dir)
    probe_ref = load_img(probe_dir)

    # Read recon settings from config file
    init_guess_form = config['recon']['init_guess_form']
    display = config['recon']['display']
    window_coords = config['recon']['window_coords']
    num_iter = config['recon']['num_iter']

    # Determine output directory with time stamp
    today_date = dt.date.today()
    date_time = dt.datetime.strftime(today_date, '%Y-%m-%d_%H_%M/')
    save_dir = os.path.join(os.path.join(root_dir, config['recon']['out_dir']), date_time)

    # Create the directory
    try:
        os.makedirs(save_dir, exist_ok=True)
        print("Output directory '%s' created successfully" % save_dir)
    except OSError as error:
        print("Output directory '%s' can not be created" % save_dir)

    # Set reconstruction parameters
    rand_seed = 0
    np.random.seed(rand_seed)
    recon_pm = pd.read_csv(data_dir + 'Tuned_param.tsv.txt', sep=None, engine='python', header=0)
    probe_dist = recon_pm['probe_spacing'].to_numpy()
    epie_pm = recon_pm['epie_step_sz'].to_numpy()
    sharp_pm = recon_pm['sharp_relax_pm'].to_numpy()
    sharp_plus_pm = recon_pm['sharp_plus_relax_pm'].to_numpy()
    pmace_pm = recon_pm['pmace_alpha'].to_numpy()
    reg_wgt = recon_pm['reg_pmace_reg_wgt'].to_numpy()
    reg_psd = recon_pm['reg_pmace_denoising_pm'].to_numpy()

    # Produce the cover/window for comparison
    if window_coords is not None:
        xmin, xmax, ymin, ymax = window_coords[0], window_coords[1], window_coords[2], window_coords[3]
        display_win = np.zeros(obj_ref.shape)
        display_win[xmin:xmax, ymin:ymax] = 1
    else:
        display_win = None

    for idx in range(len(probe_dist)):
        # Obtain probe spacing d
        d = probe_dist[idx]

        # Load intensity only measurements(data) from file and pre-process the data
        diffract_data = load_measurement(data_dir + 'probe_spacing_{}/photon_rate_100000.0/frame_data/'.format(d), display=display)

        # Load scan points
        scan_loc_data = pd.read_csv(data_dir + 'probe_spacing_{}/photon_rate_100000.0/Translations.tsv.txt'.format(d),
                                    sep=None, engine='python', header=0)
        scan_loc = scan_loc_data[['FCx', 'FCy']].to_numpy()

        # calculate coordinates of projections
        projection_coords = get_proj_coords_from_data(scan_loc, diffract_data)
        # Generate formulated initial guess for reconstruction
        init_obj = gen_init_obj(obj_ref, probe_ref, projection_coords, diffract_data, formation=init_guess_form, display=display)

        # ePIE recon
        epie_result = pie.epie_recon(init_obj, diffract_data, projection_coords, obj_ref, probe_ref,
                                     num_iter=num_iter, step_sz=epie_pm[idx], cstr_win=display_win,
                                     save_dir=save_dir + 'probe_spacing_{}/ePIE/step_sz_{}/'.format(d, epie_pm[idx]))

        # AWF recon
        awf_result = wf.wf_recon(init_obj, diffract_data, projection_coords, obj_ref, probe_ref,
                                 num_iter=num_iter, accel=True, cstr_win=display_win,
                                 save_dir=save_dir + 'probe_spacing_{}/AWF/'.format(d))

        # SHARP recon
        sharp_dir = save_dir + 'probe_spacing_{}/SHARP/relax_pm_{}/'.format(d, sharp_pm[idx])
        sharp_result = sharp.sharp_recon(init_obj, diffract_data, projection_coords, obj_ref, probe_ref,
                                         num_iter=num_iter, relax_pm=sharp_pm[idx],
                                         cstr_win=display_win, save_dir=sharp_dir)
        # SHARP+ recon
        sharp_plus_dir = save_dir + 'probe_spacing_{}/SHARP_plus/relax_pm_{}/'.format(d, sharp_plus_pm[idx])
        sharp_plus_result = sharp.sharp_plus_recon(init_obj, diffract_data, projection_coords, obj_ref, probe_ref,
                                                   num_iter=num_iter, relax_pm=sharp_plus_pm[idx],
                                                   cstr_win=display_win, save_dir=sharp_plus_dir)

        # PMACE recon
        pmace_dir = save_dir + 'probe_spacing_{}/PMACE/NSR_{}/'.format(d, pmace_pm[idx])
        pmace_result = pmace.pmace_recon(init_obj, diffract_data, projection_coords, obj_ref, probe_ref,
                                         num_iter=num_iter, obj_nsr_pm=pmace_pm[idx], rho=0.5, probe_exp=1.25,
                                         cstr_win=display_win, save_dir=pmace_dir)

        # reg-PMACE recon
        reg_pmace_dir = save_dir + 'probe_spacing_{}/reg_PMACE/denoise_pm_{}/'.format(d, reg_psd[idx])
        reg_PMACE_result = pmace.reg_pmace_recon(init_obj, diffract_data, projection_coords, obj_ref, probe_ref,
                                                 num_iter=num_iter, obj_nsr_pm=pmace_pm[idx], rho=0.5, probe_exp=1.25,
                                                 reg_wgt=0.6, noise_std=reg_psd[idx],
                                                 cstr_win=display_win, save_dir=reg_pmace_dir)

        # Convergence plots
        num_iter = np.asarray(np.arange(num_iter+1))
        init_err = compute_nrmse(init_obj * display_win, obj_ref * display_win, display_win)
        plt.figure(num=None, figsize=(10, 6), dpi=400, facecolor='w', edgecolor='k')
        plt.semilogy(num_iter, np.insert(epie_result['obj_err'], 0, init_err), 'C5', label=r'ePIE')
        plt.semilogy(num_iter, np.insert(awf_result, 0, init_err), 'C4', label=r'AWF')
        plt.semilogy(num_iter, np.insert(sharp_result, 0, init_err), 'C3', label=r'SHARP')
        plt.semilogy(num_iter, np.insert(sharp_plus_result, 0, init_err), 'C2', label=r'SHARP+')
        plt.semilogy(num_iter, np.insert(pmace_result, 0, init_err), 'C1', label=r'PMACE')
        plt.semilogy(num_iter, np.insert(reg_PMACE_result, 0, init_err), 'C0', label=r'reg-PMACE')
        plt.ylabel('NRMSE (in log scale) ')
        plt.xlabel('Number of iterations')
        plt.legend(loc='best')
        plt.title('Convergence plots (probe_spacing = {})'.format(d))
        plt.grid(True)
        plt.savefig(save_dir + 'probe_spacing_{}/photon_rate_100000.0/convergence_plot'.format(d))
        plt.clf()

    # Save config file to output directory
    copyfile(args.config_dir, save_dir + 'config.yaml')


if __name__ == '__main__':
    main()
