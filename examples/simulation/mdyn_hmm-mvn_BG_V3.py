print("Importing packages")
import os
import sys
import random
import numpy as np
import pickle
import argparse
from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.simulation import HMM_MVN, MDyn_HMM_MVN
from osl_dynamics.data import Data
from osl_dynamics.utils import plotting



# Get the current working directory and move up one level to the parent directory, then to 'modules'
sys.path.append('/well/win-fmrib-analysis/users/psz102/profumo_hmm/modules')

from simulation import temporal_ica_data, whiten_data_temporal

#%% Parse command line arguments
parser = argparse.ArgumentParser(description="Run HMM on simulated data with varying noise levels")
parser.add_argument("--model", type=str, required=True, 
                    help="Model variant to use (e.g., 'means', 'stds', 'meansandstds', 'stdsandcorrs', 'all')")
parser.add_argument("--data", type=str, required=True, 
                    help="Data variant to use (e.g., 'all', 'means', 'corrs', 'stds', 'meansandcorrs', 'meansandstds', 'stdsandcorrs', 'means_then_covs')")
parser.add_argument("--repetition", type=int, default=1, 
                    help="Repetition number (if applicable)")


args = parser.parse_args()

model_key = args.model
data_key = args.data
repetition = args.repetition

# noise_levels = np.arange(0, 2, 0.1)
noise_levels = [0, 0.5]

def build_model_configs(n_states, n_channels, sequence_length=200, batch_size=16):
    """Builds and returns a dictionary of model configurations.
    
    Comments indicate the options not used.
    """
    #%% Define model configurations
    config_means = Config(
        n_states=n_states,
        n_channels=n_channels,
        sequence_length=sequence_length,
        learn_means=True,
        learn_covariances=False,
        diagonal_covariances=False,
        learn_trans_prob=True,
        batch_size=batch_size,
        learning_rate=0.01,
        n_epochs=20,
    )

    # config_corrs = Config(
    #     n_states=n_states,
    #     n_channels=n_channels,
    #     sequence_length=sequence_length,
    #     learn_means=False,
    #     learn_covariances=True,
    #     # diagonal_covariances=False,
    #     learn_trans_prob=True,
    #     batch_size=16,
    #     learning_rate=0.01,
    #     n_epochs=20,
    # )

    config_stds = Config(
        n_states=n_states,
        n_channels=n_channels,
        sequence_length=sequence_length,
        learn_means=False,
        learn_covariances=True,
        diagonal_covariances=True,
        learn_trans_prob=True,
        batch_size=batch_size,
        learning_rate=0.01,
        n_epochs=20,
    )

    # It currently isn't possible to learn hmm covariances without variances
    # # 1. Means & Corrs
    # config_means_corrs = Config(
    #     n_states=n_states,
    #     n_channels=n_channels,
    #     sequence_length=sequence_length,
    #     learn_means=True,
    #     learn_covariances=True,  # learn full covariance structure (i.e. correlations)
    #     diagonal_covariances=False,  # we don't restrict to diagonal => using full covariances
    #     learn_trans_prob=True,
    #     batch_size=batch_size,
    #     learning_rate=0.01,
    #     n_epochs=20,
    # )

    # 2. Means & Stds
    config_means_stds = Config(
        n_states=n_states,
        n_channels=n_channels,
        sequence_length=sequence_length,
        learn_means=True,
        learn_covariances=True,  # we want to learn covariances, but restrict to diagonal for stds
        diagonal_covariances=True,
        learn_trans_prob=True,
        batch_size=batch_size,
        learning_rate=0.01,
        n_epochs=20,
    )

    # 3. Stds & Corrs
    config_stds_corrs = Config(
        n_states=n_states,
        n_channels=n_channels,
        sequence_length=sequence_length,
        learn_means=False,
        learn_covariances=True,
        diagonal_covariances=False,  # we want to capture both correlations and then later combine with stds, so full covariances
        learn_trans_prob=True,
        batch_size=batch_size,
        learning_rate=0.01,
        n_epochs=20,
    )

    # 3. All
    config_all = Config(
        n_states=n_states,
        n_channels=n_channels,
        sequence_length=sequence_length,
        learn_means=True,
        learn_covariances=True,
        diagonal_covariances=False,  # we want to capture both correlations and then later combine with stds, so full covariances
        learn_trans_prob=True,
        batch_size=batch_size,
        learning_rate=0.01,
        n_epochs=20,
    )

    # Dictionary mapping model names to their configs
    model_configs = {
        'means': config_means,
        # 'corrs': config_corrs,         # not used
        'stds': config_stds,
        # 'means_corrs': config_means_corrs,  # not used
        'meansandstds': config_means_stds,
        'stdsandcorrs': config_stds_corrs,
        'all': config_all,
    }

    return model_configs

def build_data_dict(sim_data_dict):
    """Build a dictionary mapping variant keys to Data objects."""
    data_dict = {}
    for key, ts in sim_data_dict.items():
        data_dict[key] = Data(ts)
    return data_dict

results_dir = "/gpfs3/well/win-fmrib-analysis/users/psz102/git_repos/osl-dynamics/examples/simulation/results_hmm_variants_with_noise/temporal_ica/"
os.makedirs(results_dir, exist_ok=True)

#%% Simulation parameters
n_states = 5
n_channels = 11
n_samples = 25600

# Build model configurations dictionary
model_configs = build_model_configs(n_states, n_channels)
if model_key not in model_configs:
    raise ValueError(f"Unknown model variant: {model_key}")
config = model_configs[model_key]


for rep in range(repetition, repetition + 1):
    # Set random seed for reproducibility
    random.seed(rep)
        
    # Loop over noise levels
    for noise in noise_levels:
        noise = round(noise, 1)

        print(f"\nRunning for noise level: {noise}")
        print("Simulating data")
        sim = MDyn_HMM_MVN(
            n_samples=n_samples,
            n_states=n_states,
            n_modes=n_states,
            n_channels=n_channels,
            trans_prob="sequence",
            stay_prob=0.9,
            means="random",
            covariances="random",
            observation_error=noise
        )

        sim_alpha_stc, sim_beta_stc, sim_gamma_stc = sim.mode_time_course
        
        # Retrieve simulated time series for each variant
        sim_data_dict = {
            'all': sim.time_series,
            'means': sim.time_series_means,
            'corrs': sim.time_series_corrs,
            'stds': sim.time_series_stds,
            'meansandcorrs': sim.time_series_means_corrs,
            'meansandstds': sim.time_series_means_stds,
            'stdsandcorrs': sim.time_series_stds_corrs,
            'meansthencovs': sim.time_series_means_then_covs
        }
        
        # Create Data objects for each simulated variant
        data_dict = build_data_dict(sim_data_dict)
        
        # Use the same ground truth state time courses for all variants
        sim_stcs = {k: sim.mode_time_course[0] for k in sim_data_dict.keys()}
        
        # For the chosen data variant, create three processed versions:
        # proc_types = ['original', 'whitened', 'ica']
        proc_types = ['original', 'ica']
        for proc_type in proc_types:
            if proc_type == 'original':
                d = data_dict[data_key]
            elif proc_type == 'whitened':
                # Get raw data from the original Data object
                raw = data_dict[data_key].time_series()
                whitened_ts = whiten_data_temporal(raw)
                d = Data(whitened_ts)
            elif proc_type == 'ica':
                raw = data_dict[data_key].time_series()
                ica_data, _ = temporal_ica_data(raw)
                d = Data(ica_data)
            else:
                continue

            formatted_noise = f"{noise:.1f}".replace('.', '_')
            results_file = os.path.join(results_dir, f"results_{model_key}_{data_key}_{proc_type}_noise_{formatted_noise}_rep{rep}.pkl")
            if os.path.exists(results_file):
                print(f"Results for {results_file} already exist. Skipping.")
                continue

            # Create new model instance and run the analysis
            model = Model(config)
            d.prepare({"standardize": {}})
            model.random_state_time_course_initialization(d, n_init=5, n_epochs=1)
            history = model.fit(d)
            free_energy = model.free_energy(d)
            alp = model.get_alpha(d)
            inf_means, inf_corrs = model.get_means_covariances()
            inf_stc = modes.argmax_time_courses(alp)

            noise_str = f"{noise:g}".replace('.', '_')

            if data_key == 'meansthencovs':
                # Use the two ground truths from the simulation:
                alpha_gt = sim.mode_time_course[0]  # ground truth for means
                beta_gt  = sim.mode_time_course[1]  # ground truth for covariances

                # Match and compute Dice for alpha (means) comparison:
                inf_stc_alpha, sim_stc_alpha = modes.match_modes(inf_stc, alpha_gt)
                dice_alpha = metrics.dice_coefficient(inf_stc_alpha, sim_stc_alpha)

                # Match and compute Dice for beta (covariances) comparison:
                inf_stc_beta, sim_stc_beta = modes.match_modes(inf_stc, beta_gt)
                dice_beta = metrics.dice_coefficient(inf_stc_beta, sim_stc_beta)

                # Create separate dictionaries:
                run_results_alpha = {
                    'free_energy': free_energy,
                    'dice': dice_alpha,
                    'means': inf_means,
                    'corrs': inf_corrs,
                    'inf_stc': inf_stc_alpha,
                    'sim_stc': sim_stc_alpha,
                    'noise': noise,
                    'model': model_key,
                    'data': data_key,
                    'processing': proc_type,
                }
                
                run_results_beta = {
                    'free_energy': free_energy,
                    'dice': dice_beta,
                    'means': inf_means,
                    'corrs': inf_corrs,
                    'inf_stc': inf_stc_beta,
                    'sim_stc': sim_stc_beta,
                    'noise': noise,
                    'model': model_key,
                    'data': data_key,
                    'processing': proc_type,
                }

                # Save results for the alpha (means) comparison:
                results_file_alpha = os.path.join(
                    results_dir, 
                    f"results_{model_key}_{data_key}_comparetomeans_{proc_type}_noise_{str(noise_str).replace('.', '_')}_rep{rep}.pkl"
                )
                with open(results_file_alpha, "wb") as f:
                    pickle.dump(run_results_alpha, f)
                print(f"Saved alpha (means) comparison results to {results_file_alpha}")

                # Save results for the beta (covariances) comparison:
                results_file_beta = os.path.join(
                    results_dir, 
                    f"results_{model_key}_{data_key}_comparetocovs_{proc_type}_noise_{str(noise_str).replace('.', '_')}_rep{rep}.pkl"
                )
                with open(results_file_beta, "wb") as f:
                    pickle.dump(run_results_beta, f)
                print(f"Saved beta (covariances) comparison results to {results_file_beta}")
            else:
                # Regular case: use the ground truth for the requested data variant.
                inf_stc, sim_stc = modes.match_modes(inf_stc, sim_stcs[data_key])
                dice = metrics.dice_coefficient(inf_stc, sim_stc)
                run_results = {
                    'free_energy': free_energy,
                    'dice': dice,
                    'means': inf_means,
                    'corrs': inf_corrs,
                    'inf_stc': inf_stc,
                    'sim_stc': sim_stc,
                    'noise': noise,
                    'model': model_key,
                    'data': data_key,
                    'processing': proc_type,
                }
                formatted_noise = f"{noise:.1f}".replace('.', '_')
                results_file = os.path.join(results_dir, f"results_{model_key}_{data_key}_{proc_type}_noise_{formatted_noise}_rep{rep}.pkl")
                with open(results_file, "wb") as f:
                    pickle.dump(run_results, f)
                print(f"Noise {noise}, processing {proc_type}: Free energy: {free_energy}, Dice: {dice}")
                print(f"Saved results to {results_file}")

                    
