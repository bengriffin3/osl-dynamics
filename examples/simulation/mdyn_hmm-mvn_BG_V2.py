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

from simulation import temporally_ica_data, whiten_data

#%% Parse command line arguments
parser = argparse.ArgumentParser(description="Run HMM on simulated data with varying noise levels")
parser.add_argument("--model", type=str, required=True, 
                    help="Model variant to use (e.g., 'means', 'stds', 'means&stds', 'stds&corrs', 'all')")
parser.add_argument("--data", type=str, required=True, 
                    help="Data variant to use (e.g., 'all', 'means', 'corrs', 'stds', 'means&corrs', 'means&stds', 'stds&corrs')")
parser.add_argument("--repetition", type=int, default=1, 
                    help="Repetition number (if applicable)")


args = parser.parse_args()

model_key = args.model
data_key = args.data
repetition = args.repetition

# noise_levels = np.arange(0, 2, 0.2)
noise_levels = [0.5]

def build_model_configs(n_states, n_channels, sequence_length=100, batch_size=8):
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
        'means&stds': config_means_stds,
        'stds&corrs': config_stds_corrs,
        'all': config_all,
    }

    return model_configs

def build_data_dict(sim_data_dict):
    """Build a dictionary mapping variant keys to Data objects."""
    data_dict = {}
    for key, ts in sim_data_dict.items():
        data_dict[key] = Data(ts)
    return data_dict


# Set random seed for reproducibility
random.seed(repetition)

# Create directory for results
results_dir = "/gpfs3/well/win-fmrib-analysis/users/psz102/git_repos/osl-dynamics/examples/simulation/results_hmm_variants_with_noise/"
os.makedirs(results_dir, exist_ok=True)

#%% Simulate data
n_states = 5
n_channels = 11
n_samples = 25600

# Build model configurations dictionary
model_configs = build_model_configs(n_states, n_channels)

# Choose the configuration for the requested model variant
if model_key not in model_configs:
    raise ValueError(f"Unknown model variant: {model_key}")

config = model_configs[model_key]


# Loop over each noise level
for noise in noise_levels:
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

    # Retrieve simulated ground truth state time courses
    sim_alpha_stc, sim_beta_stc, sim_gamma_stc = sim.mode_time_course
    # sim_alpha_stc = sim.mode_time_course

    # Retrieve simulated time series for each variant
    sim_all_time_series         = sim.time_series
    sim_means_time_series       = sim.time_series_means
    sim_corrs_time_series       = sim.time_series_corrs
    sim_stds_time_series        = sim.time_series_stds
    sim_means_corrs_time_series = sim.time_series_means_corrs
    sim_means_stds_time_series  = sim.time_series_means_stds
    sim_stds_corrs_time_series  = sim.time_series_stds_corrs


    # Create Data objects for training for each variant
    data_all          = data.Data(sim_all_time_series)  
    data_means        = data.Data(sim_means_time_series)
    data_corrs        = data.Data(sim_corrs_time_series)
    data_stds         = data.Data(sim_stds_time_series)
    data_means_corrs  = data.Data(sim_means_corrs_time_series)
    data_means_stds   = data.Data(sim_means_stds_time_series)
    data_stds_corrs   = data.Data(sim_stds_corrs_time_series)


    means = sim.means
    corrs = sim.corrs
    stds = sim.stds

    # Dictionary mapping dataset names to Data objects
    data_dict = {
        'all' : data_all,
        'means': data_means,
        'corrs': data_corrs,
        'stds': data_stds,
        'means&corrs': data_means_corrs,
        'means&stds': data_means_stds,
        'stds&corrs': data_stds_corrs,
    }

    # Dictionary mapping dataset names to ground truth state time courses
    sim_stcs = {
        'all' : sim_alpha_stc,
        'means': sim_alpha_stc,
        'corrs': sim_alpha_stc,
        'stds': sim_alpha_stc,
        'means&corrs': sim_alpha_stc,
        'means&stds': sim_alpha_stc,
        'stds&corrs': sim_alpha_stc,
    }


    # Now, for the chosen data variant (data_key), create three versions:
    # (1) Original
    # (2) Whitened version
    # (3) Temporally ICA version
    proc_types = ['original', 'whitened', 'ica']
    for proc_type in proc_types:
        if proc_type == 'original':
            d = data_dict[data_key]
        elif proc_type == 'whitened':
            d = Data(temporally_ica_data(data_dict[data_key]))
        elif proc_type == 'ica':
            ica_data, _ = temporally_ica_data(data_dict[data_key])
            d = Data(ica_data)
        else:
            continue

        # Create a new model instance for this combination
        model = Model(config)
        
        # Standardize data (if not already done)
        d.prepare({"standardize": {}})
        
        # Initialization (adjust n_init/n_epochs as needed)
        init_history = model.random_state_time_course_initialization(d, n_init=5, n_epochs=1)
        
        # Full training
        history = model.fit(d)
        
        # Calculate free energy
        free_energy = model.free_energy(d)
        
        # Get inferred state probabilities and compute Viterbi path
        alp = model.get_alpha(d)
        inf_means, inf_corrs = model.get_means_covariances()
        # inf_means, inf_stds, inf_corrs = model.get_means_stds_corrs()
        inf_stc = modes.argmax_time_courses(alp)
        
        # Re-order the inferred state time course to match the ground truth
        inf_stc, sim_stc = modes.match_modes(inf_stc, sim_stcs[data_key])
        
        # Calculate dice coefficient comparing inferred vs. simulated state time courses
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
        }
        
        print(f"Free energy: {free_energy}")
        print(f"Dice coefficient: {dice}")

        # Save the individual run results to a file with noise level in its name
        results_file = os.path.join(results_dir, f"results_{model_key}_{data_key}_{proc_type}_noise_{str(noise).replace('.', '_')}_rep{repetition}.pkl")

        with open(results_file, "wb") as f:
            pickle.dump(run_results, f)
        print(f"Noise level {noise}: Free energy: {free_energy}, Dice: {dice}")
        print(f"Saved results to {results_file}")

