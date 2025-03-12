print("Importing packages")
import os
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

#%% Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("noise_level", type=float, help = 'How much to scale the noise?')
parser.add_argument("repetition", type=int, help='Which repetition to run?')
# parser.add_argument("trans_prob_diag", type=int, help = 'Prior on TPM') 
# parser.add_argument("n_ICs", type=int, help = 'No. ICs of parcellation', choices = [25, 50])
# parser.add_argument("model_mean", type=int, help = 'Model mean for HMM states?', choices = [0, 1])

args = parser.parse_args()
noise_level = round(args.noise_level, 2)
repetition = args.repetition

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


# Set random seed for reproducibility
random.seed(repetition)

# Create directory for results
results_dir = "/gpfs3/well/win-fmrib-analysis/users/psz102/git_repos/osl-dynamics/examples/simulation/results_hmm_variants/"
os.makedirs(results_dir, exist_ok=True)

#%% Simulate data
n_states = 5
n_channels = 11
n_samples = 25600

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
    observation_error=noise_error
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

# Build model configurations dictionary
model_configs = build_model_configs(n_states, n_channels)

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

#%% Run models on all datasets and store results
results = {}  # Structure: results[model_name][data_name] = {free_energy, dice, inf_stc, sim_stc}

# Loop over model types
for model_name, config in model_configs.items():
    results[model_name] = {}
    # Loop over dataset types
    for data_name, data in data_dict.items():
        print(f"\nRunning model '{model_name}' on dataset '{data_name}'")
        # Create a new model instance for this combination
        model = Model(config)
        
        # Standardize data (if not already done)
        data.prepare({"standardize": {}})
        
        # Initialization (adjust n_init/n_epochs as needed)
        init_history = model.random_state_time_course_initialization(data, n_init=5, n_epochs=1)
        
        # Full training
        history = model.fit(data)
        
        # Calculate free energy
        free_energy = model.free_energy(data)
        
        # Get inferred state probabilities and compute Viterbi path
        alp = model.get_alpha(data)
        inf_means, inf_corrs = model.get_means_covariances()
        # inf_means, inf_stds, inf_corrs = model.get_means_stds_corrs()
        inf_stc = modes.argmax_time_courses(alp)
        
        # Re-order the inferred state time course to match the ground truth
        inf_stc, sim_stc = modes.match_modes(inf_stc, sim_stcs[data_name])
        
        # Calculate dice coefficient comparing inferred vs. simulated state time courses
        dice = metrics.dice_coefficient(inf_stc, sim_stc)
        
        # Save the results for this run
        results[model_name][data_name] = {
            'free_energy': free_energy,
            'dice': dice,
            'means': inf_means,
            'corrs': inf_corrs,
            # 'stds': inf_stds,
            'inf_stc': inf_stc,
            'sim_stc': sim_stc
        }
        
        print(f"Free energy: {free_energy}")
        print(f"Dice coefficient: {dice}")

        # Save individual run results to a file
        combination_file = os.path.join(results_dir, f"results_{model_name}_{data_name}_rep_{repetition}.pkl")
        with open(combination_file, "wb") as f:
            pickle.dump(results[model_name][data_name], f)
        print(f"Saved results for model '{model_name}' on dataset '{data_name}' to {combination_file}")

# Print summary of all runs
print("\nSummary of all runs:")
for m_name, runs in results.items():
    for d_name, res in runs.items():
        print(f"Model '{m_name}' on dataset '{d_name}': Free energy = {res['free_energy']}, Dice = {res['dice']}")


# Save the results object to a file in the results directory
results_file = os.path.join(results_dir, f"results_all_rep_{repetition}.pkl")
with open(results_file, "wb") as f:
    pickle.dump(results, f)
    
print(f"Results saved to {results_file}")