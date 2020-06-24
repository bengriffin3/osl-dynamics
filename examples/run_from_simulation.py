import logging
import pathlib

import numpy as np
import vrad.inference.metrics
import yaml
from vrad import array_ops
from vrad.data import Data
from vrad.inference import metrics
from vrad.inference.gmm import find_cholesky_decompositions, learn_mu_sigma
from vrad.inference.models import create_model
from vrad.inference.tf_ops import gpu_growth, train_predict_dataset
from vrad.simulation import HiddenSemiMarkovSimulation
from vrad.utils import plotting

# Restrict GPU memory usage
gpu_growth()

script_dir = str(pathlib.Path(__file__).parent.absolute())

logger = logging.getLogger("VRAD")
logger.setLevel(logging.INFO)

# Get all configuration options from a YAML file
logger.info("Reading configuration from 'run_from_simulation.yml'")
with open(script_dir + "/run_from_simulation.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

# Simulate data and store in a Data object
logger.info("Simulating data")
sim = HiddenSemiMarkovSimulation(**config["HiddenSemiMarkovSimulation"])
meg_data = Data(sim)
state_time_course = sim.state_time_course
n_states = sim.n_states

# Perform standard scaling/PCA
meg_data.standardize(pre_scale=True, do_pca=False, post_scale=False, n_components=1)

# Create TensorFlow Datasets
logger.info("Creating datasets")
training_dataset, prediction_dataset = train_predict_dataset(
    time_series=meg_data, **config["dataset"]
)

# Model states using a GaussianMixtureModel
logger.info("Fitting Gaussian mixture model")
covariance, means = learn_mu_sigma(
    data=meg_data,
    n_states=n_states,
    take_random_sample=20000,
    retry_attempts=5,
    learn_means=False,
    gmm_kwargs={
        "n_init": 1,
        "verbose": 2,
        "verbose_interval": 50,
        "max_iter": 10000,
        "tol": 1e-6,
    },
)

cholesky_covs = find_cholesky_decompositions(covariance, means, True)

# Create model
logger.info("Creating InferenceRNN")
config["model"]["n_channels"] = meg_data.shape[1]
model = create_model(
    **config["model"], initial_mean=means, initial_pseudo_cov=cholesky_covs
)

# Create trainer and callback for checking dice coefficient
logger.info("Creating trainer")

# Train
n_epochs = 100
logger.info(f"Training for {n_epochs} epochs")
model.fit(training_dataset, epochs=n_epochs)
logger.info("Training complete")

# Analysis

inf_stc = array_ops.get_one_hot(
    np.concatenate(model.predict(prediction_dataset)[3]).argmax(axis=1)
)

aligned_stc, aligned_inf_stc = array_ops.align_arrays(state_time_course, inf_stc)
matched_stc, matched_inf_stc = array_ops.match_states(aligned_stc, aligned_inf_stc)

print(f"Dice coefficient is {metrics.dice_coefficient(matched_stc, matched_inf_stc)}")

plotting.compare_state_data(matched_inf_stc, matched_stc)

plotting.plot_state_sums(matched_stc)
plotting.plot_state_sums(matched_inf_stc)

plotting.confusion_matrix(matched_stc, matched_inf_stc)

plotting.plot_state_highlighted_data(meg_data, matched_inf_stc, n_time_points=10000)

plotting.plot_state_lifetimes(matched_inf_stc)