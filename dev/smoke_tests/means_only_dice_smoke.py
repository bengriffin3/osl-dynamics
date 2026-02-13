#!/usr/bin/env python3
"""
Means-only HMM smoke test on simulated HMM-MVN data.

Goal:
- Verify that a means-only HMM (learn_covariances=False) can recover the state path
  with high Dice coefficient when state means are well-separated.

Run:
  python dev/smoke_tests/means_only_dice_smoke.py
"""

import os
import numpy as np
import tensorflow as tf

from osl_dynamics.simulation import HMM_MVN
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.inference import modes, metrics


def make_offset_means(
    n_states: int,
    n_channels: int,
    offset: float = 4.0,
    channels_per_state: int = 2,
    seed: int = 1,
) -> np.ndarray:
    """
    Create separated state means by adding a state-specific offset to a subset of channels.
    """
    rng = np.random.default_rng(seed)
    means = 0.1 * rng.standard_normal(size=(n_states, n_channels))

    for k in range(n_states):
        start = (k * channels_per_state) % n_channels
        idx = [(start + j) % n_channels for j in range(channels_per_state)]
        means[k, idx] += offset

    return means


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(0)

    out_dir = os.path.join(os.path.dirname(__file__), "_outputs", "means_only_dice")
    os.makedirs(out_dir, exist_ok=True)

    # --- simulate means-only data ---
    n_channels = 11
    n_states = 5
    n_samples = 25600

    identity_covs = np.tile(np.eye(n_channels)[None, :, :], (n_states, 1, 1))
    sim_means = make_offset_means(
        n_states=n_states,
        n_channels=n_channels,
        offset=4.0,
        channels_per_state=2,
        seed=1,
    )

    print("Simulating data")
    sim = HMM_MVN(
        n_samples=n_samples,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob="sequence",
        stay_prob=0.9,
        means=sim_means,
        covariances=identity_covs,
    )

    data = Data(sim.time_series)
    data.standardize()

    # --- build model (means-only inference) ---
    config = Config(
        n_states=n_states,
        n_channels=n_channels,
        sequence_length=200,
        learn_means=True,
        learn_covariances=False,
        batch_size=16,
        learning_rate=0.01,
        n_epochs=20,
    )
    model = Model(config)

    # Ensure model uses the same fixed covariances as the simulation.
    # If set_covariances is a no-op when learn_covariances=False in your version, that’s fine.
    model.set_covariances(identity_covs)

    # Init then fit
    model.random_state_time_course_initialization(data, n_init=3, n_epochs=2)
    model.fit(data)

    # --- evaluate ---
    alp = model.get_alpha(data)
    stc = modes.argmax_time_courses(alp)
    inf_stc, sim_stc = modes.match_modes(stc, sim.state_time_course)
    dice = float(metrics.dice_coefficient(inf_stc, sim_stc))

    print(f"Dice coefficient: {dice:.4f}")

    # --- test-like assertion ---
    # With well-separated means this should be very high; keep threshold modest initially.
    DICE_MIN = 0.95
    assert dice >= DICE_MIN, f"Dice too low: {dice:.4f} < {DICE_MIN}"

    print("PASS ✅")
