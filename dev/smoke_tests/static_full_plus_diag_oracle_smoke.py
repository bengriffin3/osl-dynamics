#!/usr/bin/env python3
"""
Oracle-init smoke test for the NEW covariance_structure='static_full_plus_diag'.

We simulate from: Σ_k = C + D_k (means=0)
Then we fit the HMM with covariance_structure='static_full_plus_diag'.

Key points:
- We standardize the data (Data.standardize()).
- We transform true covariances into the standardized space.
- We call model.set_covariances(covs_init) ONCE (oracle-ish init).
- Run ONCE (fast-ish).

Expected for the NEW model:
- Dice should be high under oracle init.
- Inferred covs should have ~identical off-diagonals across states
  (max_off_diff should be ~0 up to small numerical noise).

Run:
  python dev/smoke_tests/static_full_plus_diag_oracle_smoke.py
"""

import os
import numpy as np
import tensorflow as tf

from osl_dynamics.simulation import HMM_MVN
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.inference import modes, metrics


def make_pd_cov(rng: np.random.Generator, n_channels: int, jitter: float = 1e-3) -> np.ndarray:
    A = rng.standard_normal((n_channels, n_channels))
    C = A @ A.T
    C = C / np.mean(np.diag(C))
    C = C + jitter * np.eye(n_channels)
    return C


def simulate_C_plus_D(
    n_samples: int = 25600,
    n_states: int = 5,
    n_channels: int = 11,
    stay_prob: float = 0.9,
    seed: int = 1,
    c_scale: float = 0.2,
    channels_per_state: int = 6,
    diag_bump: float = 4.0,
):
    """
    Construct Σ_k = C + diag(d_k) with shared off-diagonals across k.
    """
    rng = np.random.default_rng(seed)

    C = c_scale * make_pd_cov(rng, n_channels)

    D = np.ones((n_states, n_channels))
    for k in range(n_states):
        start = (k * channels_per_state) % n_channels
        idx = [(start + j) % n_channels for j in range(channels_per_state)]
        D[k, idx] += diag_bump

    covs = np.repeat(C[None, :, :], n_states, axis=0) + np.array([np.diag(D[k]) for k in range(n_states)])

    sim = HMM_MVN(
        n_samples=n_samples,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob="sequence",
        stay_prob=stay_prob,
        means="zero",
        covariances=covs,
    )
    return sim, covs


def covs_to_standardized_space(covs: np.ndarray, X_raw: np.ndarray) -> np.ndarray:
    """
    If X_std = (X_raw - mean)/std (per-channel), then:
      Cov[X_std] = S^{-1} Cov[X_raw] S^{-1}, where S = diag(std).
    We assume Data.standardize() uses per-channel std in the same way.
    """
    std = X_raw.std(axis=0, ddof=0)
    S_inv = np.diag(1.0 / std)
    return np.einsum("ij,kjl,lm->kim", S_inv, covs, S_inv)


def cov_stats(covs: np.ndarray):
    """
    Returns:
      max_off_diff: max absolute off-diagonal difference across states relative to state 0
      mean_diag_std: mean (over channels) of std over states of the diagonal entries
    """
    K, P, _ = covs.shape
    mask = np.ones((P, P), dtype=covs.dtype) - np.eye(P, dtype=covs.dtype)
    off = covs * mask[None, :, :]
    max_off_diff = float(np.max(np.abs(off - off[0:1])))
    diag = np.diagonal(covs, axis1=1, axis2=2)
    mean_diag_std = float(np.mean(np.std(diag, axis=0)))
    return max_off_diff, mean_diag_std


if __name__ == "__main__":
    # Determinism (helps reduce flakiness)
    tf.keras.utils.set_random_seed(0)

    out_dir = os.path.join(os.path.dirname(__file__), "_outputs", "static_full_plus_diag_oracle")
    os.makedirs(out_dir, exist_ok=True)

    # --- simulate ---
    n_states, n_channels, n_samples = 5, 11, 25600
    sim, covs_true_raw = simulate_C_plus_D(
        n_samples=n_samples,
        n_states=n_states,
        n_channels=n_channels,
        stay_prob=0.9,
        seed=1,
        c_scale=0.2,
        channels_per_state=6,
        diag_bump=4.0,
    )

    # --- standardize data ---
    X_raw = sim.time_series
    data = Data(X_raw)
    data.standardize()

    # --- true covariances in standardized space (what the model sees) ---
    covs_true_std = covs_to_standardized_space(covs_true_raw, X_raw)
    max_off_diff_true, mean_diag_std_true = cov_stats(covs_true_std)
    print("TRUE cov stats (standardized space):",
          f"max_off_diff={max_off_diff_true:.3g}, mean_diag_std={mean_diag_std_true:.3g}")

    # Simulation sanity: off-diagonals should be identical across states (up to numerical noise)
    assert max_off_diff_true <= 1e-6, f"Simulation off-diagonals not identical: {max_off_diff_true:.3g}"

    # --- build model: NEW covariance structure ---
    config = Config(
        n_states=n_states,
        n_channels=n_channels,
        sequence_length=200,
        learn_means=False,
        learn_covariances=True,
        covariance_structure="static_full_plus_diag",  # <-- NEW
        batch_size=16,
        learning_rate=1e-3,
        n_epochs=30,
    )
    model = Model(config)

    # --- oracle-ish init of FULL covariances (setter should decompose into shared+diag) ---
    rng = np.random.default_rng(0)
    covs_init = covs_true_std + 1e-4 * rng.standard_normal(covs_true_std.shape)
    covs_init = 0.5 * (covs_init + np.transpose(covs_init, (0, 2, 1)))
    covs_init += 1e-6 * np.eye(n_channels)[None, :, :]

    model.set_covariances(covs_init)
    print("Initialised covariances near TRUE (standardized space).")

    # --- fit ---
    model.fit(data)

    # --- evaluate segmentation ---
    alp = model.get_alpha(data)
    stc = modes.argmax_time_courses(alp)
    inf_stc, sim_stc = modes.match_modes(stc, sim.state_time_course)
    dice = float(metrics.dice_coefficient(inf_stc, sim_stc))

    # --- evaluate structure of inferred covariances ---
    _, covs_hat = model.get_means_covariances()
    max_off_diff_hat, mean_diag_std_hat = cov_stats(covs_hat)

    print("\nRESULTS (static_full_plus_diag, oracle init, 1 run)")
    print(f"Dice: {dice:.4f}")
    print(f"Inferred cov stats: max_off_diff={max_off_diff_hat:.3g}, mean_diag_std={mean_diag_std_hat:.3g}")

    # --- test-like assertions ---
    # Intentionally loose at first to avoid flakiness; tighten once stable.
    DICE_MIN = 0.90
    MAX_OFF_DIFF_MAX = 1e-2

    assert dice >= DICE_MIN, f"Dice too low: {dice:.4f} < {DICE_MIN}"
    assert max_off_diff_hat <= MAX_OFF_DIFF_MAX, (
        f"Off-diagonals differ across states too much: {max_off_diff_hat:.3g} > {MAX_OFF_DIFF_MAX}"
    )

    print("\nPASS ✅")
