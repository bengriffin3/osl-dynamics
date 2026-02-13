"""
Oracle-init unit test for the NEW covariance_structure='static_full_plus_diag'.

We simulate from: Σ_k = C + D_k (means=0)
Then we fit the HMM with covariance_structure='static_full_plus_diag'.

Key points:
- We standardize the data.
- We transform true covariances into standardized space.
- We call model.set_covariances(covs_init) ONCE (oracle-ish init).
- Run ONCE (fast).

Expected for the NEW model:
- Dice should be high (close to 1) under oracle init.
- Inferred covs should have ~identical off-diagonals across states
  (max_off_diff should be ~0, up to tiny numerical noise).
"""

import os
import numpy as np

from osl_dynamics.simulation import HMM_MVN
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.inference import modes, metrics


def make_pd_cov(rng, n_channels, jitter=1e-3):
    A = rng.standard_normal((n_channels, n_channels))
    C = A @ A.T
    C = C / np.mean(np.diag(C))
    C = C + jitter * np.eye(n_channels)
    return C


def simulate_C_plus_D(
    n_samples=25600,
    n_states=5,
    n_channels=11,
    stay_prob=0.9,
    seed=1,
    c_scale=0.2,
    channels_per_state=6,
    diag_bump=4.0,
):
    rng = np.random.default_rng(seed)

    C = c_scale * make_pd_cov(rng, n_channels)

    D = np.ones((n_states, n_channels))
    for k in range(n_states):
        start = (k * channels_per_state) % n_channels
        idx = [(start + j) % n_channels for j in range(channels_per_state)]
        D[k, idx] += diag_bump

    covs = np.repeat(C[None, :, :], n_states, axis=0) + np.array(
        [np.diag(D[k]) for k in range(n_states)]
    )

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


def covs_to_standardized_space(covs, X):
    std = X.std(axis=0, ddof=0)
    S_inv = np.diag(1.0 / std)
    return np.einsum("ij,kjl,lm->kim", S_inv, covs, S_inv)


def cov_stats(covs):
    K, P, _ = covs.shape
    mask = np.ones((P, P)) - np.eye(P)
    off = covs * mask[None, :, :]
    max_off_diff = float(np.max(np.abs(off - off[0:1])))
    diag = np.diagonal(covs, axis1=1, axis2=2)
    mean_diag_std = float(np.mean(np.std(diag, axis=0)))
    return max_off_diff, mean_diag_std


if __name__ == "__main__":
    os.makedirs("results_static_full_plus_diag_oracle", exist_ok=True)

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
    X = sim.time_series
    data = Data(X)
    data.standardize()

    # --- true covariances in standardized space (what the model sees) ---
    covs_true_std = covs_to_standardized_space(covs_true_raw, X)
    print(
        "TRUE cov stats (standardized space):",
        "max_off_diff=", cov_stats(covs_true_std)[0],
        "mean_diag_std=", cov_stats(covs_true_std)[1],
    )

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

    # --- oracle-ish init of FULL covariances (setter will decompose into shared+diag) ---
    rng = np.random.default_rng(0)
    # covs_init = covs_true_std + 1e-3 * rng.standard_normal(covs_true_std.shape)
    covs_init = covs_true_std + 1e-4 * rng.standard_normal(covs_true_std.shape)
    covs_init = 0.5 * (covs_init + np.transpose(covs_init, (0, 2, 1)))
    covs_init += 1e-6 * np.eye(n_channels)[None, :, :]

    model.set_covariances(covs_init)
    print("Initialised covariances near TRUE (standardized space).")

    # --- fit ---
    model.fit(data)

    # --- evaluate ---
    alp = model.get_alpha(data)
    stc = modes.argmax_time_courses(alp)
    inf_stc, sim_stc = modes.match_modes(stc, sim.state_time_course)
    dice = float(metrics.dice_coefficient(inf_stc, sim_stc))

    _, covs_hat = model.get_means_covariances()
    max_off_diff_hat, mean_diag_std_hat = cov_stats(covs_hat)

    print("\nRESULTS (static_full_plus_diag, oracle init, 1 run)")
    print(f"Dice: {dice:.4f}")
    print(f"Inferred cov stats: max_off_diff={max_off_diff_hat:.3g}, mean_diag_std={mean_diag_std_hat:.3g}")

