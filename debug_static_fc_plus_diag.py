"""
Minimal sanity test for HMM covariance_structure='static_full_plus_diag'.

Checks:
1) Training runs end-to-end.
2) Inferred covariances have identical off-diagonals across states.
"""

print("Importing packages")
import os
import numpy as np

from osl_dynamics.simulation import HMM_MVN
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.inference import modes, metrics

# -----------------------
# Settings
# -----------------------
results_dir = "results_static_fc_plus_diag"
os.makedirs(results_dir, exist_ok=True)

np.random.seed(0)

K = 5
P = 11
n_samples = 25600

# -----------------------
# Build a shared covariance + state-specific diagonal boosts
# -----------------------
print("Constructing ground-truth covariances Σ_k = C + D_k")

# Make a random SPD shared covariance C_shared
A = np.random.randn(P, P)
C_shared = A @ A.T
C_shared = C_shared / np.mean(np.diag(C_shared))  # normalize scale a bit

# Make state-specific diagonal boosts (positive)
# Ensure boosts are not tiny so states are distinguishable
D_diag = np.exp(np.random.randn(K, P) * 0.3)  # lognormal-ish
# Optionally spread them a bit more
D_diag = D_diag * np.linspace(0.8, 1.5, K)[:, None]

# Combine: Σ_k = C_shared + diag(D_k)
covs = np.zeros((K, P, P))
for k in range(K):
    covs[k] = C_shared + np.diag(D_diag[k])

# Means: zero
means = np.zeros((K, P), dtype=np.float32)

# -----------------------
# Simulate data with HMM_MVN using our custom covs
# -----------------------
print("Simulating data")
sim = HMM_MVN(
    n_samples=n_samples,
    n_states=K,
    n_channels=P,
    trans_prob="sequence",
    stay_prob=0.9,
    means=means,
    covariances=covs,
)

data = Data(sim.time_series)
data.standardize()

# -----------------------
# Build model
# -----------------------
print("Building model")
config = Config(
    n_states=K,
    n_channels=P,
    sequence_length=200,
    learn_means=False,
    learn_covariances=True,
    covariance_structure="static_full_plus_diag",
    batch_size=16,
    learning_rate=0.01,
    n_epochs=15,
)

model = Model(config)
model.summary()

# -----------------------
# Train
# -----------------------
print("Training model")
model.train(data)

# -----------------------
# Check inferred covariances
# -----------------------
print("Checking inferred covariances")
inf_means, inf_covs = model.get_means_covariances()  # inf_covs should be (K,P,P)

# Off-diagonal identity check
off = inf_covs.copy()
idx = np.arange(P)
off[:, idx, idx] = 0.0

max_offdiag_diff = np.max(np.abs(off - off[0:1]))
print("Max |offdiag_k - offdiag_0|:", max_offdiag_diff)

# Diagonal variability check (should differ across states)
diag = np.diagonal(inf_covs, axis1=1, axis2=2)
mean_diag_std = diag.std(axis=0).mean()
print("Mean std of diagonal across states:", mean_diag_std)

# -----------------------
# Optional: dice coefficient (not the main pass/fail)
# -----------------------
print("Computing dice coefficient (optional)")
alp = model.get_alpha(data)
stc = modes.argmax_time_courses(alp)
inf_stc, sim_stc = modes.match_modes(stc, sim.state_time_course)
dice = metrics.dice_coefficient(inf_stc, sim_stc)
print("Dice coefficient:", dice)

# -----------------------
# Simple pass/fail thresholds
# -----------------------
# Offdiagonal should be essentially identical (allow tiny float tolerance)
if max_offdiag_diff < 1e-5:
    print("PASS: off-diagonals identical across states (within tolerance).")
else:
    print("FAIL: off-diagonals differ across states more than expected.")
