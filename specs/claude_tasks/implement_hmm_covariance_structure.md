# Claude Task Brief: Implement HMM covariance_structure (minimal patch)

## Goal
Implement new config option `covariance_structure` for osl_dynamics.models.hmm HMM observation covariances,
without introducing a new model file or unrelated refactors.

Target is to support:

- "state_full" (existing): Σ_k free full per-state
- "state_diag" (existing): Σ_k diagonal per-state
- "static_full_plus_diag" (A): Σ_k = C + D_k (shared full C + state-specific diagonal D_k)
- "scaled_shared_cov" (B): Σ_k = A_k C A_k (shared full C + state-specific per-channel gains A_k)

## Non-negotiables
- Diff-first. Minimal changes. No refactors.
- Do not change smoke tests except to fix genuine bugs.
- All shell commands require approval.
- Keep backwards compatibility with `diagonal_covariances`.
- Make `model.set_covariances(...)` work for new structures (esp. oracle init in smoke test).
- Ensure `get_covariances()` returns Σ_k for all structures.

## Acceptance criteria (must pass locally)
Run from repo root:
- `python dev/smoke_tests/means_only_dice_smoke.py`
- `python dev/smoke_tests/static_full_plus_diag_oracle_smoke.py`
(Additional tests for scaled_shared_cov will be added later.)

## Notes / existing WIP
Prior WIP exists on the cluster machine and will be added here as reference patches under:
`specs/cluster_wip_patches/`

Use those patches as **reference only** (audit/borrow logic), but do **not** apply them blindly.
The current local `osl_dynamics/models/hmm.py` may not yet include any of that WIP.


## Design guidance (keep it simple)
- Shared C should be parameterised PD (Cholesky via existing CovarianceMatricesLayer).
- D_k diagonal (and A_k gains) should be positive via existing DiagonalMatricesLayer or a minimal equivalent.
- Avoid dtype mismatches between covariance construction, log-likelihood, and Baum–Welch layers.
- Ensure layer names are stable for checkpoints: e.g. `covs_shared`, `covs_diag`, etc.

## Checklist before finalizing
- Config validation and precedence rules (covariance_structure vs diagonal_covariances)
- build_model Σ_k construction shapes = (K, D, D)
- set_covariances: can accept (K,D,D) and correctly init the underlying parameter layers for new structures
- get_covariances / get_means_covariances returns correct Σ_k
- dual_estimation doesn’t crash (even if it ignores decomposition)
- docs/specs updated if any public API changes
