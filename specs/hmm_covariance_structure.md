# HMM Covariance Structure Option

## Goal
Add a config field `covariance_structure` to parameterise state covariances Σ_k in the HMM observation model
with minimal changes to the existing HMM implementation.

This option selects how covariances are built inside `Model.build_model` and then passed into the existing
log-likelihood and inference layers.

## Public API
### New config field
- **Name:** `covariance_structure`
- **Type:** string
- **Default:** `"state_full"` (matches existing behaviour)

### Allowed values
- `"state_full"`: existing full per-state covariance Σ_k
- `"state_diag"`: existing diagonal per-state covariance Σ_k
- `"static_full_plus_diag"`: (A) shared full + state-specific diagonal add-on
- `"scaled_shared_cov"`: (B) shared full scaled by state-specific per-channel gains

### Mapping / definitions
Let:
- K = number of states
- D = number of channels
- C ∈ R^{D×D}: shared full covariance, positive definite (PD)
- D_k ∈ R^{D×D}: state-specific diagonal matrix with positive diagonal entries
- A_k ∈ R^{D×D}: state-specific diagonal gain matrix with positive diagonal entries

#### (A) static_full_plus_diag
- **Σ_k = C + D_k**
- C is shared across k.
- D_k is diagonal with strictly positive diagonal entries.
- Off-diagonals are identical across states; only variances differ.

#### (B) scaled_shared_cov
- **Σ_k = A_k C A_k**
- C is shared across k.
- A_k is diagonal with strictly positive diagonal entries.
- Off-diagonals vary only through per-channel scaling:
  Σ_k(i,j) = a_{k,i} a_{k,j} C(i,j)

## Parameterisations and PD guarantees
### Shared full covariance C
- Parameterise via a Cholesky factor L (lower triangular) with positive diagonal.
- C = L L^T
- Ensure positivity of diag(L) via `softplus` (or equivalent safe transform) + small epsilon.

### State-specific diagonal D_k (A)
- D_k = diag( softplus(d_k) + eps )
- Guarantees D_k is PD diagonal; C + D_k remains PD if C is PD and D_k is positive diagonal.

### State-specific gains A_k (B)
- A_k = diag( softplus(g_k) + eps )
- If C is PD and A_k has strictly positive diagonal, then A_k C A_k is PD.

### Numerical stability
- Use a single small epsilon (e.g. 1e-6) added to diagonal transforms.
- Keep dtype consistent (float32 unless the model is explicitly float64).

## Backwards compatibility
### Existing option: `diagonal_covariances`
- If `covariance_structure` is not provided:
  - if `diagonal_covariances=True` -> behave as `"state_diag"`
  - else -> behave as `"state_full"`
- If `covariance_structure` is provided, it takes precedence.
- Deprecation plan (optional): warn when `diagonal_covariances` is used without `covariance_structure`.

### Interaction with learn flags
- `learn_covariances=False` should freeze all covariance parameters relevant to the chosen structure:
  - state_full/state_diag: state covariance parameters
  - static_full_plus_diag: both C and D_k parameters
  - scaled_shared_cov: both C and A_k parameters
- `learn_means` unaffected.

## Expected tensor shapes (key points)
Assume batch dimension B and time T in data flow.
- Covariances passed to log-likelihood should be shaped as:
  - **(K, D, D)** for per-state covariances
- For shared C:
  - C is **(D, D)** and must be expanded/tiled to **(K, D, D)** when combining with state-specific terms.
- For D_k diagonal:
  - store unconstrained params as **(K, D)** then build diag matrices **(K, D, D)**
- For gains:
  - store unconstrained gains as **(K, D)** then build A_k as diag matrices or use broadcasted multiplication.

## Implementation location
- Σ_k must be built in **`Model.build_model`** (existing HMM class) and then fed into the existing
  log-likelihood + inference layers, without creating a new model file.
