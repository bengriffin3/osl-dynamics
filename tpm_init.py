# tpm_init.py
import numpy as np

def sticky_uniform_tpm(n_states: int, p_stay: float) -> np.ndarray:
    """Diagonal = p_stay, off-diagonals uniform."""
    tpm = np.full((n_states, n_states), (1.0 - p_stay) / (n_states - 1))
    np.fill_diagonal(tpm, p_stay)
    return tpm