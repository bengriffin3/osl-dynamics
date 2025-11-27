"""Wrapper functions for use in the config API.

All of the functions in this module can be listed in the config passed to
:code:`osl_dynamics.run_pipeline`.

All wrapper functions have the structure::

    func(data, output_dir, **kwargs)

where:

- :code:`data` is an :code:`osl_dynamics.data.Data` object.
- :code:`output_dir` is the path to save output to.
- :code:`kwargs` are keyword arguments for function specific options.
"""

import os
import warnings
import json
import pickle
import logging
from pathlib import Path

import numpy as np

from osl_dynamics import array_ops
from osl_dynamics.utils.misc import load, override_dict_defaults, save
from osl_dynamics.utils.plotting import plot_line

_logger = logging.getLogger("osl-dynamics")


# def load_data(inputs, kwargs=None, prepare=None):
#     """Load and prepare data.

#     Parameters
#     ----------
#     inputs : str
#         Path to directory containing :code:`npy` files.
#     kwargs : dict, optional
#         Keyword arguments to pass to the `Data <https://osl-dynamics\
#         .readthedocs.io/en/latest/autoapi/osl_dynamics/data/index.html\
#         #osl_dynamics.data.Data>`_ class. Useful keyword arguments to pass are
#         :code:`sampling_frequency`, :code:`mask_file` and
#         :code:`parcellation_file`.
#     prepare : dict, optional
#         Methods dict to pass to the prepare method. See docstring for
#         `Data <https://osl-dynamics.readthedocs.io/en/latest/autoapi\
#         /osl_dynamics/data/index.html#osl_dynamics.data.Data>`_.prepare.

#     Returns
#     -------
#     data : osl_dynamics.data.Data
#         Data object.
#     """
#     from osl_dynamics.data import Data

#     kwargs = {} if kwargs is None else kwargs
#     prepare = {} if prepare is None else prepare

#     data = Data(inputs, **kwargs)
#     data.prepare(prepare)
#     return data

def load_data(inputs, kwargs=None, prepare=None):
    from osl_dynamics.data import Data

    kwargs = {} if kwargs is None else kwargs
    prepare = {} if prepare is None else prepare

    data = Data(inputs, **kwargs)
    data.prepare(prepare)

    # --- NEW: remember exactly which channels were selected in this fold/run ---
    try:
        sel = prepare.get("select", {}).get("channels", None)
        if sel is not None:
            # store as a numpy int array for downstream use
            import numpy as np
            data.selected_channels = np.array(sel, dtype=int)
            _logger.info(f"Recorded selected channels on Data: len={len(data.selected_channels)}")
    except Exception as e:
        _logger.warning(f"Could not record selected channels on Data: {e}")
    # ---------------------------------------------------------------------------

    return data



# def build_hmm(
#         data,
#         output_dir,
#         config_kwargs,
# ):
#     """Build up a `Hidden Markov Model <https://osl-dynamics.readthedocs.io/en\
#        /latest/autoapi/osl_dynamics/models/hmm/index.html>`_.

#        This function will:

#        1. Build an :code:`hmm.Model` object.
#        2. Save the model in :code:`<output_dir>/model`

#        This function will create two directories:

#        - :code:`<output_dir>/model`, which contains the model.

#        Parameters
#        ----------
#        data : osl_dynamics.data.Data
#            Data object. Serves as a place holder.
#        output_dir : str
#            Path to output directory.
#        config_kwargs : dict
#            Keyword arguments to pass to `hmm.Config <https://osl-dynamics\
#            .readthedocs.io/en/latest/autoapi/osl_dynamics/models/hmm/index.html\
#            #osl_dynamics.models.hmm.Config>`_. Defaults to::

#                {'sequence_length': 2000,
#                 'batch_size': 32,
#                 'learning_rate': 0.01,
#                 'n_epochs': 20}.
#     """

#     from osl_dynamics.models import hmm
#     # Directories
#     model_dir = output_dir + "/model"

#     # Create the model object
#     _logger.info("Building model")
#     default_config_kwargs = {
#         "n_channels": data.n_channels,
#         "sequence_length": 2000,
#         "batch_size": 32,
#         "learning_rate": 0.01,
#         "n_epochs": 20,
#     }
#     config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
#     _logger.info(f"Using config_kwargs: {config_kwargs}")

#     ######################################################################################

#     # --- Inject per-state initial covariances using the exact channels selected in load_data ---
#     init_cov = config_kwargs.get("initial_covariances", None)
#     K = int(config_kwargs.get("n_states", 0))
#     C = int(config_kwargs.get("n_channels", data.n_channels))
#     diag = bool(config_kwargs.get("diagonal_covariances", False))

#     if isinstance(init_cov, str):
#         arr = np.load(init_cov)

#         if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
#             # Case A: full FC (D,D) -> slice to (C,C) using selected channels, then tile
#             if not hasattr(data, "selected_channels"):
#                 raise RuntimeError("[*] Data.selected_channels not found; needed to slice full FC.")
#             chan_idx = np.array(data.selected_channels, dtype=int).ravel()
#             if chan_idx.size != C:
#                 raise RuntimeError(f"[*] selected_channels has {chan_idx.size} entries but n_channels={C}.")

#             R = arr[np.ix_(chan_idx, chan_idx)]
#             R = (R + R.T) / 2.0
#             np.fill_diagonal(R, 1.0)
#             R = R + np.eye(C) * 1e-6

#             if diag:
#                 variances = np.diag(R).astype(np.float32)
#                 init_cov_arr = np.tile(variances[None, :], (K, 1))            # (K, C)
#             else:
#                 init_cov_arr = np.tile(R[None, :, :], (K, 1, 1)).astype(np.float32)  # (K, C, C)

#             config_kwargs["initial_covariances"] = init_cov_arr

#         elif arr.ndim == 3 and arr.shape == (K, C, C):
#             # Case B: already (K,C,C) -> use as is
#             config_kwargs["initial_covariances"] = arr.astype(np.float32, copy=False)

#         elif arr.ndim == 2 and arr.shape == (K, C) and diag:
#             # Case C: already (K,C) variances for diagonal model
#             config_kwargs["initial_covariances"] = arr.astype(np.float32, copy=False)

#         else:
#             raise ValueError(
#                 f"initial_covariances at {init_cov} has unexpected shape {arr.shape} "
#                 f"for K={K}, C={C}, diagonal={diag}."
#             )

#     elif isinstance(init_cov, np.ndarray):
#         # Validate ndarray provided directly
#         expected = (K, C) if diag else (K, C, C)
#         if init_cov.shape != expected:
#             raise ValueError(f"initial_covariances array has shape {init_cov.shape} but expected {expected}")
#         config_kwargs["initial_covariances"] = init_cov.astype(np.float32, copy=False)

#     # --- end injection ---

#     ######################################################################################


#     config = hmm.Config(**config_kwargs)
#     model = hmm.Model(config)
#     # Save trained model
#     _logger.info(f"Saving model to: {model_dir}")
#     model.save(model_dir)

# def build_hmm(
#         data,
#         output_dir,
#         config_kwargs,
# ):
#     """Build up an OSL-Dynamics HMM Model and save it to <output_dir>/model.

#     If config_kwargs["initial_covariances"] == "static_fc":
#       compute static covariance from prepared data on-the-fly,
#       replicate across K states (diagonal or full), inject, then build & save.
#     """
#     from osl_dynamics.models import hmm
#     from osl_dynamics.array_ops import estimate_gaussian_distribution

#     model_dir = output_dir + "/model"

#     _logger.info("Building model")
#     default_config_kwargs = {
#         "n_channels": data.n_channels,
#         "sequence_length": 2000,
#         "batch_size": 32,
#         "learning_rate": 0.01,
#         "n_epochs": 20,
#     }
#     config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
#     _logger.info(f"Using config_kwargs: {config_kwargs}")

#     # --- initial_covariances injection ---
#     init_cov = config_kwargs.get("initial_covariances", None)
#     K = int(config_kwargs.get("n_states", 0))
#     C_cfg = int(config_kwargs.get("n_channels", data.n_channels))
#     diag = bool(config_kwargs.get("diagonal_covariances", False))

#     if isinstance(init_cov, str) and init_cov == "static_fc":
#         _logger.info("[static_fc] Computing static covariance on-the-fly from prepared data.")
#         ts = data.time_series(prepared=True, concatenate=False)
#         ts = [ts[i] for i in getattr(data, "keep", range(len(ts)))]
#         ts = np.concatenate(ts, axis=0)  # (T_total, C_actual)
#         _logger.info(f"[dbg] static_fc ts shape={ts.shape}, dtype={ts.dtype}")

#         means_s, covs_s = estimate_gaussian_distribution(
#             ts, nonzero_means=config_kwargs.get("learn_means", False)
#         )
#         covs_s = np.asarray(covs_s)
#         # Handle batched (1,C,C) output
#         if covs_s.ndim == 3 and covs_s.shape[0] == 1:
#             covs_s = covs_s[0]
#         elif covs_s.ndim != 2:
#             raise ValueError(f"[static_fc] Unexpected covs_s shape: {covs_s.shape}")

#         C_actual = covs_s.shape[0]
#         if C_actual != C_cfg:
#             _logger.warning(f"[static_fc] n_channels mismatch: covs={C_actual} vs config={C_cfg}. "
#                             f"Using C_actual={C_actual}.")
#             config_kwargs["n_channels"] = C_actual  # keep model consistent

#         # symmetrize + jitter
#         covs_s = (covs_s + covs_s.T) / 2.0
#         covs_s = covs_s.astype(np.float32, copy=False)
#         covs_s += np.eye(C_actual, dtype=np.float32) * 1e-6

#         if diag:
#             variances = np.diag(covs_s).astype(np.float32)           # (C_actual,)
#             init_cov_arr = np.repeat(variances[None, :], max(K, 1), axis=0)   # (K, C_actual)
#         else:
#             init_cov_arr = np.repeat(covs_s[None, :, :], max(K, 1), axis=0)   # (K, C_actual, C_actual)

#         config_kwargs["initial_covariances"] = init_cov_arr
#         _logger.info(f"[static_fc] Injected initial_covariances shape={init_cov_arr.shape} "
#                      f"(expected {(K, C_actual) if diag else (K, C_actual, C_actual)})")

#     elif isinstance(init_cov, np.ndarray):
#         expected = (K, C_cfg) if diag else (K, C_cfg, C_cfg)
#         if init_cov.shape != expected:
#             raise ValueError(f"initial_covariances array has shape {init_cov.shape} but expected {expected}")
#         config_kwargs["initial_covariances"] = init_cov.astype(np.float32, copy=False)

#     # --- build & save ---
#     config = hmm.Config(**config_kwargs)
#     model = hmm.Model(config)
#     _logger.info(f"Saving model to: {model_dir}")
#     model.save(model_dir)


# def build_hmm(
#         data,
#         output_dir,
#         config_kwargs,
# ):
#     """
#     Build an OSL-Dynamics HMM Model and save it to <output_dir>/model.

#     Behaviour:
#       1. If n_states == 1:
#            → Compute and save static Gaussian (means & covs) to inf_params.
#              Also saves static_cov.npy under debug/.
#       2. If n_states > 1 and initial_covariances == "static_fc":
#            → Compute static FC (same estimator as Mode 1).
#            → Use it as the *initializer passed into the model*:
#                 - diagonal_covariances=True  -> K×C variances from diag(static FC)
#                 - otherwise (full)           -> K×C×C static FC replicated across states
#            → Build and save the HMM model with that initializer.
#       3. If n_states > 1 and no 'initial_covariances':
#            → Build and save the HMM model directly.
#     """
#     from osl_dynamics.models import hmm
#     from osl_dynamics.array_ops import estimate_gaussian_distribution

#     # -------------------------------------------------------------------------
#     # Setup
#     # -------------------------------------------------------------------------
#     model_dir = os.path.join(output_dir, "model")
#     debug_dir = os.path.join(output_dir, "debug")
#     inf_params_dir = os.path.join(output_dir, "inf_params")
#     os.makedirs(debug_dir, exist_ok=True)
#     os.makedirs(inf_params_dir, exist_ok=True)

#     _logger.info("Building HMM configuration.")
#     default_config_kwargs = {
#         "n_channels": data.n_channels,
#         "sequence_length": 2000,
#         "batch_size": 32,
#         "learning_rate": 0.01,
#         "n_epochs": 20,
#     }
#     config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)

#     K = int(config_kwargs.get("n_states", 0))
#     C_cfg = int(config_kwargs.get("n_channels", data.n_channels))
#     diag = bool(config_kwargs.get("diagonal_covariances", False))
#     init_cov_mode = config_kwargs.get("initial_covariances", None)

#     _logger.info(f"[dbg] Config: K={K}, C={C_cfg}, diag={diag}, init_cov={init_cov_mode!r}")

#     # -------------------------------------------------------------------------
#     # Case 1: n_states == 1 → compute static FC and save means/covs
#     # -------------------------------------------------------------------------
#     if K == 1:
#         _logger.info("[mode 1] n_states=1 → computing static means/covs (no HMM model).")

#         ts = data.time_series(prepared=True, concatenate=False)
#         ts = [ts[i] for i in getattr(data, "keep", range(len(ts)))]
#         ts = np.concatenate(ts, axis=0)

#         means, covs = estimate_gaussian_distribution(
#             ts, nonzero_means=bool(config_kwargs.get("learn_means", False))
#         )

#         covs = np.asarray(covs)
#         if covs.ndim == 3 and covs.shape[0] == 1:
#             covs = covs[0]
#         elif covs.ndim != 2:
#             raise ValueError(f"[static] Unexpected covs shape: {covs.shape}")

#         covs = ((covs + covs.T) / 2.0).astype(np.float32, copy=False)

#         # Save everything needed for downstream
#         save(os.path.join(inf_params_dir, "means.npy"), means)
#         save(os.path.join(inf_params_dir, "covs.npy"), covs)
#         np.save(os.path.join(debug_dir, "static_cov.npy"), covs)

#         _logger.info(f"[static] Saved means.npy (shape {np.asarray(means).shape}) "
#                      f"and covs.npy (shape {covs.shape}) to {inf_params_dir}")
#         _logger.info(f"[static] First 5×5 block:\n{covs[:5, :5]}")
#         return

#     # -------------------------------------------------------------------------
#     # Case 2: n_states > 1 and initial_covariances == "static_fc"
#     # -------------------------------------------------------------------------
#     if isinstance(init_cov_mode, str) and init_cov_mode == "static_fc":
#         _logger.info("[mode 2] Multi-state + 'static_fc' → compute static FC and USE as initializer.")

#         ts_train = data.time_series(prepared=True, concatenate=False)
#         ts_train = [ts_train[i] for i in getattr(data, "keep", range(len(ts_train)))]
#         ts_train = np.concatenate(ts_train, axis=0)

#         # Same estimator as Mode 1; means flag mirrors learn_means
#         _, cov_static = estimate_gaussian_distribution(
#             ts_train, nonzero_means=bool(config_kwargs.get("learn_means", False))
#         )
#         cov_static = np.asarray(cov_static)
#         if cov_static.ndim == 3 and cov_static.shape[0] == 1:
#             cov_static = cov_static[0]
#         elif cov_static.ndim != 2:
#             raise ValueError(f"[static] Unexpected cov_static shape: {cov_static.shape}")

#         cov_static = ((cov_static + cov_static.T) / 2.0).astype(np.float32, copy=False)
#         np.save(os.path.join(debug_dir, "static_cov.npy"), cov_static)
#         _logger.info(f"[static] Computed static FC, shape={cov_static.shape}")
#         _logger.info(f"[static] First 5×5 block:\n{cov_static[:5, :5]}")

#         # Build initializer expected by OSL (diag vs full)
#         if diag:
#             variances = np.diag(cov_static).astype(np.float32)         # (C,)
#             init_cov_arr = np.repeat(variances[None, :], K, axis=0)    # (K, C)
#         else:
#             init_cov_arr = np.repeat(cov_static[None, :, :], K, axis=0).astype(np.float32)  # (K, C, C)

#         # Replace the "static_fc" string with the actual array initializer
#         cfg = dict(config_kwargs)  # shallow copy
#         cfg["initial_covariances"] = init_cov_arr

#         _logger.info(f"[mode 2] Initializer shape set to {init_cov_arr.shape}; building & saving model.")
#         config = hmm.Config(**cfg)
#         model = hmm.Model(config)
#         model.save(model_dir)
#         _logger.info(f"[build] Model saved to {model_dir}")
#         return

#     # -------------------------------------------------------------------------
#     # Case 3: n_states > 1 and no 'static_fc' usage
#     # -------------------------------------------------------------------------
#     _logger.info("[mode 3] Standard HMM model build (no static FC used).")

#     config = hmm.Config(**config_kwargs)
#     model = hmm.Model(config)
#     model.save(model_dir)
#     _logger.info(f"[build] Model saved to {model_dir}")
#     return


def build_hmm(
        data,
        output_dir,
        config_kwargs,
):
    """
    Build an OSL-Dynamics HMM Model and save it to <output_dir>/model.

    Behaviour:
      1. n_states == 1  -> compute & save static Gaussian (means/covs).
      2. n_states > 1 and initial_covariances == "static_fc"
         -> compute static FC once and use it as initializer (diag/full).
      3. n_states > 1 and no 'initial_covariances' -> standard build.
    """
    import os
    import numpy as np
    from osl_dynamics.models import hmm
    from osl_dynamics.array_ops import estimate_gaussian_distribution

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    model_dir = os.path.join(output_dir, "model")
    debug_dir = os.path.join(output_dir, "debug")
    inf_params_dir = os.path.join(output_dir, "inf_params")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(inf_params_dir, exist_ok=True)

    _logger.info("Building HMM configuration.")
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "sequence_length": 2000,
        "batch_size": 32,
        "learning_rate": 0.01,
        "n_epochs": 20,
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)

    K = int(config_kwargs.get("n_states", 0))
    C_cfg = int(config_kwargs.get("n_channels", data.n_channels))
    diag = bool(config_kwargs.get("diagonal_covariances", False))
    init_cov_mode = config_kwargs.get("initial_covariances", None)

    _logger.info(f"[dbg] Config: K={K}, C={C_cfg}, diag={diag}, init_cov={init_cov_mode!r}")

    # -------------------------------------------------------------------------
    # Case 1: n_states == 1 → compute static FC and save means/covs
    # -------------------------------------------------------------------------
    if K == 1:
        _logger.info("[mode 1] n_states=1 → computing static means/covs (no HMM model).")

        ts = data.time_series(prepared=True, concatenate=False)
        ts = [ts[i] for i in getattr(data, "keep", range(len(ts)))]
        ts = np.concatenate(ts, axis=0)

        means, covs = estimate_gaussian_distribution(
            ts, nonzero_means=bool(config_kwargs.get("learn_means", False))
        )

        covs = np.asarray(covs)
        if covs.ndim == 3 and covs.shape[0] == 1:
            covs = covs[0]
        elif covs.ndim != 2:
            raise ValueError(f"[static] Unexpected covs shape: {covs.shape}")

        covs = ((covs + covs.T) / 2.0).astype(np.float32, copy=False)

        save(os.path.join(inf_params_dir, "means.npy"), means)
        save(os.path.join(inf_params_dir, "covs.npy"), covs)
        np.save(os.path.join(debug_dir, "static_cov.npy"), covs)

        _logger.info(f"[static] Saved means.npy (shape {np.asarray(means).shape}) "
                     f"and covs.npy (shape {covs.shape}) to {inf_params_dir}")
        _logger.info(f"[static] First 5×5 block:\n{covs[:5, :5]}")
        return

    # Small helper to normalise initial_trans_prob *before* building Config
    def _normalise_initial_trans_prob(cfg_dict: dict, where: str):
        ip = cfg_dict.get("initial_trans_prob", None)
        if isinstance(ip, list):
            ip_arr = np.array(ip, dtype=float)
            cfg_dict["initial_trans_prob"] = ip_arr
            _logger.info(f"[{where}] Converted initial_trans_prob list→ndarray shape={ip_arr.shape}")
        elif isinstance(ip, np.ndarray):
            _logger.info(f"[{where}] initial_trans_prob ndarray shape={ip.shape}")
        elif isinstance(ip, str):  # e.g., "random"
            _logger.info(f"[{where}] initial_trans_prob='{ip}'")
            return
        elif ip is None:
            _logger.info(f"[{where}] initial_trans_prob not provided (defaults apply)")
            return

        # Validate shape if ndarray was provided/created
        ip_now = cfg_dict.get("initial_trans_prob", None)
        if isinstance(ip_now, np.ndarray):
            if ip_now.ndim != 2 or ip_now.shape != (K, K):
                raise ValueError(
                    f"[{where}] initial_trans_prob must be shape (K,K)={(K,K)}, got {ip_now.shape}"
                )

    # If someone passes initial means/covs as lists, turn into arrays (paths are fine)
    def _normalise_init_params(cfg_dict: dict, where: str):
        for key in ("initial_means", "initial_covariances"):
            val = cfg_dict.get(key, None)
            if isinstance(val, list):
                cfg_dict[key] = np.array(val)
                _logger.info(f"[{where}] Converted {key} list→ndarray shape={cfg_dict[key].shape}")

    # -------------------------------------------------------------------------
    # Case 2: n_states > 1 and initial_covariances == "static_fc"
    # -------------------------------------------------------------------------
    if isinstance(init_cov_mode, str) and init_cov_mode == "static_fc":
        _logger.info("[mode 2] Multi-state + 'static_fc' → compute static FC and USE as initializer.")

        ts_train = data.time_series(prepared=True, concatenate=False)
        ts_train = [ts_train[i] for i in getattr(data, "keep", range(len(ts_train)))]
        ts_train = np.concatenate(ts_train, axis=0)

        _, cov_static = estimate_gaussian_distribution(
            ts_train, nonzero_means=bool(config_kwargs.get("learn_means", False))
        )
        cov_static = np.asarray(cov_static)
        if cov_static.ndim == 3 and cov_static.shape[0] == 1:
            cov_static = cov_static[0]
        elif cov_static.ndim != 2:
            raise ValueError(f"[static] Unexpected cov_static shape: {cov_static.shape}")

        cov_static = ((cov_static + cov_static.T) / 2.0).astype(np.float32, copy=False)
        np.save(os.path.join(debug_dir, "static_cov.npy"), cov_static)
        _logger.info(f"[static] Computed static FC, shape={cov_static.shape}")
        _logger.info(f"[static] First 5×5 block:\n{cov_static[:5, :5]}")

        # Build initializer (diag vs full)
        if diag:
            variances = np.diag(cov_static).astype(np.float32)         # (C,)
            init_cov_arr = np.repeat(variances[None, :], K, axis=0)    # (K, C)
        else:
            init_cov_arr = np.repeat(cov_static[None, :, :], K, axis=0).astype(np.float32)  # (K,C,C)

        cfg = dict(config_kwargs)  # shallow copy so we don't mutate caller state
        cfg["initial_covariances"] = init_cov_arr

        # NEW: normalise ip + any list init params before Config()
        _normalise_initial_trans_prob(cfg, where="build_hmm/mode2")
        _normalise_init_params(cfg, where="build_hmm/mode2")

        _logger.info(f"[mode 2] Initializer shape set to {init_cov_arr.shape}; building & saving model.")
        config = hmm.Config(**cfg)
        model = hmm.Model(config)
        model.save(model_dir)
        _logger.info(f"[build] Model saved to {model_dir}")
        return

    # -------------------------------------------------------------------------
    # Case 3: n_states > 1 and no 'static_fc' usage
    # -------------------------------------------------------------------------
    _logger.info("[mode 3] Standard HMM model build (no static FC used).")

    # NEW: normalise ip + any list init params before Config()
    _normalise_initial_trans_prob(config_kwargs, where="build_hmm/mode3")
    _normalise_init_params(config_kwargs, where="build_hmm/mode3")

    config = hmm.Config(**config_kwargs)
    model = hmm.Model(config)
    model.save(model_dir)
    _logger.info(f"[build] Model saved to {model_dir}")
    return


def build_dynemo(
        data,
        output_dir,
        config_kwargs,
):
    """Train `DyNeMo <https://osl-dynamics.readthedocs.io/en/latest/autoapi\
    /osl_dynamics/models/dynemo/index.html>`_.

    This function will:

    1. Build a :code:`dynemo.Model` object.
    2. Save the model in :code:`<output_dir>/model`

    This function will create two directories:

    - :code:`<output_dir>/model`, which contains the trained model.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object for training the model.
    output_dir : str
        Path to output directory.
    config_kwargs : dict
        Keyword arguments to pass to `dynemo.Config <https://osl-dynamics\
        .readthedocs.io/en/latest/autoapi/osl_dynamics/models/dynemo\
        /index.html#osl_dynamics.models.dynemo.Config>`_. Defaults to::

            {'n_channels': data.n_channels.
             'sequence_length': 200,
             'inference_n_units': 64,
             'inference_normalization': 'layer',
             'model_n_units': 64,
             'model_normalization': 'layer',
             'learn_alpha_temperature': True,
             'initial_alpha_temperature': 1.0,
             'do_kl_annealing': True,
             'kl_annealing_curve': 'tanh',
             'kl_annealing_sharpness': 10,
             'n_kl_annealing_epochs': 20,
             'batch_size': 128,
             'learning_rate': 0.01,
             'lr_decay': 0.1,
             'n_epochs': 40}
    """

    from osl_dynamics.models import dynemo
    # Directories
    model_dir = output_dir + "/model"

    # Create the model object
    _logger.info("Building model")
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "sequence_length": 200,
        "inference_n_units": 64,
        "inference_normalization": "layer",
        "model_n_units": 64,
        "model_normalization": "layer",
        "learn_alpha_temperature": True,
        "initial_alpha_temperature": 1.0,
        "do_kl_annealing": True,
        "kl_annealing_curve": "tanh",
        "kl_annealing_sharpness": 10,
        "n_kl_annealing_epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.01,
        "lr_decay": 0.1,
        "n_epochs": 40,
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
    _logger.info(f"Using config_kwargs: {config_kwargs}")
    config = dynemo.Config(**config_kwargs)
    model = dynemo.Model(config)
    # Save trained model
    _logger.info(f"Saving model to: {model_dir}")
    model.save(model_dir)

def build_swc(
        data,
        output_dir,
        config_kwargs
):
    """
    Build up a Sliding Window Correlation Model.
    """
    from osl_dynamics.models import swc

    # Create the model object
    _logger.info("Building model")
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "window_length": 100,
        "window_offset": 75,
        "window_type": 'rectangular',
        'learn_means': False,
        'learn_covariances': True
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
    _logger.info(f"Using config_kwargs: {config_kwargs}")
    config = swc.Config(**config_kwargs)
    model = swc.Model(config)
    # Save trained model
    model_dir = f'{output_dir}/model/'
    _logger.info(f"Saving model to: {model_dir}")
    model.save(model_dir)
def train_swc(
        data,
        output_dir,
        config_kwargs,
        init_kwargs=None,
        fit_kwargs=None,
):
    """
    Fit a Sliding Window Correlation Model.
    """

    if data is None:
        raise ValueError("data must be passed.")

    init_kwargs = {} if init_kwargs is None else init_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

    from osl_dynamics.models import swc

    # Create the model object
    _logger.info("Building model")
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "window_length": 100,
        "window_offset": 75,
        "window_type": 'rectangular',
        'learn_means': False,
        'learn_covariances': True
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
    _logger.info(f"Using config_kwargs: {config_kwargs}")

    # Deal with the special case of static FC model (n_state = 1 )
    if config_kwargs['n_states'] == 1:
        ts = data.time_series(prepared=True, concatenate=False)
        # Note training_data.keep is in order. You need to preserve the order
        # between data and alpha.
        ts = [ts[i] for i in data.keep]
        # Concatenate across all sessions
        ts = np.concatenate(ts, axis=0)

        from osl_dynamics.array_ops import estimate_gaussian_distribution
        means, covs = estimate_gaussian_distribution(ts, nonzero_means=config_kwargs['learn_means'])

        inf_params_dir = output_dir + "/inf_params"
        os.makedirs(inf_params_dir, exist_ok=True)

        save(f"{inf_params_dir}/means.npy", means)
        save(f"{inf_params_dir}/covs.npy", covs)
        return



    config = swc.Config(**config_kwargs)
    model = swc.Model(config)
    # model.summary()

    # Training
    covs, alpha = model.fit(data, **fit_kwargs)
    inf_params_dir = f'{output_dir}/inf_params/'
    if not os.path.exists(inf_params_dir):
        os.makedirs(inf_params_dir, exist_ok=True)
    # Save trained model
    _logger.info(f"Saving Sliding Window Correlation results to: {inf_params_dir}")

    # Get the inferred parameters
    means = np.zeros((config_kwargs['n_states'], config_kwargs['n_channels']))

    # Save inferred parameters
    save(f"{inf_params_dir}/alp.pkl", alpha)
    save(f"{inf_params_dir}/means.npy", means)
    save(f"{inf_params_dir}/covs.npy", covs)

    model.save(f'{output_dir}/model/')


def train_swc_spatial(
        data,
        output_dir,
        config_kwargs,
        init_kwargs=None,
        fit_kwargs=None,
):
    """
    Fit a Sliding Window Correlation Model, inferring the spatial statistics
    while keeping the temporal labels fixed f'{output_dir}/inf_params/alp.pkl'
    The output means and covs should be saved in
    f'{output_dir}/dual_estimates/means.npy', f'{output_dir}/dual_estimates/covs.npy'
    """

    if data is None:
        raise ValueError("data must be passed.")

    init_kwargs = {} if init_kwargs is None else init_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

    from osl_dynamics.models import swc

    # Create the model object
    _logger.info("Building model")
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "window_length": 100,
        "window_offset": 75,
        "window_type": 'rectangular',
        'learn_means': False,
        'learn_covariances': True
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
    _logger.info(f"Using config_kwargs: {config_kwargs}")

    config = swc.Config(**config_kwargs)
    model = swc.Model(config)

    # Load alpha
    with open(f'{output_dir}/inf_params/alp.pkl', 'rb') as file:
        alpha = pickle.load(file)
    # Training
    covs = model.infer_spatial(data, alpha, **fit_kwargs)
    dual_estimates_dir = f'{output_dir}/dual_estimates/'
    if not os.path.exists(dual_estimates_dir):
        os.makedirs(dual_estimates_dir)
    _logger.info(f"Saving Sliding Window Correlation infer_spatial results to: {dual_estimates_dir}")

    # Get the inferred parameters
    means = np.zeros((config_kwargs['n_states'], config_kwargs['n_channels']))

    # Save inferred parameters

    save(f"{dual_estimates_dir}/means.npy", means)
    save(f"{dual_estimates_dir}/covs.npy", covs)


def train_swc_temporal(
        data,
        output_dir,
        config_kwargs,
        init_kwargs=None,
        fit_kwargs=None,
):
    """
    Fit a Sliding Window Correlation Model, inferring the temporal statistics
    while keeping the spatial statistics fixed
    f'{output_dir}/inf_params/means.npy' f'{output_dir}/inf_params/covs.npy'
    The output alpha should be saved in f'{output_dir}/inf_params/alp.pkl'
    """

    if data is None:
        raise ValueError("data must be passed.")

    init_kwargs = {} if init_kwargs is None else init_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

    from osl_dynamics.models import swc

    # Create the model object
    _logger.info("Building model")
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "window_length": 100,
        "window_offset": 75,
        "window_type": 'rectangular',
        'learn_means': False,
        'learn_covariances': True
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
    _logger.info(f"Using config_kwargs: {config_kwargs}")

    config = swc.Config(**config_kwargs)
    model = swc.Model(config)

    inf_params_dir = f'{output_dir}/inf_params/'
    means = np.load(f'{inf_params_dir}/means.npy')
    covs = np.load(f'{inf_params_dir}/covs.npy')
    # Training
    alpha = model.infer_temporal(data, means, covs, **fit_kwargs)
    _logger.info(f"Saving Sliding Window Correlation infer_temporal results to: {inf_params_dir}")

    # Save inferred parameters
    with open(f'{inf_params_dir}/alp.pkl', 'wb') as file:
        pickle.dump(alpha, file)
    return f'{inf_params_dir}/alp.pkl'

'''
def train_swc_log_likelihood(
        data,
        output_dir,
        config_kwargs,
        init_kwargs=None,
        fit_kwargs=None,
):
    """
    Fit a Sliding Window Correlation Model, calculating the average log likelihood
    of the window while keeping the temporal and spatial statistics fixed
    The input should be found in
    f'{output_dir}/inf_params/alp.pkl' &
    f'{output_dir}/inf_params/means.npy' f'{output_dir}/inf_params/covs.npy',
    The output should be a json file found in f'{output_dir}/metrics.json'

    """
    if data is None:
        raise ValueError("data must be passed.")

    init_kwargs = {} if init_kwargs is None else init_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

    from osl_dynamics.models import swc

    # Create the model object
    _logger.info("Building model")
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "window_length": 100,
        "window_offset": 75,
        "window_type": 'rectangular',
        'learn_means': False,
        'learn_covariances': True
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
    _logger.info(f"Using config_kwargs: {config_kwargs}")

    config = swc.Config(**config_kwargs)
    model = swc.Model(config)

    inf_params_dir = f'{output_dir}/inf_params/'
    # Read all the parameters
    means = np.load(f'{inf_params_dir}/means.npy')
    covs = np.load(f'{inf_params_dir}/covs.npy')
    with open(f'{inf_params_dir}/alp.pkl', 'rb') as file:
        alpha = pickle.load(file)
    # Note: the metrics should be the average log likelihood of the window
    metrics = model.log_likelihood(data, alpha, means, covs, **fit_kwargs) * (data.n_samples/data.n_sessions)
    _logger.info(f"Saving Sliding Window Correlation log_likelihood results to: {output_dir}/metrics.json")

    # Save inferred parameters
    with open(f'{output_dir}/metrics.json', 'w') as file:
        json.dump({'log_likelihood':float(metrics)}, file)
    return f'{output_dir}/metrics.json'
'''

# def train_hmm(
#         data,
#         output_dir,
#         config_kwargs,
#         init_kwargs=None,
#         infer_spatial='sample',
#         fit_kwargs=None,
#         save_inf_params=True,
#         calculate_free_energy=True
# ):
#     """Train a `Hidden Markov Model <https://osl-dynamics.readthedocs.io/en\
#     /latest/autoapi/osl_dynamics/models/hmm/index.html>`_.

#     This function will:

#     1. Build an :code:`hmm.Model` object.
#     2. Initialize the parameters of the model using
#        :code:`Model.random_state_time_course_initialization`.
#     3. Perform full training.
#     4. Save the inferred parameters (state probabilities, means and covariances)
#        if :code:`save_inf_params=True`.

#     This function will create two directories:

#     - :code:`<output_dir>/model`, which contains the trained model.
#     - :code:`<output_dir>/inf_params`, which contains the inferred parameters.
#       This directory is only created if :code:`save_inf_params=True`.
#     - :code:`<output_dir>/metrics`, which contains the free energy on the training data.
#       This directory is only created if :code:`calculate_free_energy=True`.

#     Parameters
#     ----------
#     data : osl_dynamics.data.Data
#         Data object for training the model.
#     output_dir : str
#         Path to output directory.
#     config_kwargs : dict
#         Keyword arguments to pass to `hmm.Config <https://osl-dynamics\
#         .readthedocs.io/en/latest/autoapi/osl_dynamics/models/hmm/index.html\
#         #osl_dynamics.models.hmm.Config>`_. Defaults to::

#             {'sequence_length': 2000,
#              'batch_size': 32,
#              'learning_rate': 0.01,
#              'n_epochs': 20}.
#     init_kwargs : dict, optional
#         Keyword arguments to pass to
#         :code:`Model.random_state_time_course_initialization`. Defaults to::

#             {'n_init': 3, 'n_epochs': 1}.
#     fit_kwargs : dict, optional
#         Keyword arguments to pass to the :code:`Model.fit`. No defaults.
#     save_inf_params : bool, optional
#         Should we save the inferred parameters?
#     """
#     if data is None:
#         raise ValueError("data must be passed.")

#     from osl_dynamics.models import hmm

#     init_kwargs = {} if init_kwargs is None else init_kwargs
#     fit_kwargs = {} if fit_kwargs is None else fit_kwargs

#     # Directories
#     model_dir = output_dir + "/model"

#     # Create the model object
#     _logger.info("Building model")
#     default_config_kwargs = {
#         "n_channels": data.n_channels,
#         "sequence_length": 2000,
#         "batch_size": 32,
#         "learning_rate": 0.01,
#         "n_epochs": 20,
#     }
#     config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
#     _logger.info(f"Using config_kwargs: {config_kwargs}")

#     #################################################################################################
    
#     # --- Inject per-state initial covariances using the exact channels selected in load_data ---
#     init_cov = config_kwargs.get("initial_covariances", None)
#     K = int(config_kwargs.get("n_states", 0))
#     C = int(config_kwargs.get("n_channels", data.n_channels))
#     diag = bool(config_kwargs.get("diagonal_covariances", False))

#     if isinstance(init_cov, str):
#         arr = np.load(init_cov)

#         if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
#             # Case A: full FC (D,D) -> slice to (C,C) using selected channels, then tile
#             if not hasattr(data, "selected_channels"):
#                 raise RuntimeError("[*] Data.selected_channels not found; needed to slice full FC.")
#             chan_idx = np.array(data.selected_channels, dtype=int).ravel()
#             if chan_idx.size != C:
#                 raise RuntimeError(f"[*] selected_channels has {chan_idx.size} entries but n_channels={C}.")

#             R = arr[np.ix_(chan_idx, chan_idx)]
#             R = (R + R.T) / 2.0
#             np.fill_diagonal(R, 1.0)
#             R = R + np.eye(C) * 1e-6

#             if diag:
#                 variances = np.diag(R).astype(np.float32)
#                 init_cov_arr = np.tile(variances[None, :], (K, 1))            # (K, C)
#             else:
#                 init_cov_arr = np.tile(R[None, :, :], (K, 1, 1)).astype(np.float32)  # (K, C, C)

#             config_kwargs["initial_covariances"] = init_cov_arr

#         elif arr.ndim == 3 and arr.shape == (K, C, C):
#             # Case B: already (K,C,C) -> use as is
#             config_kwargs["initial_covariances"] = arr.astype(np.float32, copy=False)

#         elif arr.ndim == 2 and arr.shape == (K, C) and diag:
#             # Case C: already (K,C) variances for diagonal model
#             config_kwargs["initial_covariances"] = arr.astype(np.float32, copy=False)

#         else:
#             raise ValueError(
#                 f"initial_covariances at {init_cov} has unexpected shape {arr.shape} "
#                 f"for K={K}, C={C}, diagonal={diag}."
#             )

#     elif isinstance(init_cov, np.ndarray):
#         # Validate ndarray provided directly
#         expected = (K, C) if diag else (K, C, C)
#         if init_cov.shape != expected:
#             raise ValueError(f"initial_covariances array has shape {init_cov.shape} but expected {expected}")
#         config_kwargs["initial_covariances"] = init_cov.astype(np.float32, copy=False)

#     # --- end injection ---

#     #################################################################################################

    # # Deal with the special case of static FC model (n_state = 1 )
    # if config_kwargs['n_states'] == 1:
    #     ts = data.time_series(prepared=True, concatenate=False)
    #     # Note training_data.keep is in order. You need to preserve the order
    #     # between data and alpha.
    #     ts = [ts[i] for i in data.keep]
    #     # Concatenate across all sessions
    #     ts = np.concatenate(ts, axis=0)

    #     from osl_dynamics.array_ops import estimate_gaussian_distribution
    #     means, covs = estimate_gaussian_distribution(ts, nonzero_means=config_kwargs['learn_means'])

    #     inf_params_dir = output_dir + "/inf_params"
    #     os.makedirs(inf_params_dir, exist_ok=True)

    #     save(f"{inf_params_dir}/means.npy", means)
    #     save(f"{inf_params_dir}/covs.npy", covs)
    #     return

#     config = hmm.Config(**config_kwargs)
#     model = hmm.Model(config)
#     model.summary()

#     # Initialisation
#     default_init_kwargs = {"n_init": 3, "n_epochs": 1}
#     init_kwargs = override_dict_defaults(default_init_kwargs, init_kwargs)
#     _logger.info(f"Using init_kwargs: {init_kwargs}")
#     init_history = model.random_state_time_course_initialization(
#         data,
#         **init_kwargs,
#     )

#     # Training
#     history = model.fit(data, **fit_kwargs)

#     # Get the variational free energy
#     history["free_energy"] = model.free_energy(data)

#     # Save trained model
#     _logger.info(f"Saving model to: {model_dir}")
#     model.save(model_dir)
#     save(f"{model_dir}/init_history.pkl", init_history)
#     save(f"{model_dir}/history.pkl", history)

#     if save_inf_params:
#         # Make output directory
#         inf_params_dir = output_dir + "/inf_params"
#         os.makedirs(inf_params_dir, exist_ok=True)

#         # Get the inferred parameters
#         alpha = model.get_alpha(data)
#         means, covs = model.get_means_covariances()

#         # Save inferred parameters
#         save(f"{inf_params_dir}/alp.pkl", alpha)
#         save(f"{inf_params_dir}/means.npy", means)
#         save(f"{inf_params_dir}/covs.npy", covs)

#     if calculate_free_energy:
#         # Make output directory
#         metric_dir = output_dir + "/metrics/"
#         os.makedirs(metric_dir, exist_ok=True)

#         # Get the free energy
#         free_energy = model.free_energy(data, return_components=True)
#         free_energy, log_likelihood, entropy, prior = model.free_energy(data, return_components=True)
#         evidence = model.evidence(data)
#         metrics = {'free_energy': float(free_energy),
#                    'log_likelihood': float(log_likelihood),
#                    'entropy': float(entropy),
#                    'prior': float(prior),
#                    'evidence': float(evidence),
#                    }
#         with open(f'{metric_dir}metrics.json', "w") as json_file:
#             # Use json.dump to write the data to the file
#             json.dump(metrics, json_file)

#     # Concatenate loss from init_history and history
#     all_losses = np.concatenate((init_history['loss'], history['loss']))

#     # Generate corresponding x-axis values
#     epochs = np.arange(1, len(all_losses) + 1)

#     # Plot the loss function
#     plot_line(
#         x=[epochs],
#         y=[all_losses],
#         labels=["Loss"],
#         x_label="Epochs",
#         y_label="Loss",
#         title="Training Loss Function",
#         filename=os.path.join(output_dir, 'loss_function.pdf')
#     )

### THIS HAS MEANS AND STATIC FC WORKING BUT NOT VARIANCES (see next function) ###

# def train_hmm(
#         data,
#         output_dir,
#         config_kwargs,
#         init_kwargs=None,
#         infer_spatial='sample',
#         fit_kwargs=None,
#         save_inf_params=True,
#         calculate_free_energy=True
# ):
#     """Train a Hidden Markov Model (OSL-Dynamics) with optional on-the-fly static FC init.

#     Only instrumentation added for debugging; core logic unchanged.
#     """
#     if data is None:
#         raise ValueError("data must be passed.")

#     from osl_dynamics.models import hmm
#     from osl_dynamics.array_ops import estimate_gaussian_distribution

#     init_kwargs = {} if init_kwargs is None else init_kwargs
#     fit_kwargs = {} if fit_kwargs is None else fit_kwargs

#     # Directories
#     model_dir = output_dir + "/model"
#     debug_dir = os.path.join(output_dir, "debug")
#     os.makedirs(debug_dir, exist_ok=True)

#     # Create the model object
#     _logger.info("Building model")
#     default_config_kwargs = {
#         "n_channels": data.n_channels,
#         "sequence_length": 2000,
#         "batch_size": 32,
#         "learning_rate": 0.01,
#         "n_epochs": 20,
#     }
#     config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
#     _logger.info(f"Using config_kwargs: {config_kwargs}")

#     # Common derived config
#     K = int(config_kwargs.get("n_states", 0))
#     C_cfg = int(config_kwargs.get("n_channels", data.n_channels))
#     diag = bool(config_kwargs.get("diagonal_covariances", False))
#     init_cov = config_kwargs.get("initial_covariances", None)

#     _logger.info(f"[dbg] K (n_states)={K}, C (from config)={C_cfg}, diag={diag}, init_cov={init_cov!r}")

#     #################################################################################################
#     # Option A: On-the-fly static FC requested for multi-state runs
#     #################################################################################################
#     if isinstance(init_cov, str) and init_cov == "static_fc" and K > 1:
#         _logger.info("[static_fc] Computing static covariance on-the-fly from prepared data (K>1 path).")
#         ts = data.time_series(prepared=True, concatenate=False)
#         ts = [ts[i] for i in getattr(data, "keep", range(len(ts)))]  # preserve training order if present
#         ts = np.concatenate(ts, axis=0)  # (T_total, C_actual)

#         C_actual = ts.shape[1]
#         _logger.info(f"[dbg K>1] ts shape={ts.shape}, dtype={ts.dtype}, C_actual={C_actual}")
#         if C_actual != C_cfg:
#             _logger.warning(f"[dbg K>1] C mismatch: config={C_cfg}, data={C_actual}. Proceeding with data size.")
#         C = C_actual  # use what data says for all shape checks below

#         # Estimate Gaussian
#         nonzero_means = bool(config_kwargs.get("learn_means", False))
#         _logger.info(f"[dbg K>1] Calling estimate_gaussian_distribution(nonzero_means={nonzero_means})")
#         means_s, covs_s = estimate_gaussian_distribution(ts, nonzero_means=nonzero_means)

#         # --- NEW: normalise shapes ---
#         # covs_s may be (C, C) or (1, C, C). Make it (C, C).
#         covs_s = np.asarray(covs_s)
#         if covs_s.ndim == 3 and covs_s.shape[0] == 1:
#             covs_s = covs_s[0]
#         elif covs_s.ndim != 2:
#             raise ValueError(f"[static_fc] Unexpected covs_s shape: {covs_s.shape}")

#         # ensure symmetry + jitter
#         covs_s = (covs_s + covs_s.T) / 2.0
#         covs_s = covs_s.astype(np.float32, copy=False)
#         covs_s = covs_s + np.eye(covs_s.shape[0], dtype=np.float32) * 1e-6

#         K = int(config_kwargs.get("n_states", 0))
#         diag = bool(config_kwargs.get("diagonal_covariances", False))

#         if diag:
#             variances = np.diag(covs_s).astype(np.float32)             # (C,)
#             init_cov_arr = np.repeat(variances[None, :], K, axis=0)    # (K, C)
#         else:
#             init_cov_arr = np.repeat(covs_s[None, :, :], K, axis=0)    # (K, C, C)

#         _logger.info(f"[dbg K>1] init_cov_arr.shape={init_cov_arr.shape} "
#                     f"(expected {(K, covs_s.shape[0]) if diag else (K, covs_s.shape[0], covs_s.shape[0])})")

#         config_kwargs["initial_covariances"] = init_cov_arr

#         _logger.info(f"[static_fc] Injected initial_covariances with shape {init_cov_arr.shape}.")

#     #################################################################################################
#     # True static FC model (n_states == 1): unchanged shortcut + early return
#     #################################################################################################
    # if config_kwargs['n_states'] == 1:
    #     _logger.info("[n_states==1] Static shortcut: estimating single Gaussian and returning.")
    #     ts = data.time_series(prepared=True, concatenate=False)
    #     ts = [ts[i] for i in getattr(data, "keep", range(len(ts)))]
    #     ts = np.concatenate(ts, axis=0)

    #     _logger.info(f"[dbg n==1] ts shape={ts.shape}, dtype={ts.dtype}")
    #     nonzero_means = bool(config_kwargs.get("learn_means", False))
    #     _logger.info(f"[dbg n==1] Calling estimate_gaussian_distribution(nonzero_means={nonzero_means})")

    #     means, covs = estimate_gaussian_distribution(ts, nonzero_means=nonzero_means)
    #     _logger.info(f"[dbg n==1] means.shape={np.shape(means)}, covs.shape={np.shape(covs)}")
    #     _logger.info(f"[dbg n==1] covs sample 5x5:\n{covs[:5, :5]}")

    #     # Save debug artifacts for cross-run comparison
    #     np.save(os.path.join(debug_dir, "static_covs_n1.npy"), covs)

    #     inf_params_dir = output_dir + "/inf_params"
    #     os.makedirs(inf_params_dir, exist_ok=True)

    #     save(f"{inf_params_dir}/means.npy", means)
    #     save(f"{inf_params_dir}/covs.npy", covs)
    #     return

#     # Build & train model (standard path)
#     config = hmm.Config(**config_kwargs)
#     model = hmm.Model(config)
#     model.summary()

#     # Initialisation
#     default_init_kwargs = {"n_init": 3, "n_epochs": 1}
#     init_kwargs = override_dict_defaults(default_init_kwargs, init_kwargs)
#     _logger.info(f"Using init_kwargs: {init_kwargs}")
#     init_history = model.random_state_time_course_initialization(data, **init_kwargs)

#     # Training
#     history = model.fit(data, **fit_kwargs)

#     # Get the variational free energy
#     history["free_energy"] = model.free_energy(data)

#     # Save trained model
#     _logger.info(f"Saving model to: {model_dir}")
#     model.save(model_dir)
#     save(f"{model_dir}/init_history.pkl", init_history)
#     save(f"{model_dir}/history.pkl", history)

#     if save_inf_params:
#         # Make output directory
#         inf_params_dir = output_dir + "/inf_params"
#         os.makedirs(inf_params_dir, exist_ok=True)

#         # Get the inferred parameters
#         alpha = model.get_alpha(data)
#         means, covs = model.get_means_covariances()

#         # Save inferred parameters
#         save(f"{inf_params_dir}/alp.pkl", alpha)
#         save(f"{inf_params_dir}/means.npy", means)
#         save(f"{inf_params_dir}/covs.npy", covs)

#     if calculate_free_energy:
#         # Make output directory
#         metric_dir = output_dir + "/metrics/"
#         os.makedirs(metric_dir, exist_ok=True)

#         # Get the free energy
#         free_energy, log_likelihood, entropy, prior = model.free_energy(data, return_components=True)
#         evidence = model.evidence(data)
#         metrics = {
#             'free_energy': float(free_energy),
#             'log_likelihood': float(log_likelihood),
#             'entropy': float(entropy),
#             'prior': float(prior),
#             'evidence': float(evidence),
#         }
#         with open(f'{metric_dir}metrics.json', "w") as json_file:
#             json.dump(metrics, json_file)

#     # Concatenate loss from init_history and history
#     all_losses = np.concatenate((init_history['loss'], history['loss']))

#     # Generate corresponding x-axis values
#     epochs = np.arange(1, len(all_losses) + 1)

#     # Plot the loss function
#     plot_line(
#         x=[epochs],
#         y=[all_losses],
#         labels=["Loss"],
#         x_label="Epochs",
#         y_label="Loss",
#         title="Training Loss Function",
#         filename=os.path.join(output_dir, 'loss_function.pdf')
#     )

def train_hmm(
        data,
        output_dir,
        config_kwargs,
        init_kwargs=None,
        infer_spatial='sample',
        fit_kwargs=None,
        save_inf_params=True,
        calculate_free_energy=True,
):
    """
    Train an OSL-Dynamics HMM with optional static FC usage.

    Modes:
    1) n_states == 1
       - Compute static Gaussian (means & covs) from prepared data.
       - Save to <output_dir>/inf_params/{means.npy,covs.npy}.
       - Also save <output_dir>/debug/static_cov.npy for convenience.
       - No HMM is built.

    2) n_states > 1 AND initial_covariances == "static_fc"
       - Compute static FC (same estimator as in mode 1).
       - Use it as the *initializer passed into the model*:
         • if diagonal_covariances=True  -> K×C variances from diag(static FC)
         • else (full covariances)       -> K×C×C static FC replicated across states
       - Then build/train HMM normally. Whether covariances are updated is
         governed by learn_covariances and diagonal_covariances.

    3) n_states > 1 AND initial_covariances not set
       - Standard HMM training; no static FC used.
    """
    if data is None:
        raise ValueError("data must be passed.")

    from osl_dynamics.models import hmm
    from osl_dynamics.array_ops import estimate_gaussian_distribution

    init_kwargs = {} if init_kwargs is None else init_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

    # ------------------- Directories -------------------
    model_dir = os.path.join(output_dir, "model")
    debug_dir = os.path.join(output_dir, "debug")
    inf_params_dir = os.path.join(output_dir, "inf_params")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(inf_params_dir, exist_ok=True)

    # ------------------- Config -------------------
    _logger.info("Building model configuration.")
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "sequence_length": 2000,
        "batch_size": 32,
        "learning_rate": 0.01,
        "n_epochs": 20,
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)

    K = int(config_kwargs.get("n_states", 0))
    C_cfg = int(config_kwargs.get("n_channels", data.n_channels))
    diag = bool(config_kwargs.get("diagonal_covariances", False))
    init_cov_mode = config_kwargs.get("initial_covariances", None)

    _logger.info(f"[dbg] Config: K={K}, C={C_cfg}, diagonal={diag}, init_cov={init_cov_mode!r}")

    def _compute_static_gaussian():
        """Returns (means, cov) using the same estimator as the original 1-state path."""
        ts = data.time_series(prepared=True, concatenate=False)
        ts = [ts[i] for i in getattr(data, "keep", range(len(ts)))]
        ts = np.concatenate(ts, axis=0)  # (T_total, C_actual)
        C_actual = ts.shape[1]
        if C_actual != C_cfg:
            raise ValueError(f"[static] Channel mismatch: data={C_actual}, config={C_cfg}")
        means, cov = estimate_gaussian_distribution(
            ts, nonzero_means=bool(config_kwargs.get("learn_means", False))
        )
        cov = np.asarray(cov)
        if cov.ndim == 3 and cov.shape[0] == 1:
            cov = cov[0]
        elif cov.ndim != 2:
            raise ValueError(f"[static] Unexpected cov shape: {cov.shape}")
        cov = ((cov + cov.T) / 2.0).astype(np.float32, copy=False)
        return means, cov

    # =================== MODE 1: n_states == 1 ===================
    if K == 1:
        _logger.info("[mode 1] n_states=1 → compute static means/covariances and save only.")
        means_static, cov_static = _compute_static_gaussian()

        # Save for downstream
        save(os.path.join(inf_params_dir, "means.npy"), means_static)
        save(os.path.join(inf_params_dir, "covs.npy"), cov_static)

        # Debug copy
        np.save(os.path.join(debug_dir, "static_cov.npy"), cov_static)
        _logger.info(f"[static] Saved means.npy (shape {np.asarray(means_static).shape}) "
                     f"and covs.npy (shape {cov_static.shape}) to {inf_params_dir}")
        _logger.info(f"[static] First 5×5 block of cov:\n{cov_static[:5, :5]}")
        return

    # =================== MODE 2: K>1 and init_cov == "static_fc" ===================
    if isinstance(init_cov_mode, str) and init_cov_mode == "static_fc":
        _logger.info("[mode 2] Multi-state + 'static_fc' → compute static FC and USE as initializer.")
        # Compute static FC once (same as mode 1)
        means_static, cov_static = _compute_static_gaussian()
        np.save(os.path.join(debug_dir, "static_cov.npy"), cov_static)
        save(os.path.join(debug_dir, "static_means.npy"), means_static)
        _logger.info(f"[static] Computed static FC with shape {cov_static.shape}")

        # Build initializer array expected by OSL given diagonal/full setting
        if diag:
            variances = np.diag(cov_static).astype(np.float32)      # (C,)
            init_cov_arr = np.repeat(variances[None, :], K, axis=0) # (K, C)
        else:
            init_cov_arr = np.repeat(cov_static[None, :, :], K, axis=0).astype(np.float32)  # (K, C, C)

        # Replace the "static_fc" string with the actual array initializer
        config_kwargs = dict(config_kwargs)  # shallow copy just in case
        config_kwargs["initial_covariances"] = init_cov_arr

        # ---- Build & train HMM (standard flow) ----
        ip = config_kwargs.get("initial_trans_prob", None)
        if isinstance(ip, list):
            config_kwargs["initial_trans_prob"] = np.array(ip, dtype=float)
            _logger.info(f"[train] Converted initial_trans_prob from list to np.ndarray "
                        f"with shape {config_kwargs['initial_trans_prob'].shape}")
        elif isinstance(ip, np.ndarray):
            _logger.info(f"[train] initial_trans_prob already np.ndarray with shape {ip.shape}")
        else:
            _logger.info("[train] No initial_trans_prob provided (using random or default)")

        config = hmm.Config(**config_kwargs)
        model = hmm.Model(config)
        model.summary()

        default_init_kwargs = {"n_init": 3, "n_epochs": 1}
        init_kwargs = override_dict_defaults(default_init_kwargs, init_kwargs)
        _logger.info(f"[train] Using init_kwargs: {init_kwargs}")

        init_history = model.random_state_time_course_initialization(data, **init_kwargs)
        history = model.fit(data, **fit_kwargs)
        history["free_energy"] = model.free_energy(data)

        _logger.info(f"[train] Saving model to: {model_dir}")
        model.save(model_dir)
        save(os.path.join(model_dir, "init_history.pkl"), init_history)
        save(os.path.join(model_dir, "history.pkl"), history)

        if save_inf_params:
            alpha = model.get_alpha(data)
            means, covs = model.get_means_covariances()
            save(os.path.join(inf_params_dir, "alp.pkl"), alpha)
            save(os.path.join(inf_params_dir, "means.npy"), means)
            save(os.path.join(inf_params_dir, "covs.npy"), covs)

        # Metrics (optional)
        if calculate_free_energy:
            metric_dir = os.path.join(output_dir, "metrics")
            os.makedirs(metric_dir, exist_ok=True)
            fe, ll, ent, prior = model.free_energy(data, return_components=True)
            evidence = model.evidence(data)
            with open(os.path.join(metric_dir, "metrics.json"), "w") as f:
                json.dump({
                    "free_energy": float(fe),
                    "log_likelihood": float(ll),
                    "entropy": float(ent),
                    "prior": float(prior),
                    "evidence": float(evidence),
                }, f, indent=2)

        # Plot loss
        if "loss" in init_history and "loss" in history:
            all_losses = np.concatenate((init_history["loss"], history["loss"]))
            epochs = np.arange(1, len(all_losses) + 1)
            plot_line(
                x=[epochs],
                y=[all_losses],
                labels=["Loss"],
                x_label="Epochs",
                y_label="Loss",
                title="Training Loss Function",
                filename=os.path.join(output_dir, "loss_function.pdf"),
            )
        return

    # =================== MODE 3: K>1, no 'static_fc' ===================
    _logger.info("[mode 3] Standard HMM training (no static FC used).")

    # --- ensure initial_trans_prob is a numpy array (Config requires it) ---
    ip = config_kwargs.get("initial_trans_prob", None)
    if isinstance(ip, list):
        config_kwargs["initial_trans_prob"] = np.asarray(ip, dtype=float)
        _logger.info(f"[train] Converted initial_trans_prob list→ndarray "
                    f"shape={config_kwargs['initial_trans_prob'].shape}")
    elif isinstance(ip, np.ndarray):
        _logger.info(f"[train] initial_trans_prob already ndarray shape={ip.shape}")
    else:
        _logger.info("[train] No initial_trans_prob provided (using random/default)")

    config = hmm.Config(**config_kwargs)
    model = hmm.Model(config)
    model.summary()

    default_init_kwargs = {"n_init": 3, "n_epochs": 1}
    init_kwargs = override_dict_defaults(default_init_kwargs, init_kwargs)
    _logger.info(f"[train] Using init_kwargs: {init_kwargs}")

    init_history = model.random_state_time_course_initialization(data, **init_kwargs)
    history = model.fit(data, **fit_kwargs)
    history["free_energy"] = model.free_energy(data)

    _logger.info(f"[train] Saving model to: {model_dir}")
    model.save(model_dir)
    save(os.path.join(model_dir, "init_history.pkl"), init_history)
    save(os.path.join(model_dir, "history.pkl"), history)

    if save_inf_params:
        alpha = model.get_alpha(data)
        means, covs = model.get_means_covariances()
        save(os.path.join(inf_params_dir, "alp.pkl"), alpha)
        save(os.path.join(inf_params_dir, "means.npy"), means)
        save(os.path.join(inf_params_dir, "covs.npy"), covs)

    # Metrics (optional)
    if calculate_free_energy:
        metric_dir = os.path.join(output_dir, "metrics")
        os.makedirs(metric_dir, exist_ok=True)
        fe, ll, ent, prior = model.free_energy(data, return_components=True)
        evidence = model.evidence(data)
        with open(os.path.join(metric_dir, "metrics.json"), "w") as f:
            json.dump({
                "free_energy": float(fe),
                "log_likelihood": float(ll),
                "entropy": float(ent),
                "prior": float(prior),
                "evidence": float(evidence),
            }, f, indent=2)

    # Plot loss
    if "loss" in init_history and "loss" in history:
        all_losses = np.concatenate((init_history["loss"], history["loss"]))
        epochs = np.arange(1, len(all_losses) + 1)
        plot_line(
            x=[epochs],
            y=[all_losses],
            labels=["Loss"],
            x_label="Epochs",
            y_label="Loss",
            title="Training Loss Function",
            filename=os.path.join(output_dir, "loss_function.pdf"),
        )


def train_dynemo(
        data,
        output_dir,
        config_kwargs,
        init_kwargs=None,
        fit_kwargs=None,
        save_inf_params=True,
        calculate_free_energy=True
):
    """Train `DyNeMo <https://osl-dynamics.readthedocs.io/en/latest/autoapi\
    /osl_dynamics/models/dynemo/index.html>`_.

    This function will:

    1. Build a :code:`dynemo.Model` object.
    2. Initialize the parameters of the model using
       :code:`Model.random_subset_initialization`.
    3. Perform full training.
    4. Save the inferred parameters (mode mixing coefficients, means and
       covariances) if :code:`save_inf_params=True`.

    This function will create two directories:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object for training the model.
    output_dir : str
        Path to output directory.
    config_kwargs : dict
        Keyword arguments to pass to `dynemo.Config <https://osl-dynamics\
        .readthedocs.io/en/latest/autoapi/osl_dynamics/models/dynemo\
        /index.html#osl_dynamics.models.dynemo.Config>`_. Defaults to::

            {'n_channels': data.n_channels.
             'sequence_length': 200,
             'inference_n_units': 64,
             'inference_normalization': 'layer',
             'model_n_units': 64,
             'model_normalization': 'layer',
             'learn_alpha_temperature': True,
             'initial_alpha_temperature': 1.0,
             'do_kl_annealing': True,
             'kl_annealing_curve': 'tanh',
             'kl_annealing_sharpness': 10,
             'n_kl_annealing_epochs': 20,
             'batch_size': 128,
             'learning_rate': 0.01,
             'lr_decay': 0.1,
             'n_epochs': 40}
    init_kwargs : dict, optional
        Keyword arguments to pass to :code:`Model.random_subset_initialization`.
        Defaults to::

            {'n_init': 5, 'n_epochs': 2, 'take': 1}.
    fit_kwargs : dict, optional
        Keyword arguments to pass to the :code:`Model.fit`.
    save_inf_params : bool, optional
        Should we save the inferred parameters?
    calculate_free_energy: bool, optional
        Should we calculate free energy on training set?
    """

    init_kwargs = {} if init_kwargs is None else init_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

    if data is None:
        raise ValueError("data must be passed.")

    # Deal with the special case of static FC model (n_state = 1 )
    if config_kwargs['n_modes'] == 1:
        ts = data.time_series(prepared=True, concatenate=False)
        # Note training_data.keep is in order. You need to preserve the order
        # between data and alpha.
        ts = [ts[i] for i in data.keep]
        # Concatenate across all sessions
        ts = np.concatenate(ts, axis=0)

        from osl_dynamics.array_ops import estimate_gaussian_distribution
        means, covs = estimate_gaussian_distribution(ts, nonzero_means=config_kwargs['learn_means'])

        inf_params_dir = output_dir + "/inf_params"
        os.makedirs(inf_params_dir, exist_ok=True)

        save(f"{inf_params_dir}/means.npy", means)
        save(f"{inf_params_dir}/covs.npy", covs)
        return

    from osl_dynamics.models import dynemo

    # Directories
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"

    # Create the model object
    _logger.info("Building model")
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "sequence_length": 200,
        "inference_n_units": 64,
        "inference_normalization": "layer",
        "model_n_units": 64,
        "model_normalization": "layer",
        "learn_alpha_temperature": True,
        "initial_alpha_temperature": 1.0,
        "do_kl_annealing": True,
        "kl_annealing_curve": "tanh",
        "kl_annealing_sharpness": 10,
        "n_kl_annealing_epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.01,
        "lr_decay": 0.1,
        "n_epochs": 40,
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
    _logger.info(f"Using config_kwargs: {config_kwargs}")
    config = dynemo.Config(**config_kwargs)
    model = dynemo.Model(config)
    model.summary()

    # Set regularisers
    model.set_regularizers(data)

    # Initialisation
    default_init_kwargs = {"n_init": 5, "n_epochs": 2, "take": 1}
    init_kwargs = override_dict_defaults(default_init_kwargs, init_kwargs)
    _logger.info(f"Using init_kwargs: {init_kwargs}")
    init_history = model.random_subset_initialization(data, **init_kwargs)

    # Keyword arguments for the fit method
    default_fit_kwargs = {}
    fit_kwargs = override_dict_defaults(default_fit_kwargs, fit_kwargs)
    _logger.info(f"Using fit_kwargs: {fit_kwargs}")

    # Training
    history = model.fit(data, **fit_kwargs)

    # Add free energy to the history object
    history["free_energy"] = history["loss"][-1]

    # Save trained model
    _logger.info(f"Saving model to: {model_dir}")
    model.save(model_dir)
    save(f"{model_dir}/init_history.pkl", init_history)
    save(f"{model_dir}/history.pkl", history)

    if save_inf_params:
        os.makedirs(inf_params_dir, exist_ok=True)

        # Get the inferred parameters
        alpha = model.get_alpha(data)
        means, covs = model.get_means_covariances()

        # Save inferred parameters
        save(f"{inf_params_dir}/alp.pkl", alpha)
        save(f"{inf_params_dir}/means.npy", means)
        save(f"{inf_params_dir}/covs.npy", covs)

    if calculate_free_energy:
        # Make output directory
        metric_dir = output_dir + "/metrics/"
        os.makedirs(metric_dir, exist_ok=True)

        # Get the free energy
        free_energy = model.free_energy(data)
        metrics = {'free_energy': float(free_energy)}
        with open(f'{metric_dir}metrics.json', "w") as json_file:
            # Use json.dump to write the data to the file
            json.dump(metrics, json_file)

    # Concatenate loss from init_history and history
    all_losses = np.concatenate((init_history['loss'], history['loss']))

    # Generate corresponding x-axis values
    epochs = np.arange(1, len(all_losses) + 1)

    # Plot the loss function
    plot_line(
        x=[epochs],
        y=[all_losses],
        labels=["Loss"],
        x_label="Epochs",
        y_label="Loss",
        title="Training Loss Function",
        filename=os.path.join(output_dir, 'loss_function.pdf')
    )


def train_hive(
        data,
        output_dir,
        config_kwargs,
        init_kwargs=None,
        fit_kwargs=None,
        save_inf_params=True,
):
    """ Train a `HIVE Model <https://osl-dynamics.\
    readthedocs.io/en/latest/autoapi/osl_dynamics/models/hive/index.html>`_.
    
    This function will:

    1. Build an :code:`hive.Model` object.
    2. Initialize the parameters of the HIVE model using
        :code:`Model.random_state_time_course_initialization`.
    3. Perform full training.
    4. Save the inferred parameters (state probabilities, means,
        covariances and embeddings) if :code:`save_inf_params=True`.
    
    This function will create two directories:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.
        This directory is only created if :code:`save_inf_params=True`.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object for training the model.
    output_dir : str
        Path to output directory.
    config_kwargs : dict
        Keyword arguments to pass to `hive.Config <https://osl-dynamics\
        .readthedocs.io/en/latest/autoapi/osl_dynamics/models/hive/index.html\
        #osl_dynamics.models.hive.Config>`_. Defaults to::

            {
                'sequence_length': 200,
                'spatial_embeddings_dim': 2,
                'dev_n_layers': 5,
                'dev_n_units': 32,
                'dev_activation': 'tanh',
                'dev_normalization': 'layer',
                'dev_regularizer': 'l1',
                'dev_regularizer_factor': 10,
                'batch_size': 128,
                'learning_rate': 0.005,
                'lr_decay': 0.1,
                'n_epochs': 30,
                'do_kl_annealing': True,
                'kl_annealing_curve': 'tanh',
                'kl_annealing_sharpness': 10,
                'n_kl_annealing_epochs': 15,
            }.
    init_kwargs : dict, optional
        Keyword arguments to pass to
        :code:`Model.random_state_time_course_initialization`. Defaults to::

            {'n_init': 10, 'n_epochs': 2}.
    fit_kwargs : dict, optional
        Keyword arguments to pass to the :code:`Model.fit`. No defaults.
    save_inf_params : bool, optional
        Should we save the inferred parameters?
    """
    if data is None:
        raise ValueError("data must be passed.")

    from osl_dynamics.models import hive

    init_kwargs = {} if init_kwargs is None else init_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

    # Directories
    model_dir = output_dir + "/model"

    _logger.info("Building model")

    # SE-HMM config
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "n_sessions": data.n_sessions,
        "sequence_length": 200,
        "spatial_embeddings_dim": 2,
        "dev_n_layers": 5,
        "dev_n_units": 32,
        "dev_activation": "tanh",
        "dev_normalization": "layer",
        "dev_regularizer": "l1",
        "dev_regularizer_factor": 10,
        "batch_size": 128,
        "learning_rate": 0.005,
        "lr_decay": 0.1,
        "n_epochs": 30,
        "do_kl_annealing": True,
        "kl_annealing_curve": "tanh",
        "kl_annealing_sharpness": 10,
        "n_kl_annealing_epochs": 15,
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)

    default_init_kwargs = {"n_init": 10, "n_epochs": 2}
    init_kwargs = override_dict_defaults(default_init_kwargs, init_kwargs)
    _logger.info(f"Using init_kwargs: {init_kwargs}")

    # Initialise and train HIVE
    _logger.info(f"Using config_kwargs: {config_kwargs}")
    config = hive.Config(**config_kwargs)
    model = hive.Model(config)
    model.summary()

    # Set regularisers
    model.set_regularizers(data)

    # Set deviation initializer
    model.set_dev_parameters_initializer(data)

    # Initialise HIVE
    _logger.info(f"Using init_kwargs: {init_kwargs}")
    init_history = model.random_state_time_course_initialization(
        data,
        **init_kwargs,
    )

    # Training
    history = model.fit(data, **fit_kwargs)

    _logger.info(f"Saving model to: {model_dir}")
    model.save(model_dir)

    del model
    model = hive.Model.load(model_dir)

    # Get the variational free energy
    history["free_energy"] = model.free_energy(data)

    # Save trained model
    save(f"{model_dir}/init_history.pkl", init_history)
    save(f"{model_dir}/history.pkl", history)

    if save_inf_params:
        # Make output directory
        inf_params_dir = output_dir + "/inf_params"
        os.makedirs(inf_params_dir, exist_ok=True)

        # Get the inferred parameters
        alpha = model.get_alpha(data)
        means, covs = model.get_means_covariances()
        session_means, session_covs = model.get_session_means_covariances()
        embeddings = model.get_embeddings()

        # Save inferred parameters
        save(f"{inf_params_dir}/alp.pkl", alpha)
        save(f"{inf_params_dir}/means.npy", means)
        save(f"{inf_params_dir}/covs.npy", covs)
        save(f"{inf_params_dir}/session_means.npy", session_means)
        save(f"{inf_params_dir}/session_covs.npy", session_covs)
        save(f"{inf_params_dir}/embeddings.npy", embeddings)


def get_inf_params(data, output_dir, observation_model_only=False):
    """Get inferred alphas.

    This function expects a model has already been trained and the following
    directory to exist:

    - :code:`<output_dir>/model`, which contains the trained model.

    This function will create the following directory:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    observation_model_only : bool, optional
        We we only want to get the observation model parameters?
    """
    # Make output directory
    inf_params_dir = output_dir + "/inf_params"
    os.makedirs(inf_params_dir, exist_ok=True)

    #  Load model
    from osl_dynamics.models import load

    model_dir = output_dir + "/model"
    model = load(model_dir)

    if observation_model_only:
        # Get the inferred parameters
        means, covs = model.get_means_covariances()

        # Save
        save(f"{inf_params_dir}/means.npy", means)
        save(f"{inf_params_dir}/covs.npy", covs)
    else:
        # Get the inferred parameters
        alpha = model.get_alpha(data)
        means, covs = model.get_means_covariances()

        # Save
        save(f"{inf_params_dir}/alp.pkl", alpha)
        save(f"{inf_params_dir}/means.npy", means)
        save(f"{inf_params_dir}/covs.npy", covs)


def plot_power_maps_from_covariances(
        data,
        output_dir,
        mask_file=None,
        parcellation_file=None,
        power_save_kwargs=None,
):
    """Plot power maps calculated directly from the inferred covariances.

    This function expects a model has already been trained and the following
    directory to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will output files called :code:`covs_.png` which contain
    plots of the power map of each state/mode taken directly from the inferred
    covariance matrices. The files will be saved to
    :code:`<output_dir>/inf_params`.

    This function also expects the data to be prepared in the same script
    that this wrapper is called from.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    mask_file : str, optional
        Mask file used to preprocess the training data. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str, optional
        Parcellation file used to parcellate the training data. If
        :code:`None`, we use :code:`data.parcellation_file`.
    power_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'filename': '<inf_params_dir>/covs_.png',
             'mask_file': data.mask_file,
             'parcellation_file': data.parcellation_file,
             'plot_kwargs': {'symmetric_cbar': True}}
    """
    # Validation
    power_save_kwargs = {} if power_save_kwargs is None else power_save_kwargs

    if mask_file is None:
        if data is None or data.mask_file is None:
            raise ValueError(
                "mask_file must be passed or specified in the Data object."
            )
        else:
            mask_file = data.mask_file

    if parcellation_file is None:
        if data is None or data.parcellation_file is None:
            raise ValueError(
                "parcellation_file must be passed or specified in the Data object."
            )
        else:
            parcellation_file = data.parcellation_file

    if hasattr(data, "n_embeddings"):
        n_embeddings = data.n_embeddings
    else:
        n_embeddings = 1

    if hasattr(data, "pca_components"):
        pca_components = data.pca_components
    else:
        pca_components = None

    # Directories
    inf_params_dir = f"{output_dir}/inf_params"

    # Load inferred covariances
    covs = load(f"{inf_params_dir}/covs.npy")

    # Reverse the effects of preparing the data
    from osl_dynamics.analysis import modes

    covs = modes.raw_covariances(covs, n_embeddings, pca_components)

    # Save
    from osl_dynamics.analysis import power

    default_power_save_kwargs = {
        "filename": f"{inf_params_dir}/covs_.png",
        "mask_file": mask_file,
        "parcellation_file": parcellation_file,
        "plot_kwargs": {"symmetric_cbar": True},
    }
    if "plot_kwargs" in power_save_kwargs:
        power_save_kwargs["plot_kwargs"] = override_dict_defaults(
            default_power_save_kwargs["plot_kwargs"],
            power_save_kwargs["plot_kwargs"],
        )
    power_save_kwargs = override_dict_defaults(
        default_power_save_kwargs, power_save_kwargs
    )
    _logger.info(f"Using power_save_kwargs: {power_save_kwargs}")
    power.save(covs, **power_save_kwargs)


def plot_tde_covariances(data, output_dir):
    """Plot inferred covariance of the time-delay embedded data.

    This function expects a model has already been trained and the following
    directory to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will output a :code:`tde_covs.png` file containing a plot of
    the covariances in the :code:`<output_dir>/inf_params` directory.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    """
    inf_params_dir = f"{output_dir}/inf_params"

    covs = load(f"{inf_params_dir}/covs.npy")

    if hasattr(data, "pca_components"):
        if data.pca_components is not None:
            from osl_dynamics.analysis import modes

            covs = modes.reverse_pca(covs, data.pca_components)

    from osl_dynamics.utils import plotting

    plotting.plot_matrices(covs, filename=f"{inf_params_dir}/tde_covs.png")


def plot_state_psds(data, output_dir):
    """Plot state PSDs.

    This function expects multitaper spectra to have already been calculated
    and are in:

    - :code:`<output_dir>/spectra`.

    This function will output a file called :code:`psds.png` which contains
    a plot of each state PSD.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    """
    spectra_dir = f"{output_dir}/spectra"

    f = load(f"{spectra_dir}/f.npy")
    psd = load(f"{spectra_dir}/psd.npy")
    psd = np.mean(psd, axis=(0, 2))  # average over arrays and channels
    n_states = psd.shape[0]

    from osl_dynamics.utils import plotting

    plotting.plot_line(
        [f] * n_states,
        psd,
        labels=[f"State {i + 1}" for i in range(n_states)],
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        x_range=[f[0], f[-1]],
        filename=f"{spectra_dir}/psds.png",
    )


def dual_estimation(data, output_dir, n_jobs=1, concatenate=False,method='sample'):
    """Dual estimation for session-specific observation model parameters.

    This function expects a model has already been trained and the following
    directories to exist:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create the following directory:

    - :code:`<output_dir>/dual_estimates`, which contains the session-specific
      means and covariances.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    n_jobs : int, optional
        Number of jobs to run in parallel.
    concatenate: bool, optional
        Whether to concatenate all the sessions before calculate state statistics.
    """
    if data is None:
        raise ValueError("data must be passed.")

    # Directories
    model_dir = f"{output_dir}/model"
    inf_params_dir = f"{output_dir}/inf_params"
    dual_estimates_dir = f"{output_dir}/dual_estimates"
    os.makedirs(dual_estimates_dir, exist_ok=True)

    #  Load model
    from osl_dynamics import models

    model = models.load(model_dir)

    # Load the inferred state probabilities
    alpha = load(f"{inf_params_dir}/alp.pkl")

    # Dual estimation
    means, covs = model.dual_estimation(data, alpha=alpha, n_jobs=n_jobs, concatenate=concatenate,method=method)

    # Save
    save(f"{dual_estimates_dir}/means.npy", means)
    save(f"{dual_estimates_dir}/covs.npy", covs)


def log_likelihood(data, output_dir, static_FC=False, spatial=None,infer_alpha=False):
    """Log-likelihood estimation for the data.

    This function expects a model has already been trained and the following
    directories to exist:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the temporal dynamics.

    This function will create the following file:

    - :code:`<output_dir>/metrics.json`, which contains the average log-likelihood
    per session

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    static_FC: bool, optional
        Whether to work only with static FC, i.e., #states=1
    spatial: dictionary, optional
        Only when static_FC = True, use the spatial map file directory here.
    infer_alpha: bool, optional
        Whether alpha should be inferred using the mode when calculating log likelihood
    """
    if data is None:
        raise ValueError("data must be passed.")

    # Get the session-specific data
    ts = data.time_series(prepared=True, concatenate=False)

    # Note training_data.keep is in order. You need to preserve the order
    # between data and alpha.
    ts = [ts[i] for i in data.keep]

    if static_FC:
        means = np.load(spatial['means'])
        covs = np.load(spatial['covs'])
    else:
        # Directories
        model_dir = f"{output_dir}/model"
        inf_params_dir = f"{output_dir}/inf_params"

        #  Load model
        from osl_dynamics import models

        model = models.load(model_dir)

        if infer_alpha:
            alpha = None
        else:
            # Load the inferred state probabilities
            alpha = load(f"{inf_params_dir}/alp.pkl")
            if len(alpha) != len(ts):
                raise ValueError(
                    "len(alpha) and training_data.n_sessions must be the same."
                )
            # Stack both ts and alpha
            alpha = np.stack(alpha)

    ts = np.stack(ts)

    if static_FC:
        from osl_dynamics.array_ops import estimate_gaussian_log_likelihood
        metrics = float(estimate_gaussian_log_likelihood(ts, means, covs, average=True))
    else:
        # Get posterior expected log-likelihood (averaged over session)
        metrics = float(model.get_posterior_expected_log_likelihood(ts, alpha,average=True))

    # Save
    with open(f"{output_dir}metrics.json", "w") as file:
        json.dump({'log_likelihood': metrics}, file)

def free_energy(data, output_dir):
    """free energy estimation for the data.

    This function expects a model has already been trained and the following
    directories to exist:

    - :code:`<output_dir>/model`, which contains the trained model.

    This function will create the following file:

    - :code:`<output_dir>/metrics.json`, which contains the average log-likelihood
    per session

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    """
    from osl_dynamics import models
    if data is None:
        raise ValueError("data must be passed.")

    model_dir = f"{output_dir}/model/"
    model = models.load(model_dir)
    free_energy = float(model.free_energy(data))
    with open(f'{output_dir}ncv_free_energy.json', 'w') as f:
        json.dump([free_energy], f)


def multitaper_spectra(data, output_dir, kwargs, nnmf_components=None):
    """Calculate multitaper spectra.

    This function expects a model has already been trained and the following
    directories exist:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create the following directory:

    - :code:`<output_dir>/spectra`, which contains the post-hoc spectra.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    kwargs : dict
        Keyword arguments to pass to `analysis.spectral.multitaper_spectra
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/spectral/index.html#osl_dynamics.analysis.spectral\
        .multitaper_spectra>`_. Defaults to::

            {'sampling_frequency': data.sampling_frequency,
             'keepdims': True}
    nnmf_components : int, optional
        Number of non-negative matrix factorization (NNMF) components to fit to
        the stacked session-specific coherence spectra.
    """
    if data is None:
        raise ValueError("data must be passed.")

    sampling_frequency = kwargs.pop("sampling_frequency", None)
    if sampling_frequency is None and data.sampling_frequency is None:
        raise ValueError(
            "sampling_frequency must be passed or specified in the Data object."
        )
    else:
        sampling_frequency = data.sampling_frequency

    default_kwargs = {
        "sampling_frequency": sampling_frequency,
        "keepdims": True,
    }
    kwargs = override_dict_defaults(default_kwargs, kwargs)
    _logger.info(f"Using kwargs: {kwargs}")

    # Directories
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"
    spectra_dir = output_dir + "/spectra"
    os.makedirs(spectra_dir, exist_ok=True)

    # Load the inferred state probabilities
    alpha = load(f"{inf_params_dir}/alp.pkl")

    # Get the config used to create the model
    from osl_dynamics.models.mod_base import ModelBase

    model_config, _ = ModelBase.load_config(model_dir)

    # Get unprepared data (i.e. the data before calling Data.prepare)
    # We also trim the data to account for the data points lost to
    # time embedding or applying a sliding window
    data = data.trim_time_series(
        sequence_length=model_config["sequence_length"], prepared=False
    )

    # Calculate multitaper
    from osl_dynamics.analysis import spectral

    spectra = spectral.multitaper_spectra(data=data, alpha=alpha, **kwargs)

    # Unpack spectra and save
    return_weights = kwargs.pop("return_weights", False)
    if return_weights:
        f, psd, coh, w = spectra
        save(f"{spectra_dir}/f.npy", f)
        save(f"{spectra_dir}/psd.npy", psd)
        save(f"{spectra_dir}/coh.npy", coh)
        save(f"{spectra_dir}/w.npy", w)
    else:
        f, psd, coh = spectra
        save(f"{spectra_dir}/f.npy", f)
        save(f"{spectra_dir}/psd.npy", psd)
        save(f"{spectra_dir}/coh.npy", coh)

    if nnmf_components is not None:
        # Calculate NNMF and save
        nnmf = spectral.decompose_spectra(coh, n_components=nnmf_components)
        save(f"{spectra_dir}/nnmf_{nnmf_components}.npy", nnmf)


def nnmf(data, output_dir, n_components):
    """Calculate non-negative matrix factorization (NNMF).

    This function expects spectra have already been calculated and are in:

    - :code:`<output_dir>/spectra`, which contains multitaper spectra.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    n_components : int
        Number of components to fit.
    """
    from osl_dynamics.analysis import spectral

    spectra_dir = output_dir + "/spectra"
    coh = load(f"{spectra_dir}/coh.npy")
    nnmf = spectral.decompose_spectra(coh, n_components=n_components)
    save(f"{spectra_dir}/nnmf_{n_components}.npy", nnmf)


def regression_spectra(data, output_dir, kwargs):
    """Calculate regression spectra.

    This function expects a model has already been trained and the following
    directories exist:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create the following directory:

    - :code:`<output_dir>/spectra`, which contains the post-hoc spectra.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    kwargs : dict
        Keyword arguments to pass to `analysis.spectral.regress_spectra
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/spectral/index.html#osl_dynamics.analysis.spectral\
        .regression_spectra>`_. Defaults to::

            {'sampling_frequency': data.sampling_frequency,
             'window_length': 4 * sampling_frequency,
             'step_size': 20,
             'n_sub_windows': 8,
             'return_coef_int': True,
             'keepdims': True}
    """
    if data is None:
        raise ValueError("data must be passed.")

    sampling_frequency = kwargs.pop("sampling_frequency", None)
    if sampling_frequency is None and data.sampling_frequency is None:
        raise ValueError(
            "sampling_frequency must be passed or specified in the Data object."
        )
    else:
        sampling_frequency = data.sampling_frequency

    default_kwargs = {
        "sampling_frequency": sampling_frequency,
        "window_length": int(4 * sampling_frequency),
        "step_size": 20,
        "n_sub_windows": 8,
        "return_coef_int": True,
        "keepdims": True,
    }
    kwargs = override_dict_defaults(default_kwargs, kwargs)
    _logger.info(f"Using kwargs: {kwargs}")

    # Directories
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"
    spectra_dir = output_dir + "/spectra"
    os.makedirs(spectra_dir, exist_ok=True)

    # Load the inferred mixing coefficients
    alpha = load(f"{inf_params_dir}/alp.pkl")

    # Get the config used to create the model
    from osl_dynamics.models.mod_base import ModelBase

    model_config, _ = ModelBase.load_config(model_dir)

    # Get unprepared data (i.e. the data before calling Data.prepare)
    # We also trim the data to account for the data points lost to
    # time embedding or applying a sliding window
    data = data.trim_time_series(
        sequence_length=model_config["sequence_length"], prepared=False
    )

    # Calculate regression spectra
    from osl_dynamics.analysis import spectral

    spectra = spectral.regression_spectra(data=data, alpha=alpha, **kwargs)

    # Unpack spectra and save
    return_weights = kwargs.pop("return_weights", False)
    if return_weights:
        f, psd, coh, w = spectra
        save(f"{spectra_dir}/f.npy", f)
        save(f"{spectra_dir}/psd.npy", psd)
        save(f"{spectra_dir}/coh.npy", coh)
        save(f"{spectra_dir}/w.npy", w)
    else:
        f, psd, coh = spectra
        save(f"{spectra_dir}/f.npy", f)
        save(f"{spectra_dir}/psd.npy", psd)
        save(f"{spectra_dir}/coh.npy", coh)


def plot_group_ae_networks(
        data,
        output_dir,
        mask_file=None,
        parcellation_file=None,
        aec_abs=True,
        power_save_kwargs=None,
        conn_save_kwargs=None,
):
    """Plot group-level amplitude envelope networks.

    This function expects a model has been trained and the following directory
    to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/networks`, which contains plots of the networks.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    mask_file : str, optional
        Mask file used to preprocess the training data. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str, optional
        Parcellation file used to parcellate the training data. If
        :code:`None`, we use :code:`data.parcellation_file`.
    aec_abs : bool, optional
        Should we take the absolute value of the amplitude envelope
        correlations?
    power_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'filename': '<output_dir>/networks/mean_.png',
             'mask_file': data.mask_file,
             'parcellation_file': data.parcellation_file,
             'plot_kwargs': {'symmetric_cbar': True}}
    conn_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.connectivity.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/connectivity/index.html#osl_dynamics.analysis.connectivity\
        .save>`_. Defaults to::

            {'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/aec_.png',
             'threshold': 0.97}
    """
    power_save_kwargs = {} if power_save_kwargs is None else power_save_kwargs
    conn_save_kwargs = {} if conn_save_kwargs is None else conn_save_kwargs

    # Validation
    if mask_file is None:
        if data is None or data.mask_file is None:
            raise ValueError(
                "mask_file must be passed or specified in the Data object."
            )
        else:
            mask_file = data.mask_file

    if parcellation_file is None:
        if data is None or data.parcellation_file is None:
            raise ValueError(
                "parcellation_file must be passed or specified in the Data object."
            )
        else:
            parcellation_file = data.parcellation_file

    # Directories
    inf_params_dir = output_dir + "/inf_params"
    networks_dir = output_dir + "/networks"
    os.makedirs(networks_dir, exist_ok=True)

    # Load inferred means and covariances
    means = load(f"{inf_params_dir}/means.npy")
    covs = load(f"{inf_params_dir}/covs.npy")
    aecs = array_ops.cov2corr(covs)
    if aec_abs:
        aecs = abs(aecs)

    # Save mean activity maps
    from osl_dynamics.analysis import power

    default_power_save_kwargs = {
        "filename": f"{networks_dir}/mean_.png",
        "mask_file": mask_file,
        "parcellation_file": parcellation_file,
        "plot_kwargs": {"symmetric_cbar": True},
    }
    if "plot_kwargs" in power_save_kwargs:
        power_save_kwargs["plot_kwargs"] = override_dict_defaults(
            default_power_save_kwargs["plot_kwargs"],
            power_save_kwargs["plot_kwargs"],
        )
    power_save_kwargs = override_dict_defaults(
        default_power_save_kwargs, power_save_kwargs
    )
    _logger.info(f"Using power_save_kwargs: {power_save_kwargs}")
    power.save(means, **power_save_kwargs)

    # Save AEC networks
    from osl_dynamics.analysis import connectivity

    default_conn_save_kwargs = {
        "parcellation_file": parcellation_file,
        "filename": f"{networks_dir}/aec_.png",
        "threshold": 0.97,
    }
    conn_save_kwargs = override_dict_defaults(
        default_conn_save_kwargs, conn_save_kwargs
    )
    _logger.info(f"Using conn_save_kwargs: {conn_save_kwargs}")
    connectivity.save(aecs, **conn_save_kwargs)


def plot_group_tde_hmm_networks(
        data,
        output_dir,
        mask_file=None,
        parcellation_file=None,
        frequency_range=None,
        percentile=97,
        power_save_kwargs=None,
        conn_save_kwargs=None,
):
    """Plot group-level TDE-HMM networks for a specified frequency band.

    This function will:

    1. Plot state PSDs.
    2. Plot the power maps.
    3. Plot coherence networks.

    This function expects spectra have already been calculated and are in:

    - :code:`<output_dir>/spectra`, which contains multitaper spectra.

    This function will create:

    - :code:`<output_dir>/networks`, which contains plots of the networks.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    mask_file : str, optional
        Mask file used to preprocess the training data. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str, optional
        Parcellation file used to parcellate the training data. If
        :code:`None`, we use :code:`data.parcellation_file`.
    frequency_range : list, optional
        List of length 2 containing the minimum and maximum frequency to
        integrate spectra over. Defaults to the full frequency range.
    percentile : float, optional
        Percentile for thresholding the coherence networks. Default is 97, which
        corresponds to the top 3% of edges (relative to the mean across states).
    power_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'mask_file': mask_file,
             'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/pow_.png',
             'subtract_mean': True,
             'plot_kwargs': {'symmetric_cbar': True}}
    conn_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.connectivity.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/connectivity/index.html#osl_dynamics.analysis.connectivity\
        .save>`_. Defaults to::

            {'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/coh_.png',
             'plot_kwargs': {'edge_cmap': 'Reds'}}
    """
    power_save_kwargs = {} if power_save_kwargs is None else power_save_kwargs
    conn_save_kwargs = {} if conn_save_kwargs is None else conn_save_kwargs

    # Validation
    if mask_file is None:
        if data is None or data.mask_file is None:
            raise ValueError(
                "mask_file must be passed or specified in the Data object."
            )
        else:
            mask_file = data.mask_file

    if parcellation_file is None:
        if data is None or data.parcellation_file is None:
            raise ValueError(
                "parcellation_file must be passed or specified in the Data object."
            )
        else:
            parcellation_file = data.parcellation_file

    # Directories
    spectra_dir = output_dir + "/spectra"
    networks_dir = output_dir + "/networks"
    os.makedirs(networks_dir, exist_ok=True)

    # Load spectra
    f = load(f"{spectra_dir}/f.npy")
    psd = load(f"{spectra_dir}/psd.npy")
    coh = load(f"{spectra_dir}/coh.npy")
    if Path(f"{spectra_dir}/w.npy").exists():
        w = load(f"{spectra_dir}/w.npy")
    else:
        w = None

    # Calculate group average
    gpsd = np.average(psd, axis=0, weights=w)
    gcoh = np.average(coh, axis=0, weights=w)

    # Calculate average PSD across channels and the standard error
    p = np.mean(gpsd, axis=-2)
    e = np.std(gpsd, axis=-2) / np.sqrt(gpsd.shape[-2])

    # Plot PSDs
    from osl_dynamics.utils import plotting

    n_states = gpsd.shape[0]
    for i in range(n_states):
        fig, ax = plotting.plot_line(
            [f],
            [p[i]],
            errors=[[p[i] - e[i]], [p[i] + e[i]]],
            labels=[f"State {i + 1}"],
            x_range=[f[0], f[-1]],
            y_range=[p.min() - 0.1 * p.max(), 1.2 * p.max()],
            x_label="Frequency (Hz)",
            y_label="PSD (a.u.)",
        )
        if frequency_range is not None:
            ax.axvspan(
                frequency_range[0],
                frequency_range[1],
                alpha=0.25,
                color="gray",
            )
        plotting.save(fig, filename=f"{networks_dir}/psd_{i}.png")

    # Calculate power maps from the group-level PSDs
    from osl_dynamics.analysis import power

    gp = power.variance_from_spectra(f, gpsd, frequency_range=frequency_range)

    # Save power maps
    default_power_save_kwargs = {
        "mask_file": mask_file,
        "parcellation_file": parcellation_file,
        "filename": f"{networks_dir}/pow_.png",
        "subtract_mean": True,
        "plot_kwargs": {"symmetric_cbar": True},
    }
    if "plot_kwargs" in power_save_kwargs:
        power_save_kwargs["plot_kwargs"] = override_dict_defaults(
            default_power_save_kwargs["plot_kwargs"],
            power_save_kwargs["plot_kwargs"],
        )
    power_save_kwargs = override_dict_defaults(
        default_power_save_kwargs, power_save_kwargs
    )
    _logger.info(f"Using power_save_kwargs: {power_save_kwargs}")
    power.save(gp, **power_save_kwargs)

    # Calculate coherence networks from group-level spectra
    from osl_dynamics.analysis import connectivity

    gc = connectivity.mean_coherence_from_spectra(
        f, gcoh, frequency_range=frequency_range
    )

    # Threshold
    gc = connectivity.threshold(gc, percentile=percentile, subtract_mean=True)

    # Save coherence networks
    default_conn_save_kwargs = {
        "parcellation_file": parcellation_file,
        "filename": f"{networks_dir}/coh_.png",
        "plot_kwargs": {"edge_cmap": "Reds"},
    }
    conn_save_kwargs = override_dict_defaults(
        default_conn_save_kwargs, conn_save_kwargs
    )
    _logger.info(f"Using conn_save_kwargs: {conn_save_kwargs}")
    connectivity.save(gc, **conn_save_kwargs)


def plot_group_nnmf_tde_hmm_networks(
        data,
        output_dir,
        nnmf_file,
        mask_file=None,
        parcellation_file=None,
        component=0,
        percentile=97,
        power_save_kwargs=None,
        conn_save_kwargs=None,
):
    """Plot group-level TDE-HMM networks using a NNMF component to integrate
    the spectra.

    This function will:

    1. Plot state PSDs.
    2. Plot the power maps.
    3. Plot coherence networks.

    This function expects spectra have already been calculated and are in:

    - :code:`<output_dir>/spectra`, which contains multitaper spectra.

    This function will create:

    - :code:`<output_dir>/networks`, which contains plots of the networks.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    nnmf_file : str
        Path relative to :code:`output_dir` for a npy file (with the output of
        `analysis.spectral.decompose_spectra
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/spectral/index.html#osl_dynamics.analysis.spectral\
        .decompose_spectra>`_) containing the NNMF components.
    mask_file : str, optional
        Mask file used to preprocess the training data. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str, optional
        Parcellation file used to parcellate the training data. If
        :code:`None`, we use :code:`data.parcellation_file`.
    component : int, optional
        NNMF component to plot. Defaults to the first component.
    percentile : float, optional
        Percentile for thresholding the coherence networks. Default is 97, which
        corresponds to the top 3% of edges (relative to the mean across states).
    power_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'mask_file': mask_file,
             'parcellation_file': parcellation_file,
             'component': component,
             'filename': '<output_dir>/networks/pow_.png',
             'subtract_mean': True,
             'plot_kwargs': {'symmetric_cbar': True}}
    conn_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.connectivity.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/connectivity/index.html#osl_dynamics.analysis.connectivity\
        .save>`_. Defaults to::

            {'parcellation_file': parcellation_file,
             'component': component,
             'filename': '<output_dir>/networks/coh_.png',
             'plot_kwargs': {'edge_cmap': 'Reds'}}
    """
    power_save_kwargs = {} if power_save_kwargs is None else power_save_kwargs
    conn_save_kwargs = {} if conn_save_kwargs is None else conn_save_kwargs

    # Validation
    if mask_file is None:
        if data is None or data.mask_file is None:
            raise ValueError(
                "mask_file must be passed or specified in the Data object."
            )
        else:
            mask_file = data.mask_file

    if parcellation_file is None:
        if data is None or data.parcellation_file is None:
            raise ValueError(
                "parcellation_file must be passed or specified in the Data object."
            )
        else:
            parcellation_file = data.parcellation_file

    # Directories
    spectra_dir = output_dir + "/spectra"
    networks_dir = output_dir + "/networks"
    os.makedirs(networks_dir, exist_ok=True)

    # Load the NNMF components
    nnmf_file = output_dir + "/" + nnmf_file
    if Path(nnmf_file).exists():
        nnmf = load(nnmf_file)
    else:
        raise ValueError(f"{nnmf_file} not found.")

    # Load spectra
    f = load(f"{spectra_dir}/f.npy")
    psd = load(f"{spectra_dir}/psd.npy")
    coh = load(f"{spectra_dir}/coh.npy")
    if Path(f"{spectra_dir}/w.npy").exists():
        w = load(f"{spectra_dir}/w.npy")
    else:
        w = None

    # Plot the NNMF components
    from osl_dynamics.utils import plotting

    n_components = nnmf.shape[0]
    plotting.plot_line(
        [f] * n_components,
        nnmf,
        labels=[f"Component {i}" for i in range(n_components)],
        x_label="Frequency (Hz)",
        y_label="Weighting",
        filename=f"{networks_dir}/nnmf.png",
    )

    # Calculate group average
    gpsd = np.average(psd, axis=0, weights=w)
    gcoh = np.average(coh, axis=0, weights=w)

    # Calculate average PSD across channels and the standard error
    p = np.mean(gpsd, axis=-2)
    e = np.std(gpsd, axis=-2) / np.sqrt(gpsd.shape[-2])

    # Plot PSDs
    n_states = gpsd.shape[0]
    for i in range(n_states):
        fig, ax = plotting.plot_line(
            [f],
            [p[i]],
            errors=[[p[i] - e[i]], [p[i] + e[i]]],
            labels=[f"State {i + 1}"],
            x_range=[f[0], f[-1]],
            y_range=[p.min() - 0.1 * p.max(), 1.2 * p.max()],
            x_label="Frequency (Hz)",
            y_label="PSD (a.u.)",
        )
        plotting.save(fig, filename=f"{networks_dir}/psd_{i}.png")

    # Calculate power maps from the group-level PSDs
    from osl_dynamics.analysis import power

    gp = power.variance_from_spectra(f, gpsd, nnmf)

    # Save power maps
    default_power_save_kwargs = {
        "mask_file": mask_file,
        "parcellation_file": parcellation_file,
        "component": component,
        "filename": f"{networks_dir}/pow_.png",
        "subtract_mean": True,
        "plot_kwargs": {"symmetric_cbar": True},
    }
    if "plot_kwargs" in power_save_kwargs:
        power_save_kwargs["plot_kwargs"] = override_dict_defaults(
            default_power_save_kwargs["plot_kwargs"],
            power_save_kwargs["plot_kwargs"],
        )
    power_save_kwargs = override_dict_defaults(
        default_power_save_kwargs, power_save_kwargs
    )
    _logger.info(f"Using power_save_kwargs: {power_save_kwargs}")
    power.save(gp, **power_save_kwargs)

    # Calculate coherence networks from group-level spectra
    from osl_dynamics.analysis import connectivity

    gc = connectivity.mean_coherence_from_spectra(f, gcoh, nnmf)

    # Threshold
    gc = connectivity.threshold(gc, percentile=percentile, subtract_mean=True)

    # Save coherence networks
    default_conn_save_kwargs = {
        "parcellation_file": parcellation_file,
        "component": component,
        "filename": f"{networks_dir}/coh_.png",
        "plot_kwargs": {"edge_cmap": "Reds"},
    }
    conn_save_kwargs = override_dict_defaults(
        default_conn_save_kwargs, conn_save_kwargs
    )
    _logger.info(f"Using conn_save_kwargs: {conn_save_kwargs}")
    connectivity.save(gc, **conn_save_kwargs)


def plot_group_tde_dynemo_networks(
        data,
        output_dir,
        mask_file=None,
        parcellation_file=None,
        frequency_range=None,
        percentile=97,
        power_save_kwargs=None,
        conn_save_kwargs=None,
):
    """Plot group-level TDE-DyNeMo networks for a specified frequency band.

    This function will:

    1. Plot mode PSDs.
    2. Plot the power maps.
    3. Plot coherence networks.

    This function expects spectra have already been calculated and are in:

    - :code:`<output_dir>/spectra`, which contains regression spectra.

    This function will create:

    - :code:`<output_dir>/networks`, which contains plots of the networks.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    mask_file : str, optional
        Mask file used to preprocess the training data. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str, optional
        Parcellation file used to parcellate the training data. If
        :code:`None`, we use :code:`data.parcellation_file`.
    frequency_range : list, optional
        List of length 2 containing the minimum and maximum frequency to
        integrate spectra over. Defaults to the full frequency range.
    percentile : float, optional
        Percentile for thresholding the coherence networks. Default is 97, which
        corresponds to the top 3% of edges (relative to the mean across states).
    plot_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'mask_file': mask_file,
             'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/pow_.png',
             'subtract_mean': True,
             'plot_kwargs': {'symmetric_cbar': True}}
    conn_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.connectivity.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/connectivity/index.html#osl_dynamics.analysis.connectivity\
        .save>`_. Defaults to::

            {'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/coh_.png',
             'plot_kwargs': {'edge_cmap': 'Reds'}}
    """
    power_save_kwargs = {} if power_save_kwargs is None else power_save_kwargs
    conn_save_kwargs = {} if conn_save_kwargs is None else conn_save_kwargs

    # Validation
    if mask_file is None:
        if data is None or data.mask_file is None:
            raise ValueError(
                "mask_file must be passed or specified in the Data object."
            )
        else:
            mask_file = data.mask_file

    if parcellation_file is None:
        if data is None or data.parcellation_file is None:
            raise ValueError(
                "parcellation_file must be passed or specified in the Data object."
            )
        else:
            parcellation_file = data.parcellation_file

    # Directories
    spectra_dir = output_dir + "/spectra"
    networks_dir = output_dir + "/networks"
    os.makedirs(networks_dir, exist_ok=True)

    # Load spectra
    f = load(f"{spectra_dir}/f.npy")
    psd = load(f"{spectra_dir}/psd.npy")
    coh = load(f"{spectra_dir}/coh.npy")
    if Path(f"{spectra_dir}/w.npy").exists():
        w = load(f"{spectra_dir}/w.npy")
    else:
        w = None

    # Only keep the regression coefficients
    psd = psd[:, 0]

    # Calculate group average
    gpsd = np.average(psd, axis=0, weights=w)
    gcoh = np.average(coh, axis=0, weights=w)

    # Calculate average PSD across channels and the standard error
    p = np.mean(gpsd, axis=-2)
    e = np.std(gpsd, axis=-2) / np.sqrt(gpsd.shape[-2])

    # Plot PSDs
    from osl_dynamics.utils import plotting

    n_modes = gpsd.shape[0]
    for i in range(n_modes):
        fig, ax = plotting.plot_line(
            [f],
            [p[i]],
            errors=[[p[i] - e[i]], [p[i] + e[i]]],
            labels=[f"Mode {i + 1}"],
            x_range=[f[0], f[-1]],
            y_range=[p.min() - 0.1 * p.max(), 1.4 * p.max()],
            x_label="Frequency (Hz)",
            y_label="PSD (a.u.)",
        )
        if frequency_range is not None:
            ax.axvspan(
                frequency_range[0],
                frequency_range[1],
                alpha=0.25,
                color="gray",
            )
        plotting.save(fig, filename=f"{networks_dir}/psd_{i}.png")

    # Calculate power maps from the group-level PSDs
    from osl_dynamics.analysis import power

    gp = power.variance_from_spectra(f, gpsd, frequency_range=frequency_range)

    # Save power maps
    default_power_save_kwargs = {
        "mask_file": mask_file,
        "parcellation_file": parcellation_file,
        "filename": f"{networks_dir}/pow_.png",
        "subtract_mean": True,
        "plot_kwargs": {"symmetric_cbar": True},
    }
    if "plot_kwargs" in power_save_kwargs:
        power_save_kwargs["plot_kwargs"] = override_dict_defaults(
            default_power_save_kwargs["plot_kwargs"],
            power_save_kwargs["plot_kwargs"],
        )
    power_save_kwargs = override_dict_defaults(
        default_power_save_kwargs, power_save_kwargs
    )
    _logger.info(f"Using power_save_kwargs: {power_save_kwargs}")
    power.save(gp, **power_save_kwargs)

    # Calculate coherence networks from group-level spectra
    from osl_dynamics.analysis import connectivity

    gc = connectivity.mean_coherence_from_spectra(
        f, gcoh, frequency_range=frequency_range
    )

    # Threshold
    gc = connectivity.threshold(gc, percentile=percentile, subtract_mean=True)

    # Save coherence networks
    default_conn_save_kwargs = {
        "parcellation_file": parcellation_file,
        "filename": f"{networks_dir}/coh_.png",
        "plot_kwargs": {"edge_cmap": "Reds"},
    }
    conn_save_kwargs = override_dict_defaults(
        default_conn_save_kwargs, conn_save_kwargs
    )
    _logger.info(f"Using conn_save_kwargs: {conn_save_kwargs}")
    connectivity.save(gc, **conn_save_kwargs)


def plot_alpha(
        data,
        output_dir,
        session=0,
        normalize=False,
        sampling_frequency=None,
        kwargs=None,
):
    """Plot inferred alphas.

    This is a wrapper for `utils.plotting.plot_alpha
    <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/utils\
    /plotting/index.html#osl_dynamics.utils.plotting.plot_alpha>`_.

    This function expects a model has been trained and the following directory
    to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/alphas`, which contains plots of the inferred alphas.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    session : int, optional
        Index for session to plot. If 'all' is passed we create a separate plot
        for each session.
    normalize : bool, optional
        Should we also plot the alphas normalized using the trace of the
        inferred covariance matrices? Useful if we are plotting the inferred
        alphas from DyNeMo.
    sampling_frequency : float, optional
        Sampling frequency in Hz. If :code:`None`, we see if it is
        present in :code:`data.sampling_frequency`.
    kwargs : dict, optional
        Keyword arguments to pass to `utils.plotting.plot_alpha
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /utils/plotting/index.html#osl_dynamics.utils.plotting.plot_alpha>`_.
        Defaults to::

            {'sampling_frequency': data.sampling_frequency,
             'filename': '<output_dir>/alphas/alpha_*.png'}
    """
    if sampling_frequency is None and data is not None:
        sampling_frequency = data.sampling_frequency

    # Directories
    inf_params_dir = output_dir + "/inf_params"
    alphas_dir = output_dir + "/alphas"
    os.makedirs(alphas_dir, exist_ok=True)

    # Load inferred alphas
    alp = load(f"{inf_params_dir}/alp.pkl")
    if isinstance(alp, np.ndarray):
        alp = [alp]

    # Plot
    from osl_dynamics.utils import plotting

    default_kwargs = {
        "sampling_frequency": sampling_frequency,
        "filename": f"{alphas_dir}/alpha_*.png",
    }
    kwargs = override_dict_defaults(default_kwargs, kwargs)
    _logger.info(f"Using kwargs: {kwargs}")

    if session == "all":
        for i in range(len(alp)):
            kwargs["filename"] = f"{alphas_dir}/alpha_{i}.png"
            plotting.plot_alpha(alp[i], **kwargs)
    else:
        kwargs["filename"] = f"{alphas_dir}/alpha_{session}.png"
        plotting.plot_alpha(alp[session], **kwargs)

    if normalize:
        from osl_dynamics.inference import modes

        # Calculate normalised alphas
        covs = load(f"{inf_params_dir}/covs.npy")
        norm_alp = modes.reweight_alphas(alp, covs)

        # Plot
        if session == "all":
            for i in range(len(alp)):
                kwargs["filename"] = f"{alphas_dir}/norm_alpha_{i}.png"
                plotting.plot_alpha(norm_alp[i], **kwargs)
        else:
            kwargs["filename"] = f"{alphas_dir}/norm_alpha_{session}.png"
            plotting.plot_alpha(norm_alp[session], **kwargs)


def calc_gmm_alpha(data, output_dir, kwargs=None):
    """Binarize inferred alphas using a two-component GMM.

    This function expects a model has been trained and the following directory
    to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create the following file:

    - :code:`<output_dir>/inf_params/gmm_alp.pkl`, which contains the binarized
      alphas.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    kwargs : dict, optional
        Keyword arguments to pass to `inference.modes.gmm_time_courses
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /inference/modes/index.html#osl_dynamics.inference.modes\
        .gmm_time_courses>`_.
    """
    kwargs = {} if kwargs is None else kwargs
    inf_params_dir = output_dir + "/inf_params"

    # Load inferred alphas
    alp_file = f"{inf_params_dir}/alp.pkl"
    if not Path(alp_file).exists():
        raise ValueError(f"{alp_file} missing.")
    alp = load(alp_file)

    # Binarise using a two-component GMM
    from osl_dynamics.inference import modes

    _logger.info(f"Using kwargs: {kwargs}")
    gmm_alp = modes.gmm_time_courses(alp, **kwargs)
    save(f"{inf_params_dir}/gmm_alp.pkl", gmm_alp)


def plot_hmm_network_summary_stats(
        data,
        output_dir,
        use_gmm_alpha=False,
        sampling_frequency=None,
        sns_kwargs=None,
):
    """Plot HMM summary statistics for networks as violin plots.

    This function will plot the distribution over sessions for the following
    summary statistics:

    - Fractional occupancy.
    - Mean lifetime (s).
    - Mean interval (s).
    - Switching rate (Hz).

    This function expects a model has been trained and the following directory
    to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/summary_stats`, which contains plots of the summary
      statistics.

    The :code:`<output_dir>/summary_stats` directory will also contain numpy
    files with the summary statistics.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    use_gmm_alpha : bool, optional
        Should we use alphas binarised using a Gaussian mixture model?
        This function assumes :code:`calc_gmm_alpha` has been called and the
        file :code:`<output_dir>/inf_params/gmm_alp.pkl` exists.
    sampling_frequency : float, optional
        Sampling frequency in Hz. If :code:`None`, we use
        :code:`data.sampling_frequency`.
    sns_kwargs : dict, optional
        Arguments to pass to :code:`sns.violinplot()`.
    """
    if sampling_frequency is None:
        if data is None or data.sampling_frequency is None:
            raise ValueError(
                "sampling_frequency must be passed or specified in the Data object."
            )
        else:
            sampling_frequency = data.sampling_frequency

    # Directories
    inf_params_dir = output_dir + "/inf_params"
    summary_stats_dir = output_dir + "/summary_stats"
    os.makedirs(summary_stats_dir, exist_ok=True)

    from osl_dynamics.inference import modes

    if use_gmm_alpha:
        # Use alphas that were binarised using a GMM
        gmm_alp_file = f"{inf_params_dir}/gmm_alp.pkl"
        if Path(gmm_alp_file).exists():
            stc = load(gmm_alp_file)
        else:
            raise ValueError(f"{gmm_alp_file} missing.")

    else:
        # Load inferred alphas and hard classify
        alp = load(f"{inf_params_dir}/alp.pkl")
        if isinstance(alp, np.ndarray):
            raise ValueError(
                "We must train on multiple sessions to plot the distribution "
                "of summary statistics."
            )
        stc = modes.argmax_time_courses(alp)

    # Calculate summary stats
    fo = modes.fractional_occupancies(stc)
    lt = modes.mean_lifetimes(stc, sampling_frequency)
    intv = modes.mean_intervals(stc, sampling_frequency)
    sr = modes.switching_rates(stc, sampling_frequency)

    # Save summary stats
    save(f"{summary_stats_dir}/fo.npy", fo)
    save(f"{summary_stats_dir}/lt.npy", lt)
    save(f"{summary_stats_dir}/intv.npy", intv)
    save(f"{summary_stats_dir}/sr.npy", sr)

    # Plot
    from osl_dynamics.utils import plotting

    n_states = fo.shape[1]
    x = range(1, n_states + 1)
    plotting.plot_violin(
        fo.T,
        x=x,
        x_label="State",
        y_label="Fractional Occupancy",
        filename=f"{summary_stats_dir}/fo.png",
        sns_kwargs=sns_kwargs,
    )
    plotting.plot_violin(
        lt.T,
        x=x,
        x_label="State",
        y_label="Mean Lifetime (s)",
        filename=f"{summary_stats_dir}/lt.png",
        sns_kwargs=sns_kwargs,
    )
    plotting.plot_violin(
        intv.T,
        x=x,
        x_label="State",
        y_label="Mean Interval (s)",
        filename=f"{summary_stats_dir}/intv.png",
        sns_kwargs=sns_kwargs,
    )
    plotting.plot_violin(
        sr.T,
        x=x,
        x_label="State",
        y_label="Switching rate (Hz)",
        filename=f"{summary_stats_dir}/sr.png",
        sns_kwargs=sns_kwargs,
    )


def plot_dynemo_network_summary_stats(data, output_dir):
    """Plot DyNeMo summary statistics for networks as violin plots.

    This function will plot the distribution over sessions for the following
    summary statistics:

    - Mean (renormalised) mixing coefficients.
    - Standard deviation of (renormalised) mixing coefficients.

    This function expects a model has been trained and the following directories
    to exist:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/summary_stats`, which contains plots of the summary
      statistics.

    The :code:`<output_dir>/summary_stats` directory will also contain numpy
    files with the summary statistics.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    """
    # Directories
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"
    summary_stats_dir = output_dir + "/summary_stats"
    os.makedirs(summary_stats_dir, exist_ok=True)

    # Load inferred parameters
    alp = load(f"{inf_params_dir}/alp.pkl")
    if isinstance(alp, np.ndarray):
        raise ValueError(
            "We must train on multiple sessions to plot the distribution "
            "of summary statistics."
        )

    # Get the config used to create the model
    from osl_dynamics.models.mod_base import ModelBase

    config, _ = ModelBase.load_config(model_dir)

    # Renormalise (only if we are learning covariances)
    from osl_dynamics.inference import modes

    if config["learn_covariances"]:
        covs = load(f"{inf_params_dir}/covs.npy")
        alp = modes.reweight_alphas(alp, covs)

    # Calculate summary stats
    alp_mean = np.array([np.mean(a, axis=0) for a in alp])
    alp_std = np.array([np.std(a, axis=0) for a in alp])
    alp_corr = np.array([np.corrcoef(a, rowvar=False) for a in alp])
    for c in alp_corr:
        np.fill_diagonal(c, 0)  # remove diagonal to see the off-diagonals better

    # Save summary stats
    save(f"{summary_stats_dir}/alp_mean.npy", alp_mean)
    save(f"{summary_stats_dir}/alp_std.npy", alp_std)
    save(f"{summary_stats_dir}/alp_corr.npy", alp_corr)

    # Plot
    from osl_dynamics.utils import plotting

    n_modes = alp_mean.shape[1]
    x = range(1, n_modes + 1)
    plotting.plot_violin(
        alp_mean.T,
        x=x,
        x_label="Mode",
        y_label="Mean",
        filename=f"{summary_stats_dir}/alp_mean.png",
    )
    plotting.plot_violin(
        alp_std.T,
        x=x,
        x_label="Mode",
        y_label="Standard Deviation",
        filename=f"{summary_stats_dir}/alp_std.png",
    )
    plotting.plot_matrices(
        np.mean(alp_corr, axis=0), filename=f"{summary_stats_dir}/alp_corr.png"
    )


def compare_groups_hmm_summary_stats(
        data,
        output_dir,
        group2_indices,
        separate_tests=False,
        covariates=None,
        n_perm=1000,
        n_jobs=1,
        sampling_frequency=None,
):
    """Compare HMM summary statistics between two groups.

    This function expects a model has been trained and the following directory
    to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/group_diff`, which contains the summary statistics
      and plots.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    group2_indices : np.ndarray or list
        Indices indicating which sessions belong to the second group.
    separate_tests : bool, optional
        Should we perform a maximum statistic permutation test for each summary
        statistic separately?
    covariates : str, optional
        Path to a pickle file containing a :code:`dict` with covariances. Each
        item in the :code:`dict` must be the covariate name and value for each
        session. The covariates will be loaded with::

            from osl_dynamics.utils.misc import load
            covariates = load("/path/to/file.pkl")

        Example covariates::

            covariates = {"age": [...], "sex": [...]}
    n_perm : int, optional
        Number of permutations.
    n_jobs : int, optional
        Number of jobs for parallel processing.
    sampling_frequency : float, optional
        Sampling frequency in Hz. If :code:`None`, we use
        :code:`data.sampling_frequency`.
    """
    if sampling_frequency is None:
        if data is None or data.sampling_frequency is None:
            raise ValueError(
                "sampling_frequency must be passed or specified in the Data object."
            )
        else:
            sampling_frequency = data.sampling_frequency

    # Directories
    inf_params_dir = output_dir + "/inf_params"
    group_diff_dir = output_dir + "/group_diff"
    os.makedirs(group_diff_dir, exist_ok=True)

    # Get inferred state time courses
    from osl_dynamics.inference import modes

    alp = load(f"{inf_params_dir}/alp.pkl")
    stc = modes.argmax_time_courses(alp)

    # Calculate summary stats
    names = ["fo", "lt", "intv", "sr"]
    fo = modes.fractional_occupancies(stc)
    lt = modes.mean_lifetimes(stc, sampling_frequency)
    intv = modes.mean_intervals(stc, sampling_frequency)
    sr = modes.switching_rates(stc, sampling_frequency)
    sum_stats = np.swapaxes([fo, lt, intv, sr], 0, 1)

    # Save
    for i in range(4):
        save(f"{group_diff_dir}/{names[i]}.npy", sum_stats[:, i])

    # Create a vector for group assignments
    n_sessions = fo.shape[0]
    assignments = np.ones(n_sessions)
    assignments[group2_indices] += 1

    # Load covariates
    if covariates is not None:
        covariates = load(covariates)
    else:
        covariates = {}

    # Perform statistical significance testing
    from osl_dynamics.analysis import statistics

    if separate_tests:
        pvalues = []
        for i in range(4):
            # Calculate a statistical significance test for each
            # summary stat separately
            _, p = statistics.group_diff_max_stat_perm(
                sum_stats[:, i],
                assignments,
                n_perm=n_perm,
                covariates=covariates,
                n_jobs=n_jobs,
            )
            pvalues.append(p)
            _logger.info(f"{names[i]}: {np.sum(p < 0.05)} states have p-value<0.05")
            save(f"{group_diff_dir}/{names[i]}_pvalues.npy", p)
        pvalues = np.array(pvalues)
    else:
        # Calculate a statistical significance test for all
        # summary stats concatenated
        _, pvalues = statistics.group_diff_max_stat_perm(
            sum_stats,
            assignments,
            n_perm=n_perm,
            covariates=covariates,
            n_jobs=n_jobs,
        )
        for i in range(4):
            _logger.info(
                f"{names[i]}: {np.sum(pvalues[i] < 0.05)} states have p-value<0.05"
            )
            save(f"{group_diff_dir}/{names[i]}_pvalues.npy", pvalues[i])

    # Plot
    from osl_dynamics.utils import plotting

    labels = [
        "Fractional Occupancy",
        "Mean Lifetime (s)",
        "Mean Interval (s)",
        "Switching Rate (Hz)",
    ]
    for i in range(4):
        plotting.plot_summary_stats_group_diff(
            name=labels[i],
            summary_stats=sum_stats[:, i],
            pvalues=pvalues[i],
            assignments=assignments,
            filename=f"{group_diff_dir}/{names[i]}.png",
        )


def plot_burst_summary_stats(data, output_dir, sampling_frequency=None):
    """Plot burst summary statistics as violin plots.

    This function will plot the distribution over sessions for the following
    summary statistics:

    - Mean lifetime (s).
    - Mean interval (s).
    - Burst count (Hz).
    - Mean amplitude (a.u.).

    This function expects a model has been trained and the following
    directories to exist:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/summary_stats`, which contains plots of the summary
      statistics.

    The :code:`<output_dir>/summary_stats` directory will also contain numpy
    files with the summary statistics.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    sampling_frequency : float, optional
        Sampling frequency in Hz. If :code:`None`, we use
        :code:`data.sampling_frequency`.
    """
    if sampling_frequency is None:
        if data is None or data.sampling_frequency is None:
            raise ValueError(
                "sampling_frequency must be passed or specified in the Data object."
            )
        else:
            sampling_frequency = data.sampling_frequency

    # Directories
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"
    summary_stats_dir = output_dir + "/summary_stats"
    os.makedirs(summary_stats_dir, exist_ok=True)

    from osl_dynamics.inference import modes

    # Load state time course
    alp = load(f"{inf_params_dir}/alp.pkl")
    stc = modes.argmax_time_courses(alp)

    # Get the config used to create the model
    from osl_dynamics.models.mod_base import ModelBase

    model_config, _ = ModelBase.load_config(model_dir)

    # Get unprepared data (i.e. the data before calling Data.prepare)
    # We also trim the data to account for the data points lost to
    # time embedding or applying a sliding window
    data = data.trim_time_series(
        sequence_length=model_config["sequence_length"], prepared=False
    )

    # Calculate summary stats
    lt = modes.mean_lifetimes(stc, sampling_frequency)
    intv = modes.mean_intervals(stc, sampling_frequency)
    bc = modes.switching_rates(stc, sampling_frequency)
    amp = modes.mean_amplitudes(stc, data)

    # Save summary stats
    save(f"{summary_stats_dir}/lt.npy", lt)
    save(f"{summary_stats_dir}/intv.npy", intv)
    save(f"{summary_stats_dir}/bc.npy", bc)
    save(f"{summary_stats_dir}/amp.npy", amp)

    from osl_dynamics.utils import plotting

    # Plot
    n_states = lt.shape[1]
    plotting.plot_violin(
        lt.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Mean Lifetime (s)",
        filename=f"{summary_stats_dir}/fo.png",
    )
    plotting.plot_violin(
        intv.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Mean Interval (s)",
        filename=f"{summary_stats_dir}/intv.png",
    )
    plotting.plot_violin(
        bc.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Burst Count (Hz)",
        filename=f"{summary_stats_dir}/bc.png",
    )
    plotting.plot_violin(
        amp.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Mean Amplitude (a.u.)",
        filename=f"{summary_stats_dir}/amp.png",
    )
