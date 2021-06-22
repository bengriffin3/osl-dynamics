"""Functions to generate spatial maps.

"""

import logging
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import plotting
from tqdm import trange
from vrad import array_ops, files

_logger = logging.getLogger("VRAD")


def state_power_maps(
    frequencies: np.ndarray,
    power_spectra: np.ndarray,
    components: np.ndarray = None,
    frequency_range: list = None,
) -> np.ndarray:
    """Calculates spatial power maps from power spectra.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis of the PSDs. Only used if frequency_range is given.
    power_spectra : np.ndarray
        Power/cross spectra for each channel. Shape is (n_states, n_channels,
        n_channels, n_f).
    components : np.ndarray
        Spectral components. Shape is (n_components, n_f). Optional.
    frequency_range : list
        Frequency range to integrate the PSD over (Hz). Optional: default is full
        range.

    Returns
    -------
    np.ndarray
        Power map for each component of each state. Shape is (n_components,
        n_states, n_channels, n_channels).
    """

    # Validation
    error_message = (
        "a 3D numpy array (n_channels, n_channels, n_frequency_bins) "
        + "or 4D numpy array (n_states, n_channels, n_channels, "
        + "n_frequency_bins) must be passed for spectra."
    )
    power_spectra = array_ops.validate(
        power_spectra,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )

    if components is not None and frequency_range is not None:
        raise ValueError(
            "Only one of the arguments components or frequency range can be passed."
        )

    if frequency_range is not None and frequencies is None:
        raise ValueError(
            "If frequency_range is passed, frequenices must also be passed."
        )

    # Number of subjects, states, channels and frequency bins
    n_subjects, n_states, n_channels, n_channels, n_f = power_spectra.shape

    # Remove cross-spectral densities from the power spectra array and concatenate
    # over subjects and states
    psd = power_spectra[:, :, range(n_channels), range(n_channels)].reshape(-1, n_f)

    # PSDs are real valued so we can recast
    psd = psd.real

    if components is not None:
        # Calculate PSD for each spectral component
        psd = components @ psd.T
        n_components = components.shape[0]
    else:
        # Integrate over the given frequency range
        if frequency_range is None:
            psd = np.sum(psd, axis=-1)
        else:
            f_min_arg = np.argwhere(frequencies > frequency_range[0])[0, 0]
            f_max_arg = np.argwhere(frequencies < frequency_range[1])[-1, 0]
            if f_max_arg < f_min_arg:
                raise ValueError("Cannot select the specified frequency range.")
            psd = np.sum(psd[..., f_min_arg : f_max_arg + 1], axis=-1)
        n_components = 1
    psd = psd.reshape(n_components, n_states, n_channels)

    # Power map
    p = np.zeros([n_components, n_states, n_channels, n_channels])
    p[:, :, range(n_channels), range(n_channels)] = psd

    return np.squeeze(p)


def spatial_map_grid(
    mask_file: str,
    parcellation_file: str,
    power_map: np.ndarray,
    component: int = 0,
    subtract_mean: bool = False,
    mean_weights: np.ndarray = None,
) -> np.ndarray:
    """Returns the power at locations on a spatial map grid."""

    # Validation
    error_message = f"Dimensionality of power_map must be 4, got ndim={power_map.ndim}."
    power_map = array_ops.validate(
        power_map,
        correct_dimensionality=4,
        allow_dimensions=[3],
        error_message=error_message,
    )

    # Load the mask
    mask = nib.load(mask_file)
    mask_grid = mask.get_data()

    # Flatten the mask
    mask_grid = mask_grid.ravel(order="F")

    # Get indices of non-zero elements, i.e. those which contain the brain
    non_zero_parcels = mask_grid != 0

    # Load the parcellation
    parcellation = nib.load(parcellation_file)
    parcellation_grid = parcellation.get_data()

    # Number of parcels
    n_parcels = parcellation.shape[-1]

    # Make a 1D array of voxel weights for each channel
    parcels = parcellation_grid.reshape(-1, n_parcels, order="F")[non_zero_parcels]

    # Number of parcels
    n_parcels = parcels.shape[0]

    # Normalise the parcels to have comparable weights
    parcels /= parcels.max(axis=0)[np.newaxis, ...]

    # Number of components, states, channels
    n_components, n_states, n_channels, n_channels = power_map.shape

    # Generate spatial map
    spatial_map = np.empty([n_parcels, n_states])
    for i in range(n_states):
        spatial_map[:, i] = parcels @ np.diag(np.squeeze(power_map[component, i]))

    # Subtract weighted mean
    if subtract_mean:
        spatial_map -= np.average(
            spatial_map,
            axis=1,
            weights=mean_weights,
        )[..., np.newaxis]

    # Convert spatial map into a grid
    spatial_map_grid = np.zeros([mask_grid.shape[0], n_states])
    spatial_map_grid[non_zero_parcels] = spatial_map
    spatial_map_grid = spatial_map_grid.reshape(
        mask.shape[0], mask.shape[1], mask.shape[2], n_states, order="F"
    )

    return spatial_map_grid


def save_nii(
    mask_file: str,
    parcellation_file: str,
    power_map: np.ndarray,
    filename: str,
    component: int = 0,
    subtract_mean: bool = False,
    mean_weights: np.ndarray = None,
):
    """Saves a NITFI file containing a map.

    Parameters
    ----------
    mask_file : str
        Mask file used to preprocess the training data.
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    power_map : np.ndarray
        Power map to save.
        Shape must be (n_components, n_states, n_channels, n_channels).
    filename : str
        Output file name.
    component : int
        Spectral component to save. Optional.
    subtract_mean : bool
        Should we subtract the mean power across states? Optional: default is False.
    mean_weights: np.ndarray
        Numpy array with weightings for each state to use to calculate the mean.
        Optional, default is equal weighting.
    """

    # If the mask file doesn't exist, check if it's in files.mask
    if not os.path.exists(mask_file):
        if os.path.exists(f"{files.mask.directory}/{mask_file}"):
            mask_file = f"{files.mask.directory}/{mask_file}"
        else:
            raise FileNotFoundError(mask_file)

    # If the parcellation file doesn't exist, check if  it's in files.parcellation
    if not os.path.exists(parcellation_file):
        if os.path.exists(f"{files.parcellation.directory}/{parcellation_file}"):
            parcellation_file = f"{files.parcellation.directory}/{parcellation_file}"
        else:
            raise FileNotFoundError(parcellation_file)

    # Calculate power maps
    power_map = spatial_map_grid(
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        power_map=power_map,
        component=component,
        subtract_mean=subtract_mean,
        mean_weights=mean_weights,
    )

    # Load the mask
    mask = nib.load(mask_file)

    # Save as nii file
    if "nii" not in filename:
        filename += ".nii.gz"
    print(f"Saving {filename}")
    nii = nib.Nifti1Image(power_map, mask.affine, mask.header)
    nib.save(nii, filename)


def save_images(
    mask_file: str,
    parcellation_file: str,
    power_map: np.ndarray,
    filename: str,
    component: int = 0,
    subtract_mean: bool = False,
    mean_weights: np.ndarray = None,
):
    """Saves power maps as an image file.

    Parameters
    ----------
    mask_file : str
        Mask file used to preprocess the training data.
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    power_map : np.ndarray
        Power map to save.
        Shape must be (n_components, n_states, n_channels, n_channels).
    filename : str
        Output file name.
    component : int
        Spectral component to save. Optional.
    subtract_mean : bool
        Should we subtract the mean power across states? Optional: default is False.
    mean_weights: np.ndarray
        Numpy array with weightings for each state to use to calculate the mean.
        Optional, default is equal weighting.
    """

    # If the mask file doesn't exist, check if it's in files.mask
    if not os.path.exists(mask_file):
        if os.path.exists(f"{files.mask.directory}/{mask_file}"):
            mask_file = f"{files.mask.directory}/{mask_file}"
        else:
            raise FileNotFoundError(mask_file)

    # If the parcellation file doesn't exist, check if  it's in files.parcellation
    if not os.path.exists(parcellation_file):
        if os.path.exists(f"{files.parcellation.directory}/{parcellation_file}"):
            parcellation_file = f"{files.parcellation.directory}/{parcellation_file}"
        else:
            raise FileNotFoundError(parcellation_file)

    # Calculate power maps
    power_map = spatial_map_grid(
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        power_map=power_map,
        component=component,
        subtract_mean=subtract_mean,
        mean_weights=mean_weights,
    )

    # Load the mask
    mask = nib.load(mask_file)

    # Save each map as an image
    n_states = power_map.shape[-1]
    for i in trange(n_states, desc="Saving images", ncols=98):
        nii = nib.Nifti1Image(power_map[:, :, :, i], mask.affine, mask.header)
        output_file = "{fn.stem}{i:0{w}d}{fn.suffix}".format(
            fn=Path(filename), i=i, w=len(str(n_states))
        )
        plotting.plot_img_on_surf(
            nii,
            views=["lateral", "medial"],
            hemispheres=["left", "right"],
            colorbar=True,
            output_file=output_file,
        )
