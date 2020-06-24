"""A series of inference-specific functions which are too broad to include in a model.

"""

import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def pseudo_sigma_to_sigma(pseudo_sigma):
    """Returns a legal non-singular covariance matrix from a 'pseudo_sigma' tensor."""
    # Convert flattened tensors into matrices
    pseudo_sigma = tfp.math.fill_triangular(pseudo_sigma)
    sigma = tf.matmul(pseudo_sigma, tf.transpose(pseudo_sigma, (0, 2, 1)))
    sigma = tf.add(
        sigma,
        1e-6
        * tf.eye(
            sigma.shape[1], batch_shape=[pseudo_sigma.shape[0]], dtype=sigma.dtype
        ),
    )
    return sigma


def normalise_covariance(covariance):
    """Normalise covariance matrix based on its trace.

    The trace of `covariance` is taken. All values are then divided by it.

    Parameters
    ----------
    covariance : tf.Tensor
        Tensor of the form [M x N x N]

    Returns
    -------
    normalised_covariance : tf.Tensor
        Tensor of the form [M x N x N]
    """
    normalisation = tf.reduce_sum(tf.linalg.diag_part(covariance), axis=1)[
        ..., tf.newaxis, tf.newaxis
    ]
    normalised_covariance = covariance / normalisation
    return normalised_covariance