"""Model class for a generative model with Gaussian observations.

"""

import logging
from operator import lt
from typing import Tuple, Union

import numpy as np
from tensorflow import Variable
from tensorflow.keras import Model, layers, optimizers
from tqdm import trange
from vrad import models
from vrad.inference import callbacks, initializers, losses
from vrad.models.layers import (
    InferenceRNNLayers,
    LogLikelihoodLayer,
    MeansCovsLayer,
    MixMeansCovsLayer,
    ModelRNNLayers,
    NormalizationLayer,
    NormalKLDivergenceLayer,
    SampleNormalDistributionLayer,
    ThetaActivationLayer,
)
from vrad.utils.misc import check_arguments, replace_argument

_logger = logging.getLogger("VRAD")


class RIGO(models.GO):
    """RNN Inference/model network and Gaussian Observations (RIGO).

    Parameters
    ----------
    dimensions : vrad.models.config.Dimensions
        Dimensions of the data in the model.
    inference_network : vrad.models.config.RNN
        Inference network hyperparameters.
    model_network : vrad.models.config.RNN
        Model network hyperparameters.
    alpha : vrad.models.config.Alpha
        Parameters related to alpha.
    observation_model : vrad.models.config.ObservationModel
        Parameters related to the observation model.
    kl_annealing : vrad.models.config.KLAnnealing
        Parameters related to KL annealing.
    training : vrad.models.config.Training
        Parameters related to training a model.
    """

    def __init__(
        self,
        dimensions: models.config.Dimensions,
        inference_network: models.config.RNN,
        model_network: models.config.RNN,
        alpha: models.config.Alpha,
        observation_model: models.config.ObservationModel,
        kl_annealing: models.config.KLAnnealing,
        training: models.config.Training,
    ):
        # Settings
        self.inference_network = inference_network
        self.model_network = model_network
        self.alpha = alpha
        self.kl_annealing = kl_annealing

        # KL annealing
        self.kl_annealing_factor = Variable(0.0) if kl_annealing.do else Variable(1.0)

        # Initialise the observation model
        # This will inherit the base model, build and compile the model
        super().__init__(dimensions, observation_model, training)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(
            self.dimensions,
            self.inference_network,
            self.model_network,
            self.alpha,
            self.observation_model,
        )

    def compile(self):
        """Wrapper for the standard keras compile method."""

        # Loss function
        ll_loss = losses.ModelOutputLoss()
        kl_loss = losses.ModelOutputLoss(self.kl_annealing_factor)
        loss = [ll_loss, kl_loss]

        # Compile
        self.model.compile(optimizer=self.training.optimizer, loss=loss)

    def fit(
        self,
        *args,
        kl_annealing_callback=None,
        alpha_temperature_annealing_callback=None,
        use_tqdm=False,
        tqdm_class=None,
        use_tensorboard=None,
        tensorboard_dir=None,
        save_best_after=None,
        save_filepath=None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Adds callbacks and then trains the model.

        Parameters
        ----------
        kl_annealing_callback : bool
            Should we update the KL annealing factor during training?
        alpha_temperature_annealing_callback : bool
            Should we update the alpha temperature annealing factor during training?
        use_tqdm : bool
            Should we use a tqdm progress bar instead of the usual output from
            tensorflow.
        tqdm_class : tqdm
            Class for the tqdm progress bar.
        use_tensorboard : bool
            Should we use TensorBoard?
        tensorboard_dir : str
            Path to the location to save the TensorBoard log files.
        save_best_after : int
            Epoch number after which we should save the best model. The best model is
            that which achieves the lowest loss.
        save_filepath : str
            Path to save the best model to.

        Returns
        -------
        history
            The training history.
        """
        if use_tqdm:
            args, kwargs = replace_argument(self.model.fit, "verbose", 0, args, kwargs)

        additional_callbacks = []

        if kl_annealing_callback is None:
            kl_annealing_callback = self.kl_annealing.do

        if kl_annealing_callback:
            kl_annealing_callback = callbacks.KLAnnealingCallback(
                kl_annealing_factor=self.kl_annealing_factor,
                annealing_sharpness=self.kl_annealing.sharpness,
                n_epochs_annealing=self.kl_annealing.n_epochs,
            )
            additional_callbacks.append(kl_annealing_callback)

        if alpha_temperature_annealing_callback is None:
            alpha_temperature_annealing_callback = self.alpha.learn_temperature

        if alpha_temperature_annealing_callback:
            alpha_temperature_annealing_callback = (
                callbacks.AlphaTemperatureAnnealingCallback(
                    initial_alpha_temperature=self.alpha.initial_temperature,
                    final_alpha_temperature=self.alpha.final_temperature,
                    n_epochs_annealing=self.alpha.n_epochs_annealing,
                )
            )
            additional_callbacks.append(alpha_temperature_annealing_callback)

        args, kwargs = replace_argument(
            func=self.model.fit,
            name="callbacks",
            item=self.create_callbacks(
                use_tqdm,
                tqdm_class,
                use_tensorboard,
                tensorboard_dir,
                save_best_after,
                save_filepath,
                additional_callbacks,
            ),
            args=args,
            kwargs=kwargs,
            append=True,
        )

        return self.model.fit(*args, **kwargs)

    def reset_weight(self):
        """Reset the model as if you've built a new model.

        Resets the model weights, optimizer and annealing factor.
        """
        self.compile()
        initializers.reinitialize_model_weights(self.model)
        if self.kl_annealing.do:
            self.kl_annealing_factor.assign(0.0)
        if self.alpha.do_annealing:
            alpha_layer = self.model.get_layer("alpha")
            alpha_layer.alpha_temperature = self.alpha.initial_temperature

    def predict(self, *args, **kwargs) -> dict:
        """Wrapper for the standard keras predict method.

        Returns
        -------
        dict
            Dictionary with labels for each prediction.
        """
        predictions = self.model.predict(*args, *kwargs)
        return_names = ["ll_loss", "kl_loss", "alpha"]
        predictions_dict = dict(zip(return_names, predictions))
        return predictions_dict

    def predict_states(
        self, inputs, *args, concatenate: bool = False, **kwargs
    ) -> Union[list, np.ndarray]:
        """State mixing factors, alpha.

        Parameters
        ----------
        inputs : tensorflow.data.Dataset
            Prediction dataset.
        concatenate : bool
            Should we concatenate alpha for each subject? Optional, default
            is False.

        Returns
        -------
        np.ndarray
            State mixing factors with shape (n_subjects, n_samples,
            dimensions.n_states) or (n_samples, dimensions.n_states).
        """
        inputs = self._make_dataset(inputs)
        outputs = []
        for dataset in inputs:
            alpha = self.predict(dataset, *args, **kwargs)["alpha"]
            alpha = np.concatenate(alpha)
            outputs.append(alpha)
        if len(outputs) == 1:
            outputs = outputs[0]
        elif concatenate:
            outputs = np.concatenate(outputs)
        return outputs

    def losses(self, dataset, return_mean: bool = False) -> Tuple[float, float]:
        """Calculates the log-likelihood and KL loss for a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate losses for.
        return_mean : bool
            Should we return the mean loss over batches? Otherwise we return
            the sum. Optional, default is False.

        Returns
        -------
        ll_loss : float
            Negative log-likelihood loss.
        kl_loss : float
            KL divergence loss.
        """
        if return_mean:
            mean_or_sum = np.mean
        else:
            mean_or_sum = np.sum
        if isinstance(dataset, list):
            predictions = [self.predict(subject) for subject in dataset]
            ll_loss = mean_or_sum([mean_or_sum(p["ll_loss"]) for p in predictions])
            kl_loss = mean_or_sum([mean_or_sum(p["kl_loss"]) for p in predictions])
        else:
            predictions = self.predict(dataset)
            ll_loss = mean_or_sum(predictions["ll_loss"])
            kl_loss = mean_or_sum(predictions["kl_loss"])
        return ll_loss, kl_loss

    def free_energy(self, dataset, return_mean: bool = False) -> float:
        """Calculates the variational free energy of a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate the variational free energy for.
        return_mean : bool
            Should we return the mean free energy over batches? Otherwise
            we return the sum. Optional, default is False.

        Returns
        -------
        float
            Variational free energy for the dataset.
        """
        ll_loss, kl_loss = self.losses(dataset, return_mean=return_mean)
        free_energy = ll_loss + kl_loss
        return free_energy

    def burn_in(
        self,
        *args,
        learn_covariances: bool = False,
        learn_alpha_temperature: bool = False,
        **kwargs,
    ):
        """Burn-in training phase.

        Fits the model with means and covariances or alpha_temperature
        non-trainable.

        Parameters
        ----------
        learn_covariances : bool
            Should we learn the means and covariances during the burn-in training?
            Optional, default is False.
        learn_alpha_temperature : bool
            Should we learn the alpha temperature during the burn-in taining?
            Optional, default is False.
        """
        if check_arguments(args, kwargs, 3, "epochs", 1, lt):
            _logger.warning(
                "Number of burn-in epochs is less than 1. Skipping burn-in."
            )
            return

        # Make means and covariances non-trainable and compile
        if not learn_covariances:
            means_covs_layer = self.model.get_layer("means_covs")
            means_covs_layer.trainable = False
            self.compile()

        # Make alpha temperature non-trainable and compile
        if not learn_alpha_temperature:
            alpha_layer = self.model.get_layer("alpha_layer")
            alpha_layer.trainable = False
            self.compile()

        # Train the model
        self.fit(*args, **kwargs)

        # Make means and covariances trainable again and compile
        if not learn_covariances:
            means_covs_layer.trainable = True
            self.compile()

        # Make alpha temperature trainable again and compile
        if not learn_alpha_temperature:
            alpha_layer.trainable = True
            self.compile()

    def get_alpha_temperature(self) -> float:
        """Alpha temperature used in the model.

        Returns
        -------
        float
            Alpha temperature.
        """
        alpha_layer = self.model.get_layer("alpha")
        alpha_temperature = alpha_layer.alpha_temperature.numpy()
        return alpha_temperature

    def sample_alpha(self, n_samples: int) -> np.ndarray:
        """Uses the model RNN to sample state mixing factors, alpha.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.

        Returns
        -------
        np.ndarray
            Sampled alpha.
        """
        # Get layers
        model_rnn_layer = self.model.get_layer("mod_rnn")
        mod_mu_layer = self.model.get_layer("mod_mu")
        mod_sigma_layer = self.model.get_layer("mod_sigma")
        theta_norm_layer = self.model.get_layer("theta_norm")
        alpha_layer = self.model.get_layer("alpha")

        # Sequence of the underlying logits theta
        theta_norm = np.zeros(
            [self.dimensions.sequence_length, self.dimensions.n_states],
            dtype=np.float32,
        )

        # Normally distributed random numbers used to sample the logits theta
        epsilon = np.random.normal(
            0, 1, [n_samples + 1, self.dimensions.n_states]
        ).astype(np.float32)

        # Activate the first state for the first sample
        theta_norm[-1, 0] = 1

        # Sample the state fixing factors
        alpha = np.empty([n_samples, self.dimensions.n_states], dtype=np.float32)
        for i in trange(n_samples, desc="Sampling state time course", ncols=98):

            # If there are leading zeros we trim theta so that we don't pass the zeros
            trimmed_theta = theta_norm[~np.all(theta_norm == 0, axis=1)][
                np.newaxis, :, :
            ]

            # Predict the probability distribution function for theta one time step
            # in the future,
            # p(theta|theta_<t) ~ N(mod_mu, sigma_theta_jt)
            model_rnn = model_rnn_layer(trimmed_theta)
            mod_mu = mod_mu_layer(model_rnn)[0, -1]
            mod_sigma = mod_sigma_layer(model_rnn)[0, -1]
            mod_sigma = np.exp(mod_sigma)

            # Shift theta one time step to the left
            theta_norm = np.roll(theta_norm, -1, axis=0)

            # Sample from the probability distribution function
            theta = mod_mu + mod_sigma * epsilon[i]
            theta_norm[-1] = theta_norm_layer(theta[np.newaxis, np.newaxis, :])[0]

            # Calculate the state mixing factors
            alpha[i] = alpha_layer(theta_norm[-1][np.newaxis, np.newaxis, :])[0, 0]

        return alpha


def _model_structure(
    dimensions: models.config.Dimensions,
    inference_network: models.config.RNN,
    model_network: models.config.RNN,
    alpha: models.config.Alpha,
    observation_model: models.config.ObservationModel,
):
    """Model structure.

    Parameters
    ----------
    dimensions : vrad.models.config.Dimensions
        Dimensions of the data in the model.
    inference_network : vrad.models.config.RNN
        Inference network hyperparameters.
    model_network : vrad.models.config.RNN
        Model network hyperparameters.
    alpha : vrad.models.config.Alpha
        Parameters related to alpha.
    observation_model : vrad.models.config.ObservationModel
        Parameters related to the observation model.
    kl_annealing : vrad.models.config.KLAnnealing
        Parameters related to KL annealing.
    training : vrad.models.config.Training
        Parameters related to training a model.

    Returns
    -------
    tensorflow.keras.Model
        Keras model built using the functional API.
    """

    # Layer for input
    inputs = layers.Input(
        shape=(dimensions.sequence_length, dimensions.n_channels), name="data"
    )

    # Inference RNN:
    # - Learns q(theta) ~ N(theta | inf_mu, inf_sigma), where
    #     - inf_mu    ~ affine(RNN(inputs_<=t))
    #     - inf_sigma ~ softplus(RNN(inputs_<=t))

    # Definition of layers
    inference_input_dropout_layer = layers.Dropout(
        inference_network.dropout_rate, name="data_drop"
    )
    inference_output_layers = InferenceRNNLayers(
        inference_network.rnn,
        inference_network.normalization,
        inference_network.n_layers,
        inference_network.n_units,
        inference_network.dropout_rate,
        name="inf_rnn",
    )
    inf_mu_layer = layers.Dense(dimensions.n_states, name="inf_mu")
    inf_sigma_layer = layers.Dense(
        dimensions.n_states, activation="softplus", name="inf_sigma"
    )

    # Layers to sample theta from q(theta) and to convert to state mixing
    # factors alpha
    theta_layer = SampleNormalDistributionLayer(name="theta")
    theta_norm_layer = NormalizationLayer(alpha.theta_normalization, name="theta_norm")
    alpha_layer = ThetaActivationLayer(
        alpha.xform,
        alpha.initial_temperature,
        alpha.learn_temperature,
        name="alpha",
    )

    # Data flow
    inference_input_dropout = inference_input_dropout_layer(inputs)
    inference_output = inference_output_layers(inference_input_dropout)
    inf_mu = inf_mu_layer(inference_output)
    inf_sigma = inf_sigma_layer(inference_output)
    theta = theta_layer([inf_mu, inf_sigma])
    theta_norm = theta_norm_layer(theta)
    alpha = alpha_layer(theta_norm)

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each state as the observation model.
    # - We calculate the likelihood of generating the training data with alpha
    #   and the observation model.

    # Definition of layers
    means_covs_layer = MeansCovsLayer(
        dimensions.n_states,
        dimensions.n_channels,
        learn_means=False,
        learn_covariances=observation_model.learn_covariances,
        normalize_covariances=observation_model.normalize_covariances,
        initial_means=None,
        initial_covariances=observation_model.initial_covariances,
        name="means_covs",
    )
    mix_means_covs_layer = MixMeansCovsLayer(
        dimensions.n_states,
        dimensions.n_channels,
        observation_model.learn_alpha_scaling,
        name="mix_means_covs",
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    mu, D = means_covs_layer(inputs)  # inputs not used
    m, C = mix_means_covs_layer([alpha, mu, D])
    ll_loss = ll_loss_layer([inputs, m, C])

    # Model RNN:
    # - Learns p(theta_t |theta_<t) ~ N(theta_t | mod_mu, mod_sigma), where
    #     - mod_mu    ~ affine(RNN(theta_<t))
    #     - mod_sigma ~ softplus(RNN(theta_<t))

    # Definition of layers
    model_input_dropout_layer = layers.Dropout(
        model_network.dropout_rate, name="theta_norm_drop"
    )
    model_output_layers = ModelRNNLayers(
        model_network.rnn,
        model_network.normalization,
        model_network.n_layers,
        model_network.n_units,
        model_network.dropout_rate,
        name="mod_rnn",
    )
    mod_mu_layer = layers.Dense(dimensions.n_states, name="mod_mu")
    mod_sigma_layer = layers.Dense(
        dimensions.n_states, activation="softplus", name="mod_sigma"
    )
    kl_loss_layer = NormalKLDivergenceLayer(name="kl")

    # Data flow
    model_input_dropout = model_input_dropout_layer(theta_norm)
    model_output = model_output_layers(model_input_dropout)
    mod_mu = mod_mu_layer(model_output)
    mod_sigma = mod_sigma_layer(model_output)
    kl_loss = kl_loss_layer([inf_mu, inf_sigma, mod_mu, mod_sigma])

    return Model(inputs=inputs, outputs=[ll_loss, kl_loss, alpha])
