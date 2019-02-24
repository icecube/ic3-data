import tensorflow as tf
import numpy as np
import os
import glob


def tf_cov(x):
    """Calculate covariance for x.

    Equivalent to np.cov(x.T)

    """
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx


def tf_corrcoef(x):
    """Calculate correlation matrix for x.

    Equivalent to np.corrcoef(x.T)

    """
    mean, variance = tf.nn.moments(x, [0])
    x /= tf.sqrt(variance)
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    corr_xx = vx - mx
    return corr_xx


def gaussian_loss(y_true, y_pred, eps=1e-7, **kwargs):
    num_latent_vars = y_pred.get_shape().as_list()[-1] // 2

    y_vars = y_pred[:, :num_latent_vars]
    y_vars_unc = tf.abs(y_pred[:, num_latent_vars:]) + eps

    y_diff = y_true - y_vars

    loss = tf.reduce_mean(
                    2*tf.log(y_vars_unc) + (y_diff / y_vars_unc)**2,
                    )
    # loss = tf.reduce_mean(y_diff**2)
    # loss = tf.reduce_mean(
    #                 2*tf.log(y_vars_unc) + (tf.stop_gradient(y_diff)
    #                                         / y_vars_unc)**2,
    #                 )

    return loss


def skewness_loss(x):
    mean, variance = tf.nn.moments(x, axes=[0])
    tmp = (x - mean) / tf.sqrt(variance)
    skewness = tf.reduce_mean(tmp**3, axis=0)
    return tf.reduce_sum(tf.abs(skewness))


def mse_unc_loss(y_true, y_pred, eps=1e-3, **kwargs):
    num_latent_vars = y_pred.get_shape().as_list()[-1] // 2

    y_vars = y_pred[:, :num_latent_vars]
    y_vars_unc = tf.abs(y_pred[:, num_latent_vars:]) + eps

    y_diff = y_true - y_vars

    loss_reco = tf.reduce_mean(y_diff**2)
    loss_unc = tf.reduce_mean((tf.stop_gradient(y_diff) - y_vars_unc)**2)
    return loss_reco + loss_unc


def corr_loss(y_latent_vars):
    """Loss term correlation.
    """
    corr = tf_corrcoef(y_latent_vars)

    return tf.reduce_sum(tf.abs(corr))


def emd_loss(y_true, y_pred, reduction_axis=None, num_bins=327, **kwargs):
    """Earth Mover Distance between two waveforms

    Parameters
    ----------
    y_true : tf.Tensor
        A tensorflow tensor defining the true waveform
        shape: [batch_size, num_bins]
    y_pred : tf.Tensor
        A tensorflow tensor defining the true waveform
        shape: [batch_size, num_bins]

    Returns
    -------
    tf.tensor
        EMD between two waveforms.
        Shape: []
    """
    y_pred = tf.reshape(y_pred, [-1, num_bins])
    y_true = tf.reshape(y_true, [-1, num_bins])

    # set first element to 0
    emd_list = [tf.zeros_like(y_true[..., 0])]

    # walk through 1D histogram
    for i in range(num_bins):
        P_i = y_true[..., i]
        Q_i = y_pred[..., i]
        emd_list.append(P_i + emd_list[-1] - Q_i)

    # calculate sum
    emd_list = tf.stack(emd_list, axis=-1)
    emd = tf.reduce_sum(tf.abs(emd_list), axis=reduction_axis)
    return emd


def np_emd_loss(y_true, y_pred, reduction_axis=None, num_bins=327, **kwargs):
    """Earth Mover Distance between two waveforms

    Parameters
    ----------
    y_true : np.ndarray
        A tensorflow tensor defining the true waveform
        shape: [batch_size, num_bins]
    y_pred : np.ndarray
        A tensorflow tensor defining the true waveform
        shape: [batch_size, num_bins]

    Returns
    -------
    np.ndarray
        EMD between two waveforms.
        Shape: []
    """
    y_pred = np.reshape(y_pred, [-1, num_bins])
    y_true = np.reshape(y_true, [-1, num_bins])

    # set first element to 0
    emd_list = [np.zeros_like(y_true[..., 0])]

    # walk through 1D histogram
    for i in range(num_bins):
        P_i = y_true[..., i]
        Q_i = y_pred[..., i]
        emd_list.append(P_i + emd_list[-1] - Q_i)

    # calculate sum
    emd_list = np.stack(emd_list, axis=-1)
    emd = np.sum(np.abs(emd_list), axis=reduction_axis)
    return emd


class TimeseriesAutoencoder():

    def __init__(self, num_bins, num_latent_vars, encoder_layers,
                 decoder_layers):
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.num_bins = num_bins
        self.num_latent_vars = num_latent_vars

        self.custom_objects = {}

        # create empty dictionaries for different encoders
        self.encoders = {}
        self.encoders_pred = {}

    def _link_layers(self, layer_input, layers):
        x = layers[0](layer_input)
        for layer_i in layers[1:-1]:
            x = layer_i(x)
        x = layers[-1](x)
        return x

    def build(self, loss=emd_loss, optimizer='adam',
              mse_weight=2000,
              emd_weight=3,
              correlation_weight=100,
              skewness_weight=0,
              freeze_encoder=False,
              ):

        # create encoder layers
        input_layer = tf.keras.layers.Input(shape=(self.num_bins, 1))
        latent_layer = self._link_layers(layer_input=input_layer,
                                         layers=self.encoder_layers)

        # create decoder layers
        decoded_layer = self._link_layers(layer_input=latent_layer,
                                          layers=self.decoder_layers)

        # create models
        self.autoencoder = tf.keras.models.Model(input_layer, decoded_layer)

        # Encoder
        self.encoder = tf.keras.models.Model(input_layer, latent_layer)

        # Decoder
        decoder_input = tf.keras.layers.Input(shape=(self.num_latent_vars,))
        decoder_output = self._link_layers(layer_input=decoder_input,
                                           layers=self.decoder_layers)
        self.decoder = tf.keras.models.Model(decoder_input, decoder_output)

        if loss == 'mse_loss_with_correlation_penalty':
            def mse_loss_with_correlation_penalty(y_true, y_pred):
                loss = tf.losses.mean_squared_error(y_true, y_pred)
                corr_penalty = corr_loss(latent_layer)
                skewness_penalty = skewness_loss(latent_layer)

                # scale correlation penalty to be on same scale as loss
                # corr_penalty = corr_penalty * (
                #                             tf.stop_gradient(loss)
                #                             / tf.stop_gradient(corr_penalty))
                corr_penalty *= 1e-6
                skewness_penalty *= 1e-6

                return loss + corr_penalty + skewness_penalty
            loss = mse_loss_with_correlation_penalty
            self.custom_objects['mse_loss_with_correlation_penalty'] = \
                mse_loss_with_correlation_penalty

        elif loss == 'calibrated_emd_loss':
            def calibrated_emd_loss(y_true, y_pred, stddev=0.3):
                loss = emd_loss(y_true, y_pred, num_bins=self.num_bins) * 3
                corr_penalty = corr_loss(latent_layer) - self.num_latent_vars
                corr_penalty *= 5000

                # draw random values
                delta_values = tf.random_normal(
                                [tf.shape(y_pred)[0], self.num_latent_vars],
                                mean=0.0, stddev=stddev)
                delta_values = tf.clip_by_value(delta_values, -1., +1.)

                calib_lossses = []
                for i in range(self.num_latent_vars):
                    latent_layer_mod = tf.unstack(latent_layer, axis=-1)
                    latent_layer_mod[i] += delta_values[:, i]
                    latent_layer_mod = tf.stack(latent_layer_mod, axis=-1)

                    decoded_layer_mod = self._link_layers(
                                            layer_input=latent_layer_mod,
                                            layers=self.decoder_layers)
                    emd_diff = emd_loss(y_pred, decoded_layer_mod,
                                        reduction_axis=-1,
                                        num_bins=self.num_bins) / self.num_bins
                    # emd_diff = tf.Print(emd_diff, [emd_diff,
                    #                                tf.reduce_min(emd_diff),
                    #                                tf.reduce_max(emd_diff)])
                    calib_loss = (tf.abs(delta_values[:, i])
                                  - tf.abs(emd_diff))**2
                    calib_lossses.append(calib_loss)

                calibration_loss = tf.reduce_sum(tf.stack(calib_lossses,
                                                          axis=0))
                calibration_loss *= (self.num_bins / self.num_latent_vars)
                loss = tf.Print(loss, [loss, calibration_loss, corr_penalty])
                return loss + calibration_loss + corr_penalty

            loss = calibrated_emd_loss
            self.custom_objects['calibrated_emd_loss'] = \
                calibrated_emd_loss

        elif loss == 'mse_calibrated_emd_loss':
            def calibrated_emd_loss(y_true, y_pred, stddev=0.1):  # 0.02 for emd calibration shift
                loss = tf.reduce_sum((y_true - y_pred)**2) * 2000
                loss += emd_loss(y_true, y_pred, num_bins=self.num_bins) * 3
                corr_penalty = corr_loss(latent_layer) - self.num_latent_vars
                corr_penalty *= 1000  # 5000
                skewness_penalty = skewness_loss(latent_layer)
                skewness_penalty *= 1000  # 1000 for emd calibration shift

                # draw random values
                delta_values = tf.random_normal(
                                [tf.shape(y_pred)[0], self.num_latent_vars],
                                mean=0.0, stddev=stddev)
                # delta_values = tf.concat([tf.random_normal(
                #                             [tf.shape(y_pred)[0], 1],
                #                             mean=0.0, stddev=std)
                #                           for std in [0.1, 0.01, 0.01]],
                #                          axis=1)

                delta_values = tf.clip_by_value(delta_values, -1., +1.)

                calib_lossses = []
                for i in range(self.num_latent_vars):
                    latent_layer_mod = tf.unstack(latent_layer, axis=-1)
                    latent_layer_mod[i] += delta_values[:, i]
                    latent_layer_mod = tf.stack(latent_layer_mod, axis=-1)

                    decoded_layer_mod = self._link_layers(
                                            layer_input=latent_layer_mod,
                                            layers=self.decoder_layers)
                    emd_diff = emd_loss(y_pred, decoded_layer_mod,
                                        reduction_axis=-1,
                                        num_bins=self.num_bins) / self.num_bins
                    # emd_diff = tf.Print(emd_diff, [emd_diff,
                    #                                tf.reduce_min(emd_diff),
                    #                                tf.reduce_max(emd_diff)])
                    calib_loss = (tf.abs(delta_values[:, i])
                                  - tf.abs(emd_diff))**2
                    # calib_loss = tf.abs((tf.abs(delta_values[:, i])
                    #                      - tf.abs(mse_diff)))
                    calib_lossses.append(calib_loss)

                calibration_loss = tf.reduce_sum(tf.stack(calib_lossses,
                                                          axis=0))
                calibration_loss *= (self.num_bins / self.num_latent_vars)*100
                loss = tf.Print(loss, [loss, calibration_loss, corr_penalty,
                                       skewness_penalty])
                return loss + calibration_loss + corr_penalty + \
                    skewness_penalty

            loss = calibrated_emd_loss
            self.custom_objects['calibrated_emd_loss'] = \
                calibrated_emd_loss

        elif loss == 'calibrated_mse_loss':
            def calibrated_mse_loss(y_true, y_pred, stddev=0.01):
                loss = tf.reduce_sum((y_true - y_pred)**2)
                corr_penalty = corr_loss(latent_layer) - self.num_latent_vars
                skewness_penalty = skewness_loss(latent_layer)

                # draw random values
                delta_values = tf.random_normal(
                                [tf.shape(y_pred)[0], self.num_latent_vars],
                                mean=0.0, stddev=stddev)
                delta_values = tf.clip_by_value(delta_values, -1.0, +1.0)

                calib_lossses = []
                for i in range(self.num_latent_vars):
                    latent_layer_mod = tf.unstack(latent_layer, axis=-1)
                    latent_layer_mod[i] += delta_values[:, i]
                    latent_layer_mod = tf.stack(latent_layer_mod, axis=-1)

                    decoded_layer_mod = self._link_layers(
                                            layer_input=latent_layer_mod,
                                            layers=self.decoder_layers)
                    mse_diff = tf.reduce_sum((decoded_layer_mod - y_pred)**2,
                                             axis=(1, 2)) / 2.
                    # mse_diff = tf.Print(mse_diff, [mse_diff,
                    #                                tf.reduce_min(mse_diff),
                    #                                tf.reduce_max(mse_diff)])
                    calib_loss = (tf.abs(delta_values[:, i])
                                  - tf.abs(mse_diff))**2
                    # calib_loss = tf.abs((tf.abs(delta_values[:, i])
                    #                      - tf.abs(mse_diff)))
                    calib_lossses.append(calib_loss)

                calibration_loss = tf.reduce_sum(tf.stack(calib_lossses,
                                                          axis=0))
                calibration_loss *= (2. / self.num_latent_vars)
                loss = tf.Print(loss, [loss, calibration_loss, corr_penalty,
                                       skewness_penalty])
                return loss + calibration_loss + corr_penalty + \
                    skewness_penalty

            loss = calibrated_mse_loss
            self.custom_objects['calibrated_mse_loss'] = \
                calibrated_mse_loss

        elif loss == 'calibrated_emd_shuffle':

            def calibrated_emd_shuffle(y_true, y_pred):
                loss = tf.reduce_sum((y_true - y_pred)**2)
                loss *= mse_weight
                loss = tf.Print(loss, [loss], 'mse')
                loss += (emd_loss(y_true, y_pred, num_bins=self.num_bins)
                         * emd_weight)
                loss = tf.Print(loss, [loss], 'mse+emd')

                if correlation_weight == 0.:
                    corr_penalty = 0.
                else:
                    corr_penalty = corr_loss(latent_layer) - \
                                        self.num_latent_vars
                    corr_penalty *= correlation_weight

                if skewness_weight == 0.:
                    skewness_penalty = 0.
                else:
                    skewness_penalty = skewness_loss(latent_layer)
                    skewness_penalty *= skewness_weight  # 1000

                # randomly shuffle events (batches are shuffled: flip event)
                latent_layer_mod = latent_layer[::-1]
                # delta_values = tf.linalg.norm(latent_layer-latent_layer_mod,
                #                               axis=1)
                delta_values = tf.reduce_sum(
                                    tf.abs(latent_layer - latent_layer_mod),
                                    axis=1)
                decoded_layer_mod = self._link_layers(
                                            layer_input=latent_layer_mod,
                                            layers=self.decoder_layers)
                emd_diff = emd_loss(y_pred, decoded_layer_mod,
                                    reduction_axis=-1,
                                    num_bins=self.num_bins) / self.num_bins
                emd_diff = tf.Print(emd_diff, [emd_diff,
                                               tf.reduce_min(emd_diff),
                                               tf.reduce_max(emd_diff)])
                emd_diff = tf.Print(emd_diff, [delta_values,
                                               tf.reduce_min(delta_values),
                                               tf.reduce_max(delta_values)])
                calib_loss = (tf.abs(delta_values) - tf.abs(emd_diff))**2
                # calib_loss = tf.abs(tf.abs(delta_values) - tf.abs(emd_diff))

                calibration_loss = tf.reduce_sum(calib_loss)
                calibration_loss *= self.num_bins*1000

                loss = tf.Print(loss, [loss, calibration_loss, corr_penalty,
                                       skewness_penalty])
                return loss + calibration_loss + corr_penalty + \
                    skewness_penalty

            loss = calibrated_emd_shuffle
            self.custom_objects['calibrated_emd_shuffle'] = \
                calibrated_emd_shuffle

        elif loss == 'calibrated_mse_shuffle':
            def calibrated_mse_shuffle(y_true, y_pred):
                loss = tf.reduce_sum((y_true - y_pred)**2)
                corr_penalty = corr_loss(latent_layer) - self.num_latent_vars
                corr_penalty *= 1.
                skewness_penalty = skewness_loss(latent_layer)
                skewness_penalty *= 0.01

                # randomly shuffle events (batches are shuffled: flip event)
                latent_layer_mod = latent_layer[::-1]
                # delta_values = tf.linalg.norm(latent_layer-latent_layer_mod,
                #                               axis=1)
                delta_values = tf.reduce_sum(
                                    tf.abs(latent_layer - latent_layer_mod),
                                    axis=1)
                decoded_layer_mod = self._link_layers(
                                            layer_input=latent_layer_mod,
                                            layers=self.decoder_layers)
                mse_diff = tf.reduce_sum((decoded_layer_mod - y_pred)**2,
                                         axis=(1, 2))
                mse_diff = tf.Print(mse_diff, [mse_diff,
                                               tf.reduce_min(mse_diff),
                                               tf.reduce_max(mse_diff)])
                mse_diff = tf.Print(mse_diff, [delta_values,
                                               tf.reduce_min(delta_values),
                                               tf.reduce_max(delta_values)])
                calib_loss = (tf.abs(delta_values) - tf.sqrt(mse_diff))**2

                calibration_loss = tf.reduce_sum(calib_loss) * 10

                loss = tf.Print(loss, [loss, calibration_loss, corr_penalty,
                                       skewness_penalty])
                return loss + calibration_loss + corr_penalty + \
                    skewness_penalty

            loss = calibrated_mse_shuffle
            self.custom_objects['calibrated_mse_shuffle'] = \
                calibrated_mse_shuffle

        elif loss == 'calibrated_reduction':

            def calibrated_reduction(y_true, y_pred):
                assert mse_weight == 0 or emd_weight == 0

                if emd_weight != 0:
                    loss = emd_loss(y_true, y_pred, num_bins=self.num_bins)
                elif mse_weight != 0:
                    loss = tf.reduce_sum((y_true - y_pred)**2)
                loss *= max(mse_weight, emd_weight)

                if correlation_weight == 0.:
                    corr_penalty = 0.
                else:
                    corr_penalty = corr_loss(latent_layer) - \
                                        self.num_latent_vars
                    corr_penalty *= correlation_weight

                if skewness_weight == 0.:
                    skewness_penalty = 0.
                else:
                    skewness_penalty = skewness_loss(latent_layer)
                    skewness_penalty *= skewness_weight  # 1000

                # randomly shuffle events (batches are shuffled: flip event)
                delta_values = tf.reduce_sum(
                                    tf.abs(latent_layer - latent_layer[::-1]),
                                    axis=1)

                # EMD calibration loss
                if emd_weight == 0.:
                    calib_emd = 0.
                else:
                    emd_diff = emd_loss(y_true, y_true[::-1],
                                        reduction_axis=-1,
                                        num_bins=self.num_bins) / self.num_bins
                    calib_emd = (tf.abs(delta_values) - tf.abs(emd_diff))**2

                # MSE calibration loss
                if mse_weight == 0.:
                    calib_mse = 0.
                else:
                    mse_diff = tf.reduce_sum(
                                        (y_true - y_true[::-1])**2,
                                        axis=(1, 2))
                    calib_mse = (tf.abs(delta_values) - tf.abs(mse_diff))**2

                calib_loss = calib_emd + calib_mse
                calib_loss = tf.Print(calib_loss,
                                      [calib_loss,
                                       tf.reduce_min(calib_loss),
                                       tf.reduce_max(calib_loss)])
                calib_loss = tf.Print(calib_loss,
                                      [delta_values,
                                       tf.reduce_min(delta_values),
                                       tf.reduce_max(delta_values)])

                calibration_loss = tf.reduce_sum(calib_loss)*100

                loss = tf.Print(loss, [loss, calibration_loss,
                                       corr_penalty, skewness_penalty])
                return loss + calibration_loss + corr_penalty + \
                    skewness_penalty

            loss = calibrated_reduction
            self.custom_objects['calibrated_reduction'] = \
                calibrated_reduction

        if freeze_encoder:
            self.freeze_encoder()
        else:
            self.unfreeze_encoder()

        self.autoencoder.compile(optimizer=optimizer,
                                 loss=loss,
                                 metrics=[])
        return self

    def freeze_encoder(self, encoder_name=None):
        """Make weights of encoder untrainable.

        Parameters
        ----------
        encoder_name : str, optional
            Layers of the specified encoder will be made untrainable.
            If no name is given the maximum oversampling encoder will be used.
        """
        if encoder_name is None:
            encoder = self.encoder
        else:
            encoder = self.encoders[encoder_name]

        for layer in encoder.layers:
            layer.trainable = False

    def unfreeze_encoder(self, encoder_name=None):
        """Make weights of encoder trainable.

        Parameters
        ----------
        encoder_name : str, optional
            Layers of the specified encoder will be made trainable.
            If no name is given the maximum oversampling encoder will be used.
        """
        if encoder_name is None:
            encoder = self.encoder
        else:
            encoder = self.encoders[encoder_name]

        for layer in encoder.layers:
            layer.trainable = True

    def build_new_encoder(self, encoder_name,
                          encoder_layers,
                          encoder_unc_layers,
                          encoder_unc_input_index=-1,
                          copy_weights=False,
                          loss='gaussian',
                          optimizer='adam',
                          ):

        input_layer = tf.keras.layers.Input(shape=(self.num_bins, 1))

        # layers shared by both: prediction and uncertainty estimate
        shared_layer = self._link_layers(
                            layer_input=input_layer,
                            layers=encoder_layers[:encoder_unc_input_index])
        latent_layer = self._link_layers(
                            layer_input=shared_layer,
                            layers=encoder_layers[encoder_unc_input_index:])
        latent_unc_layer = self._link_layers(layer_input=shared_layer,
                                             layers=encoder_unc_layers)

        def combine_latent_layers(latent_layers):
            '''Combines latent layer prediction and uncertainties.
            '''
            latent_layer, latent_unc_layer = latent_layers
            return tf.concat([latent_layer, latent_unc_layer], axis=1)

        combined_latent_layer = tf.keras.layers.Lambda(
                                        combine_latent_layers,
                                        output_shape=(2*self.num_latent_vars,))
        encoder_output_layer = combined_latent_layer([latent_layer,
                                                      latent_unc_layer])

        # plain encoder
        self.encoders_pred[encoder_name] = tf.keras.models.Model(
                                            input_layer, latent_layer)

        self.encoders[encoder_name] = tf.keras.models.Model(
                                            input_layer, encoder_output_layer)

        if loss is not 'gaussian':
            raise NotImplementedError()

        self.custom_objects['gaussian'] = gaussian_loss

        self.encoders[encoder_name].compile(optimizer='adam',
                                            loss=gaussian_loss,
                                            metrics=[])

        if copy_weights:
            encoder_weights = self.encoder.get_weights()
            for l_old, l_new in zip(self.encoder.layers,
                                    self.encoders_pred[encoder_name].layers):
                l_new.set_weights(l_old.get_weights())

        return self

    def fit_encoder(self, encoder_name, X, y_latent_vars,
                    batch_size=32, validation_split=0.1, epochs=10,
                    **kwargs):
        """Fits the specified encoder model.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_observations, n_bins)
            The input time series data used to fit the autoencoder during
            training.
        y_latent_vars : numpy.ndarray, shape (n_observations, n_latent_vars)
            The input time series data used to fit the autoencoder during
            training.
        """
        X_in = np.expand_dims(X, axis=-1)
        return self.encoders[encoder_name].fit(
                                            X_in, y_latent_vars,
                                            batch_size=batch_size,
                                            validation_split=validation_split,
                                            epochs=epochs,
                                            **kwargs)

    def fit(self, X, y=None, batch_size=32, validation_split=0.1, epochs=10,
            **kwargs):
        """Fits the autoencoder model with the given training data X.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_observations, n_bins)
            The input time series data used to fit the autoencoder during
            training.
        """
        X = np.expand_dims(X, axis=-1)
        if y is None:
            y = X
        return self.autoencoder.fit(X, y,
                                    batch_size=batch_size,
                                    validation_split=validation_split,
                                    epochs=epochs,
                                    **kwargs)

    def encode(self, X, encoder_name=None, **kwargs):
        """Encodes the given time series data X.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_observations, n_bins)
            Time series data for which the features are to be extracted.

        """

        X = np.expand_dims(X, axis=-1)

        # choose correct encoder
        if encoder_name is None:
            encoder = self.encoder
        else:
            encoder = self.encoders[encoder_name]

        # Calculate latent variables
        return encoder.predict(X, **kwargs)

    def decode(self, X, **kwargs):
        """Turn feature dataframe into reconstructed time series.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_observations, n_latent_vars)
            Latent variables (encoded time series data) which will be
            decoded.

        """
        return self.decoder.predict(X, **kwargs)

    def save_model(self, model_dir):
        """Saves model to file.

        Parameters
        ----------
        model_dir : str
            Path to directory in which the model will be saved.

        """
        if not os.path.isdir(model_dir):
            print('Creating directory:', model_dir)
            os.makedirs(model_dir)

        autoencoder_file = os.path.join(model_dir, 'model_autoencoder.hdf5')
        self.autoencoder.save(autoencoder_file)

        for key, encoder in self.encoders.items():
            encoder_file = os.path.join(model_dir,
                                        'model_encoder_{}.hdf5'.format(key))
            encoder.save(encoder_file)

    def load_model(self, model_dir):
        """Loads model from file.

        Parameters
        ----------
        model_dir : str
            Path to directory from which the model will be loaded.

        """
        autoencoder_file = os.path.join(model_dir, 'model_autoencoder.hdf5')
        self.autoencoder = tf.keras.models.load_model(
                            autoencoder_file,
                            custom_objects=self.custom_objects)

        encoder_pattern = os.path.join(model_dir, 'model_encoder_*.hdf5')
        for encoder_file in glob.glob(encoder_pattern):
            key = encoder_file[14:-5]
            self.encoders[key] = tf.keras.models.load_model(
                            encoder_file,
                            custom_objects=self.custom_objects)

    def save_weights(self, model_dir):
        """Saves weights to file.

        Parameters
        ----------
        model_dir : str
            Path to directory in which the model will be saved.

        """
        if not os.path.isdir(model_dir):
            print('Creating directory:', model_dir)
            os.makedirs(model_dir)

        autoencoder_file = os.path.join(model_dir, 'weights_autoencoder.h5')
        self.autoencoder.save_weights(autoencoder_file)

        for key, encoder in self.encoders.items():
            encoder_file = os.path.join(model_dir,
                                        'weights_encoder_{}.h5'.format(key))
            encoder.save_weights(encoder_file)

    def load_weights(self, model_dir):
        """Loads weights from file.

        Parameters
        ----------
        model_dir : str
            Path to directory from which the model will be loaded.

        """
        autoencoder_file = os.path.join(model_dir, 'weights_autoencoder.h5')
        self.autoencoder.load_weights(autoencoder_file)

        encoder_pattern = os.path.join(model_dir, 'weights_encoder_*.h5')
        for encoder_file in glob.glob(encoder_pattern):
            key = encoder_file.split('/')[-1][16:-3]
            self.encoders[key].load_weights(encoder_file)
