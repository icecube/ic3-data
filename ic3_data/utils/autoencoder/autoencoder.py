import tensorflow as tf
import numpy as np
import wf_autoencoder


def get_autoencoder(
        autoencoder_settings,
        model_dir='/data/user/mhuennefeld/DNN_reco/models/autoencoder/',
        ):

    # --------
    # settings
    # --------
    num_bins = 327
    conv_settings = {
            'filters': 10,
            'kernel_size': 20,
            'activation': 'elu',
            'padding': "same",
        }

    # homogenous autoencoder model with 2 latent vars
    if autoencoder_settings == 'homogenous_2_2':
        model_dir += 'wf_4006_all_positive_327'
        latent_dim = 2
        decoder_activation = 'relu'

    # inhomogenous autoencoder model with 3 latent vars
    elif autoencoder_settings == 'inhomogenous_3_3':
        model_dir += 'wf_1006_all_327'
        latent_dim = 3
        decoder_activation = 'relu'

    # inhomogenous autoencoder model with 3 latent vars
    elif autoencoder_settings == 'inhomogenous_3_3_skewness':
        model_dir += 'wf_1006_all_3_skewness_327'
        latent_dim = 3
        decoder_activation = 'elu'

    # inhomogenous autoencoder model with 3 latent vars
    elif autoencoder_settings == 'inhomogenous_3_3_calibrated_corr_emd':
        model_dir += 'wf_1006_all_3_calibrated_corr_327'
        latent_dim = 3
        decoder_activation = 'elu'

    # inhomogenous autoencoder model with 3 latent vars [EMD calibrated]
    elif autoencoder_settings == 'inhomogenous_3_3_calibrated_corr_emd_shift':
        model_dir += 'wf_1006_all_3_calibrated_corr_shift_327'
        latent_dim = 3
        decoder_activation = 'elu'

    # inhomogenous autoencoder model: 1 charge, 2 time,  2 l [EMD calibrated]
    elif autoencoder_settings == 'inhomogenous_2_2_calibrated_emd_wf_quantile':
        model_dir += 'wf_5007_wf_quantile_402'
        latent_dim = 2
        decoder_activation = 'elu'
        num_bins = 402

    # inhomogenous autoencoder model: 1 charge, 1 time,  2 l [EMD calibrated]
    elif autoencoder_settings == 'translational_3_MSE_602':
        model_dir += 'wf_5010_all_translational_3_MSE_602'
        latent_dim = 3
        decoder_activation = 'elu'
        num_bins = 602

    # inhomogenous autoencoder model with 2 latent vars
    elif autoencoder_settings == 'inhomogenous_2_2':
        model_dir += 'wf_1006_all_2_327'
        latent_dim = 2
        decoder_activation = 'elu'

    else:
        raise ValueError('Uknown autoencoder settings {!r}'.format(
                        autoencoder_settings))

    # --------

    encoder = [
         tf.keras.layers.BatchNormalization(axis=-1,
                                            input_shape=[num_bins, 1]),
         tf.keras.layers.Conv1D(**conv_settings),
         tf.keras.layers.Conv1D(**conv_settings),
         tf.keras.layers.MaxPool1D(pool_size=2),
         tf.keras.layers.Conv1D(**conv_settings),
         tf.keras.layers.MaxPool1D(pool_size=2),
         tf.keras.layers.Conv1D(**conv_settings),
         tf.keras.layers.MaxPool1D(pool_size=2),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dropout(rate=0.0),
         tf.keras.layers.Dense(50, activation='elu'),
         tf.keras.layers.Dense(latent_dim, activation=None),
        ]

    decoder = [
             tf.keras.layers.Dense(300, input_shape=[latent_dim],
                                   activation=decoder_activation),
             tf.keras.layers.Dropout(rate=0.00),
             tf.keras.layers.Dense(300, activation=decoder_activation),
             tf.keras.layers.Dense(num_bins, activation='softmax'),
             tf.keras.layers.Reshape([num_bins, 1]),
         ]

    encoder_wf100 = [
         tf.keras.layers.BatchNormalization(axis=-1,
                                            input_shape=[num_bins, 1]),
         tf.keras.layers.Conv1D(**conv_settings),
         tf.keras.layers.Conv1D(**conv_settings),
         tf.keras.layers.MaxPool1D(pool_size=2),
         tf.keras.layers.Conv1D(**conv_settings),
         tf.keras.layers.MaxPool1D(pool_size=2),
         tf.keras.layers.Conv1D(**conv_settings),
         tf.keras.layers.MaxPool1D(pool_size=2),
         tf.keras.layers.Dropout(rate=0.0),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(50, activation='elu'),
         tf.keras.layers.Dense(latent_dim, activation=None),
        ]

    encoder_unc_layers_wf100 = [
        tf.keras.layers.Dense(42, activation='elu', input_shape=[400, 1]),
        tf.keras.layers.Dense(latent_dim, activation='softplus'),
        ]

    encoder_wf1 = [
         tf.keras.layers.BatchNormalization(axis=-1,
                                            input_shape=[num_bins, 1]),
         tf.keras.layers.Conv1D(**conv_settings),
         tf.keras.layers.Conv1D(**conv_settings),
         tf.keras.layers.MaxPool1D(pool_size=2),
         tf.keras.layers.Conv1D(**conv_settings),
         tf.keras.layers.MaxPool1D(pool_size=2),
         tf.keras.layers.Conv1D(**conv_settings),
         tf.keras.layers.MaxPool1D(pool_size=2),
         tf.keras.layers.Dropout(rate=0.0),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(50, activation='elu'),
         tf.keras.layers.Dense(latent_dim, activation=None),
        ]

    encoder_unc_layers_wf1 = [
        tf.keras.layers.Dense(42, activation='elu', input_shape=[400, 1]),
        tf.keras.layers.Dense(latent_dim, activation='softplus'),
        ]

    autoencoder = wf_autoencoder.TimeseriesAutoencoder(
                    num_bins=num_bins,
                    num_latent_vars=latent_dim,
                    encoder_layers=encoder,
                    decoder_layers=decoder,
                    )
    autoencoder.build(loss='mse_loss_with_correlation_penalty')
    autoencoder.build_new_encoder(
                            encoder_name='wf_100',
                            encoder_layers=encoder_wf100,
                            encoder_unc_layers=encoder_unc_layers_wf100,
                            copy_weights=True,
                            encoder_unc_input_index=-2,
                            )
    autoencoder.build_new_encoder(
                            encoder_name='wf_1',
                            encoder_layers=encoder_wf1,
                            encoder_unc_layers=encoder_unc_layers_wf1,
                            copy_weights=True,
                            encoder_unc_input_index=-2,
                            )

    autoencoder.load_weights(model_dir)
    return autoencoder


def get_encoded_data(autoencoder, encoder_name, dom_times, dom_charges, bins,
                     autoencoder_settings, time_offset):

    hist, bin_edges = np.histogram(dom_times, weights=dom_charges, bins=bins)

    hist = np.expand_dims(hist, axis=0)
    total_dom_charge = np.sum(hist, axis=1)

    assert (total_dom_charge > 0).all()
    normed_hist = hist / total_dom_charge
    latent_vars = autoencoder.encode(normed_hist, encoder_name=encoder_name)[0]

    bin_values_list = []
    bin_indices_list = []

    # add total charge at DOM
    bin_values_list.append(float(total_dom_charge))
    bin_indices_list.append(0)

    # add latent variables
    for i, latent_var in enumerate(latent_vars):
        bin_values_list.append(float(latent_var))
        bin_indices_list.append(i+1)

    # Add time information
    if autoencoder_settings == 'inhomogenous_2_2_calibrated_emd_wf_quantile':
        index_offset = 1 + len(latent_vars)
        t_first_pulse = dom_times[0] + time_offset
        t_wf_quantile = time_offset
        bin_values_list.append(t_first_pulse)
        bin_indices_list.append(index_offset)
        bin_values_list.append(t_wf_quantile)
        bin_indices_list.append(index_offset + 1)

    # correct time information
    if autoencoder_settings == 'translational_3_MSE_602':
        assert bins[-2] - bins[1] == 3000 and len(bins) == 603
        bin_values_list[1] = time_offset + bins[1] + bin_values_list[1] * 3000
        bin_values_list[4] *= 3000

    return bin_values_list, bin_indices_list
