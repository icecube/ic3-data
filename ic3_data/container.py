from __future__ import print_function, division

import numpy as np
from icecube import dataclasses


class DNNDataContainer(object):
    '''Class structure to hold DNN input data
    '''
    def __init__(self):
        """Summary
        """
        self.config = {}
        self._is_configured = False
        self._is_ready = False

    def configure(self, num_bins, relative_time_method, data_format,
                  time_bins=None,
                  time_quantiles=None,
                  autoencoder_settings=None,
                  autoencoder_name=None):
        """Summary

        Parameters
        ----------
        num_bins : int
            Number of bins for each DOM input.
        relative_time_method : str
            Defines method to use to calculate the time offset for the time
            binning.
            'cascade_vertex','time_range', None, 'first_light_at_dom'
        data_format : str
            Defines the data format:
                charge_weighted_time_quantiles:
                    First bin is total charge at DOM.
                    The rest of the bins will be filled with the times at wich
                    q% of the charge was measured at the DOM, where q stands
                    for the quantiles defined in time_quantiles.
                    Quantile of zero is interpreted as time of first pulse.
                charge_bins:
                    Charges are binned in time.
                    The binning is defined with the bins parameter.
                autoencoder:
                    Use an autoencoder to encode pulses.
        time_bins : List of float, optional
            A list defining the time bin edges if data_format is 'charge_bins'
            or 'autoencoder'.
        time_quantiles : None, optional
            List of quantiles for charge weighted time quantiles.
            A quantile of zero will be defined as the time of the first pulse.
            This is only needed when data format is
            'charge_weighted_time_quantiles'.
        autoencoder_settings : str, optional
            Settings for autoencoder.
        autoencoder_name : str, optional
            Name of encoder to use: e.g. wf_100, wf_1, ...
        """
        if self._is_configured:
            raise ValueError('Data container is already configured!')

        self.config['num_bins'] = num_bins
        self.config['relative_time_method'] = relative_time_method
        self.config['data_format'] = data_format
        self.config['time_bins'] = time_bins
        self.config['time_quantiles'] = time_quantiles
        self.config['autoencoder_settings'] = autoencoder_settings
        self.config['autoencoder_name'] = autoencoder_name

        self._is_configured = True

    def load_configuration(self, model_path):
        """Loads data container configuration for a given DNN model.

        Parameters
        ----------
        model_path : str
            A path to the model directory.

        Raises
        ------
        NotImplementedError
            Description
        """
        if self._is_configured:
            raise ValueError('Data container is already configured!')

        raise NotImplementedError()

        self._is_configured = True

    def set_up(self):

        if not self._is_configured:
            raise ValueError('Config for data container is not set up yet!')

        if self._is_ready:
            raise ValueError('Data container is already set up!')

        # create data fields
        self.initalize()

        self._is_ready = True

    def initalize(self):
        """Initialize empty data fields.
        """
        batch_size = 1
        self.x_ic78 = np.zeros(batch_size, 10, 10, 60, self.config['num_bins'])
        self.x_deepcore = np.zeros(batch_size, 8, 60, self.config['num_bins'])

        self.bin_indices = dataclasses.I3MapKeyVectorInt()
        self.bin_values = dataclasses.I3MapKeyVectorDouble()

    def is_ready(self):
        """Check if data container is ready.

        Returns
        -------
        Bool
            True, if data container is ready
        """
        return self._is_ready
