from __future__ import print_function, division

import os
import logging
import numpy as np
import ruamel.yaml as yaml
from icecube import dataclasses


class DNNDataContainer(object):
    '''Class structure to hold DNN input data
    '''
    def __init__(self, batch_size=1):
        """Summary
        """
        self.config = {}
        self.batch_size = batch_size
        self._is_configured = False
        self._is_ready = False

    def configure(self, num_bins, relative_time_method, data_format,
                  time_bins=None,
                  time_quantiles=None,
                  autoencoder_settings=None,
                  autoencoder_name=None,
                  is_str_dom_format=False,
                  cascade_key=None,
                  pulse_key=None,
                  dom_exclusions=None,
                  partial_exclusion=None,
                  allowed_pulse_keys=None,
                  allowed_cascade_keys=None):
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
        is_str_dom_format : bool, optional
            If True, the data format is of shape [batch, 86, 60, num_bins].
            If False, the data is split up into the icecube part
            [batch, 10, 10, 60, num_bins] and the deepcore part
            [batch, 8, 60, num_bins].
        cascade_key : str, optional
            This parameter can be set to define the default cascade_key.
            The particle to use if the relative time method is 'vertex' or
            'first_light_at_dom'.
        pulse_key : str, optional
            This parameter can be set to define the default value.
            Name of pulses to use.
        dom_exclusions : list of str, optional
            This parameter can be set to define the default value.
            List of frame keys that define DOMs or TimeWindows that should be
            excluded. Typical values for this are:
            ['BrightDOMs','SaturationWindows',
             'BadDomsList', 'CalibrationErrata']
        partial_exclusion : bool, optional
            This parameter can be set to define the default value.
            If True, partially exclude DOMS, e.g. only omit pulses from
            excluded TimeWindows defined in 'dom_exclusions'.
            If False, all pulses from a DOM will be excluded if the omkey
            exists in the dom_exclusions.
        allowed_pulse_keys : None, optional
            This parameter can be set to define the default pulse_keys.
        allowed_cascade_keys : None, optional
            This parameter can be set to define the default cascade_keys.

        Raises
        ------
        ValueError
            Description
        """
        if self._is_configured:
            raise ValueError('Data container is already configured!')

        self.config['num_bins'] = num_bins
        self.config['relative_time_method'] = relative_time_method
        self.config['data_format'] = data_format
        if time_bins is None:
            self.config['time_bins'] = time_bins
        else:
            self.config['time_bins'] = [float(b) for b in time_bins]
        self.config['time_quantiles'] = time_quantiles
        self.config['autoencoder_settings'] = autoencoder_settings
        self.config['autoencoder_name'] = autoencoder_name
        self.config['is_str_dom_format'] = is_str_dom_format
        self.config['cascade_key'] = cascade_key
        self.config['pulse_key'] = pulse_key
        self.config['dom_exclusions'] = dom_exclusions
        self.config['partial_exclusion'] = partial_exclusion
        self.config['allowed_pulse_keys'] = allowed_pulse_keys
        self.config['allowed_cascade_keys'] = allowed_cascade_keys

        self._is_configured = True

    def load_configuration(self, model_path,
                           config_name='config_data_settings.yaml'):
        """Loads data container configuration for a given DNN model.

        Parameters
        ----------
        model_path : str
            A path to the model directory.
        config_name : str, optional
            The name of the configuration file in the model_path directory.
            This configuration file contains the data settings.

        Raises
        ------
        ValueError
            Description
        """
        if self._is_configured:
            raise ValueError('Data container is already configured!')

        config_file = os.path.join(model_path, config_name)
        with open(config_file, 'r') as stream:
            cfg_data = yaml.safe_load(stream)

        self.config['num_bins'] = cfg_data['num_bins']
        self.config['relative_time_method'] = cfg_data['relative_time_method']
        self.config['data_format'] = cfg_data['data_format']
        if cfg_data['time_bins'] is None:
            self.config['time_bins'] = cfg_data['time_bins']
        else:
            self.config['time_bins'] = [float(b) for b
                                        in cfg_data['time_bins']]
        self.config['time_quantiles'] = cfg_data['time_quantiles']
        self.config['autoencoder_settings'] = cfg_data['autoencoder_settings']
        self.config['autoencoder_name'] = cfg_data['autoencoder_name']

        # Backwards compatibility for older exported models which did not
        # include this setting. In this case the separated format, e.g.
        # icecube array + deepcore array is used as opposed to the string-dom
        # format: [batch, 86, 60, num_bins]
        if 'is_str_dom_format' in cfg_data:
            self.config['is_str_dom_format'] = cfg_data['is_str_dom_format']
        else:
            self.config['is_str_dom_format'] = False

        for key in ['cascade_key', 'pulse_key', 'dom_exclusions',
                    'partial_exclusion', 'allowed_pulse_keys',
                    'allowed_cascade_keys']:
            if key in cfg_data:
                self.config[key] = cfg_data[key]
            else:
                self.config[key] = None

        self._is_configured = True

    def set_up(self, pulse_key=None, dom_exclusions=None,
               partial_exclusion=True, cascade_key=None,
               check_settings=True):
        """Set up the container.

        Parameters
        ----------
        pulse_key : str, optional
            Name of pulses to use.
        dom_exclusions : list of str, optional
            List of frame keys that define DOMs or TimeWindows that should be
            excluded. Typical values for this are:
            ['BrightDOMs','SaturationWindows',
             'BadDomsList', 'CalibrationErrata']
        partial_exclusion : bool, optional
            If True, partially exclude DOMS, e.g. only omit pulses from
            excluded TimeWindows defined in 'dom_exclusions'.
            If False, all pulses from a DOM will be excluded if the omkey
            exists in the dom_exclusions.
        cascade_key : str, optional
            The particle to use if the relative time method is 'vertex' or
            'first_light_at_dom'.
        check_settings : bool, optional
            If True, the set values will be checked against configured or
            loaded settings if they were set. It these settings do not match
            an error will be raised. This ensures the correct use of the
            trained models.
            Sometimes it is necessary to use the model with slightly different
            settings. In this case 'check_settings' can be set to False.
            However, when doing so it can't be guaranteed that the model is
            used in a correct way.
            TLTR: use 'False' with caution.

        Raises
        ------
        ValueError
            If the pulse settings are checked and a mismatch is found.
        """
        if not self._is_configured:
            raise ValueError('Config for data container is not set up yet!')

        if self._is_ready:
            raise ValueError('Data container is already set up!')

        # go through settings and configure them if there is no mismatch
        params = {'pulse_key': pulse_key, 'dom_exclusions': dom_exclusions,
                  'partial_exclusion': partial_exclusion,
                  'cascade_key': cascade_key, 'check_settings': check_settings}
        for param in params:

            # if the value is not being skipped we can skip the rest
            if params[param] is None:
                continue

            # if it has not been configured yet, we can simply set the value
            if self.config[param] is None:
                self.config[param] = params[param]

            # The setting is already configured, now we need to check if
            # there is a mismatch
            else:
                if self.config[param] != params[param]:

                    # check for allowed pulse keys
                    if (param == 'pulse_key' and
                            self.config['allowed_pulse_keys'] is not None
                            and params[param]
                            in self.config['allowed_pulse_keys']):

                        # this is an allowed pulse, so set it
                        self.config[param] = params[param]
                        continue

                    # check for allowed cascade keys
                    if (param == 'cascade_key' and
                            self.config['allowed_cascade_keys'] is not None
                            and params[param]
                            in self.config['allowed_cascade_keys']):

                        # this is an allowed cascade key, so set it
                        self.config[param] = params[param]
                        continue

                    if check_settings:
                        msg = 'Fatal: parameter {!r} is set to {!r} which '
                        msg += 'differs from the model default value {!r}.'
                        raise ValueError(msg.format(
                            param, self.config[param], params[param]))
                    else:
                        msg = 'Warning: parameter {!r} is set to {!r} which '
                        msg += 'differs from the model default value {!r}. '
                        msg += 'Make sure this is what you intend to do!'
                        logging.warning(msg.format(
                            param, self.config[param], params[param]))
                        self.config[param] = params[param]

        # check if required variables are set [cascade_key isn't always needed]
        for param in ['pulse_key', 'dom_exclusions', 'partial_exclusion']:
            if self.config[param] is None:
                raise ValueError('Parameter {} must be set!'.format(param))

        # create data fields
        self.initialize()

        self._is_ready = True

    def initialize(self):
        """Initialize empty data fields.
        """

        # data fields that can be used as input into the network during
        # inference.
        if self.config['is_str_dom_format']:
            self.x_dom = np.zeros([self.batch_size, 86, 60,
                                   self.config['num_bins']])
        else:
            self.x_ic78 = np.zeros([self.batch_size, 10, 10, 60,
                                    self.config['num_bins']])
            self.x_deepcore = np.zeros([self.batch_size, 8, 60,
                                        self.config['num_bins']])
        self.global_time_offset_batch = np.zeros([self.batch_size])
        self.runtime_batch = np.zeros([self.batch_size])

        # data fields to hold I3Data to be written to frame
        # (This does not require batching because values are properly
        #  written to the frame for each event by the DNNContainerHandler.)
        self.bin_indices = dataclasses.I3MapKeyVectorInt()
        self.bin_values = dataclasses.I3MapKeyVectorDouble()
        self.global_time_offset = dataclasses.I3Double()
        self.runtime = dataclasses.I3Double()

    def is_ready(self):
        """Check if data container is ready.

        Returns
        -------
        Bool
            True, if data container is ready
        """
        return self._is_ready
