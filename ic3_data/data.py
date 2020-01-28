from __future__ import print_function, division
import timeit
import numpy as np

from icecube import dataclasses, icetray

from ic3_data import misc
from ic3_data import data_formats
from ic3_data.ext_pybind11 import get_time_range
from ic3_data import ext_boost
from ic3_data.utils.time import get_wf_quantile
from ic3_data.utils.time import get_time_of_first_light
from ic3_data.utils import detector
from ic3_data.utils.autoencoder import autoencoder


class DNNContainerHandler(icetray.I3ConditionalModule):
    '''Module to fill DNNDataContainer and optionally add data to frame.

    Note: settings (apart from pulses) that will affect how the dnn input data
    is calculated must be added to the DNNDataContainer and NOT here.
    In addition, these settings must also be checked by dnn_reco's application
    modules. Hence, dnn_reco must also export these new settings when a model
    is exported.
    '''
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("DNNDataContainer",
                          "The DNN data container to be used. The container "
                          "will be filled for each Physics frame")
        self.AddParameter("OutputKey",
                          "If provided, the dnn data will be written to the "
                          "frame. In this case the following keys will be "
                          "created: "
                          "'OutputKey'+_bin_values, "
                          "'OutputKey'+_bin_indices, "
                          "'OutputKey'+_bin_values, "
                          "'OutputKey'+_global_time_offset, "
                          "and optionally 'OutputKey'+_settings, ",
                          None)
        self.AddParameter("Verbose",
                          "Print out detailled information.")

    def Configure(self):
        """Configure DNN data handler

        Raises
        ------
        ValueError
            Description
        """
        self._container = self.GetParameter("DNNDataContainer")
        self._output_key = self.GetParameter("OutputKey")
        self._verbose = self.GetParameter("Verbose")
        self._pulse_key = self._container.config['pulse_key']
        self._cascade_key = self._container.config['cascade_key']
        self._dom_exclusions = self._container.config['dom_exclusions']
        self._partial_exclusion = self._container.config['partial_exclusion']

        # initalize data fields of data container
        self._container.initialize()
        self._batch_index = 0

        self._config = dict(self._container.config)

        if not self._container.is_ready():
            raise ValueError('Container is not ready to be used!')

        if self._config['data_format'] == 'charge_weighted_time_quantiles':
            # sanity check
            for q in self._config['time_quantiles']:
                if q > 1. or q < 0.:
                    msg = 'Quantile should be within [0,1]:'
                    msg += ' {!r}'.format(q)
                    raise ValueError(msg)

        if self._config['autoencoder_settings'] == \
                'inhomogenous_2_2_calibrated_emd_wf_quantile':
            if self._config['_relative_time_method'] != 'wf_quantile':
                raise ValueError('Timing method must be wf_quantile!')

        if self._config['data_format'] == 'autoencoder':
            self._config['autoencoder'] = autoencoder.get_autoencoder(
                                        self._config['autoencoder_settings'])

        # Check if it is a data format that computes everything at once,
        # or if it is one that computes the values for one DOM at a time
        if self._config['data_format'] in [
                'cascade_classification_data',
                ]:
            self._calculate_per_dom = False
            class_string = 'ic3_data.data_formats_detector.{}'.format(
                                    self._config['data_format'])
        else:
            self._calculate_per_dom = True
            class_string = 'ic3_data.data_formats.{}'.format(
                                    self._config['data_format'])
        self._data_format_func = misc.load_class(class_string)
        self._is_str_dom_format = self._container.config['is_str_dom_format']

    def Geometry(self, frame):
        """Get a dictionary with DOM positions

        Parameters
        ----------
        frame : I3Frame
            Current i3 frame.
        """
        geo_map = frame['I3Geometry'].omgeo
        self._dom_pos_dict = {i[0]: i[1].position for i in geo_map
                              if i[1].omtype.name == 'IceCube'}
        self.PushFrame(frame)

    def Physics(self, frame):
        """Fill DNNDataContainer with data.

        Parameters
        ----------
        frame : I3Frame
            Current i3 frame.
        """
        # start timer
        start_time = timeit.default_timer()

        # initialize data fields of data container if new batch is started
        if self._batch_index == self._container.batch_size:
            self._container.initialize()
            self._batch_index = 0

        # get masked pulses
        if self._dom_exclusions is not None:
            pulses = ext_boost.get_valid_pulse_map(frame, self._pulse_key,
                                                   self._dom_exclusions,
                                                   self._partial_exclusion,
                                                   self._verbose)
            exclusion_om_keys = self._get_exclusion_om_keys(
                                            frame, self._dom_exclusions)
        else:
            pulses = frame[self._pulse_key]
            exclusion_om_keys = []

        if isinstance(pulses, dataclasses.I3RecoPulseSeriesMapMask) or \
           isinstance(pulses, dataclasses.I3RecoPulseSeriesMapUnion):
            pulses = pulses.apply(frame)

        # ------------------------------------------------
        # Calculate DNN input data seperately for each DOM
        # ------------------------------------------------
        if self._calculate_per_dom:
            # restructure pulses
            # charges, times, dom_times_dict, dom_charges_dict = \
            #     self.restructure_pulses(pulses)
            charges, times, dom_times_dict, dom_charges_dict = \
                ext_boost.restructure_pulses(pulses)

            # get global time offset
            global_time_offset = self.get_global_time_offset(frame=frame,
                                                             charges=charges,
                                                             times=times)
            self._container.global_time_offset.value = global_time_offset
            self._container.global_time_offset_batch[self._batch_index] = \
                global_time_offset

            # loop through hit DOMs
            om_keys = exclusion_om_keys + dom_charges_dict.keys()
            for om_key in om_keys:

                if om_key in dom_charges_dict:
                    dom_charges = dom_charges_dict[om_key]
                    dom_times = dom_times_dict[om_key]
                else:
                    dom_charges = np.array([])
                    dom_times = np.array([])

                # calculate times relative to global time offset
                rel_dom_times = dom_times - global_time_offset

                # get local time offset for the DOM
                local_time_offset = self.get_local_time_offset(
                                                    frame=frame,
                                                    om_key=om_key,
                                                    dom_times=rel_dom_times,
                                                    dom_charges=dom_charges)

                # Calculate times relative to local offset
                rel_dom_times -= local_time_offset
                # total_time_offset = global_time_offset + local_time_offset

                # get DNN input data for this DOM
                bin_values_list, bin_indices_list, bin_exclusions_list = \
                    self._data_format_func(
                                    dom_charges=dom_charges,
                                    rel_dom_times=rel_dom_times,
                                    global_time_offset=global_time_offset,
                                    local_time_offset=local_time_offset,
                                    config=self._config,
                                    dom_exclusions=self._dom_exclusions,
                                    partial_exclusion=self._partial_exclusion,
                                    frame=frame, om_key=om_key,
                                    )

                # Fill DNNDataContainer
                self._fill_dnn_container(om_key, bin_values_list,
                                         bin_indices_list, bin_exclusions_list)

        # ---------------------------------------------------
        # Calculate DNN input data for whole detector at once
        # ---------------------------------------------------
        else:

            global_time_offset, data_dict = self._data_format_func(
                                    frame=frame,
                                    pulses=pulses,
                                    config=self._config,
                                    dom_exclusions=self._dom_exclusions,
                                    partial_exclusion=self._partial_exclusion)

            # loop through data
            for om_key, (bin_values_list,
                         bin_indices_list,
                         bin_exclusions_list) in data_dict.items():

                # Fill DNNDataContainer
                self._fill_dnn_container(om_key, bin_values_list,
                                         bin_indices_list, bin_exclusions_list)

            # set global_time offset
            self._container.global_time_offset.value = global_time_offset
            self._container.global_time_offset_batch[self._batch_index] = \
                global_time_offset
        # ---------------------------------------------------

        # measure time
        elapsed_time = timeit.default_timer() - start_time
        self._container.runtime.value = elapsed_time
        self._container.runtime_batch[self._batch_index] = elapsed_time

        # Write data to frame
        if self._output_key is not None:
            frame[self._output_key + '_bin_exclusions'] = \
                self._container.bin_exclusions
            frame[self._output_key + '_bin_indices'] = \
                self._container.bin_indices
            frame[self._output_key + '_bin_values'] = \
                self._container.bin_values
            frame[self._output_key + '_global_time_offset'] = \
                self._container.global_time_offset
            frame[self._output_key + '_runtime'] = \
                self._container.runtime

        # increase the batch event index by one
        self._batch_index += 1

        self.PushFrame(frame)

    def _get_exclusion_om_keys(self, frame, dom_exclusions):
        """Get a list of OMKeys that correspond to DOMs that have exclusions.

        Parameters
        ----------
        frame : I3Frame
            Current I3Frame.
        dom_exclusions : list of str
            A list of keys for DOM exclusions.

        Returns
        -------
        list of OMKey
            A list of DOM OMkeys
        """
        om_keys = []
        for exclusion_key in dom_exclusions:
            if exclusion_key in frame:
                if isinstance(frame[exclusion_key],
                              dataclasses.I3VectorOMKey) or \
                        isinstance(frame[exclusion_key], list):
                    for om_key in frame[exclusion_key]:
                        om_keys.append(om_key)
                elif isinstance(frame[exclusion_key],
                                dataclasses.I3TimeWindowSeriesMap):
                    for om_key in frame[exclusion_key].keys():
                        om_keys.append(om_key)
                else:
                    msg = 'Unknown exlcusion type {!r} of key {!r}'
                    raise ValueError(msg.format(type(frame[exclusion_key]),
                                                exclusion_key))

        return om_keys

    def _fill_dnn_container(self, om_key, bin_values_list, bin_indices_list,
                            bin_exclusions_list):
        """Fill in container data for a specific DOM

        Parameters
        ----------
        om_key : I3OMKey
            The key of the DOM for which to fill in the data.
        bin_values_list : list of float
            The values of the bins.
        bin_indices_list : list of int
            The bin indices.
        bin_exclusions_list : list of int
            The list of bin indices which define bins that will be excluded.
        """
        string = om_key.string
        dom = om_key.om

        # Add bin values and indices if non-empty
        if bin_values_list:
            self._container.bin_values[om_key] = bin_values_list
            self._container.bin_indices[om_key] = bin_indices_list

            for value, index in zip(bin_values_list, bin_indices_list):
                if self._is_str_dom_format:
                    self._container.x_dom[self._batch_index,
                                          string - 1, dom - 1, index] = value
                else:
                    if string > 78:
                        # deep core
                        self._container.x_deepcore[self._batch_index,
                                                   string - 78 - 1, dom - 1,
                                                   index] = value
                    else:
                        # IC78
                        a, b = detector.string_hex_coord_dict[string]
                        # Center of Detector is a,b = 0,0
                        # a goes from -4 to 5
                        # b goes from -5 to 4
                        self._container.x_ic78[self._batch_index,
                                               a + 4, b + 5, dom - 1,
                                               index] = value

        # Add bin exclusions if non-empty
        if bin_exclusions_list:
            self._container.bin_exclusions[om_key] = bin_exclusions_list
            for index in bin_exclusions_list:
                if self._is_str_dom_format:
                    if index == -1:
                        # exclude complete DOM
                        self._container.x_dom_exclusions[
                            self._batch_index, string - 1,
                            dom - 1, :] = False
                    else:
                        self._container.x_dom_exclusions[
                            self._batch_index, string - 1,
                            dom - 1, index] = False
                else:
                    if string > 78:
                        # deep core
                        if index == -1:
                            # exclude complete DOM
                            self._container.x_deepcore_exclusions[
                                self._batch_index, string - 78 - 1,
                                dom - 1, :] = False
                        else:
                            self._container.x_deepcore_exclusions[
                                self._batch_index, string - 78 - 1,
                                dom - 1, index] = False
                    else:
                        # IC78
                        a, b = detector.string_hex_coord_dict[string]
                        # Center of Detector is a,b = 0,0
                        # a goes from -4 to 5
                        # b goes from -5 to 4
                        if index == -1:
                            # exclude complete DOM
                            self._container.x_ic78_exclusions[
                                self._batch_index,
                                a + 4, b + 5, dom - 1, :] = False
                        else:
                            self._container.x_ic78_exclusions[
                                self._batch_index,
                                a + 4, b + 5, dom - 1, index] = False

    def restructure_pulses(self, pulses):
        """Restructure pulse series information

        Parameters
        ----------
        pulses : dataclasses.I3RecoPulseSeries
            The pulses to restructure.

        Returns
        -------
        numpy.ndarray
            The charges of all pulses in the event.
        numpy.ndarray
            The (unsorted) times of all pulses in the event.
        dict of numpy.ndarray
            A dictionary that contains the pulse times for the pulses
            of a specific DOM as defined by the dictionary key.
        dict of numpy.ndarray
            A dictionary that contains the pulse charges for the pulses
            of a specific DOM as defined by the dictionary key.
        """
        charges = []
        times = []
        dom_times_dict = {}
        dom_charges_dict = {}
        for key, dom_pulses in pulses:
            dom_charges = []
            dom_times = []
            for pulse in dom_pulses:
                dom_charges.append(pulse.charge)
                dom_times.append(pulse.time)
            dom_times_dict[key] = np.asarray(dom_times)
            dom_charges_dict[key] = np.asarray(dom_charges)
            charges.extend(dom_charges)
            times.extend(dom_times)
        charges = np.asarray(charges)
        times = np.asarray(times)

        return charges, times, dom_times_dict, dom_charges_dict

    def get_global_time_offset(self, frame, charges, times):
        """Get global time offset for event.

        Parameters
        ----------
        charges : numpy.ndarray
            The pulse charges
        times : numpy.ndarray
            The pulse times.

        Raises
        ------
        ValueError
            If option is unkown.

        Returns
        -------
        float
            The global time offset for the event.
        """
        if self._config['relative_time_method'].lower() == 'cascade_vertex':
            global_time_offset = frame[self._cascade_key].time

        elif self._config['relative_time_method'].lower()[:41] == \
                'cascade_vertex_random_uniform_minus_plus_':
            """
            Create a global time offset by uniformly sampling around
            vertex time. The bounds are defined by the minus and plus
            arguments encoded in the relative_time_method name.
            These must be separeted by an underscore '_'.
            Example:
                cascade_vertex_random_uniform_minus_plus_200_500
                will sample from [vertex_time - 200, vertex_time + 500]
            """
            splits = \
                self._config['relative_time_method'].lower()[41:].split('_')

            assert len(splits) == 2, 'Check format of {!r}'.format(
                self._config['relative_time_method'].lower())

            minus = float(splits[0])
            plus = float(splits[1])

            assert minus <= plus, '{!r} !<= {!r}'.format(minus, plus)

            vertex_time = frame[self._cascade_key].time
            global_time_offset = np.random.uniform(vertex_time - minus,
                                                   vertex_time + plus)

        elif self._config['relative_time_method'].lower() == 'time_range':
            sorted_indices = np.argsort(times)
            global_time_offset = get_time_range(charges[sorted_indices],
                                                times[sorted_indices],
                                                time_window_size=6000)[0]

        elif self._config['relative_time_method'] is None:
            global_time_offset = 0.

        elif self._config['relative_time_method'] == 'first_light_at_dom':
            global_time_offset = frame[self._cascade_key].time

        elif self._config['relative_time_method'] == 'wf_quantile':
            global_time_offset = 0.

        elif self._config['relative_time_method'].lower()[:20] == \
                'local_dom_time_range':
            global_time_offset = 0.

        else:
            raise ValueError('Option is uknown: {!r}'.format(
                                        self._config['relative_time_method']))
        return global_time_offset

    def get_local_time_offset(self, frame, om_key, dom_times, dom_charges):
        """Calculate local time offset for pulses of a specific DOM.

        The local time offset is the offset relative to the global time offset.

        Returns
        -------
        float
            Local time offset

        Parameters
        ----------
        om_key : I3OMKey
            The key of the DOM.
        dom_times : numpy.ndarray
            The times of the pulses measured at the DOM.
        dom_charges : numpy.ndarray
            The charges of the pulses measured at the DOM.
        """
        if self._config['relative_time_method'] == 'first_light_at_dom':
            # Global time offset is at vertex time. The times given here
            # are already correctecd for global time offset and therefore
            # the vertex time is at 0.
            local_time_offset = get_time_of_first_light(
                                    dom_pos=self._dom_pos_dict[om_key],
                                    vertex_pos=frame[self._cascade_key].pos,
                                    vertex_time=0.,
                                    )

        elif self._config['relative_time_method'] == 'wf_quantile':
            if len(dom_charges) > 0:
                local_time_offset = get_wf_quantile(
                                        times=dom_times,
                                        charges=dom_charges)
            else:
                local_time_offset = 0.

        elif self._config['relative_time_method'].lower()[:20] == \
                'local_dom_time_range':
            """
            Create a local DOM time offset by finding the time window with the
            most charge. Parameters to the get_time_range function can be
            passed by providing them in the method name.
            These must be separeted by an underscore '_' and the assumed
            order is:
                time_window_size
                step
                rel_charge_threshold
                rel_diff_threshold
            That means, if you want to provde rel_diff_threshold, then you
            must define all previous values as well.
            Example:
                local_dom_time_range_1000_5_0.02
                will result in the paramters:
                    time_window_size = 1000
                    step = 5
                    rel_charge_threshold = 0.02
                    rel_diff_threshold = -1. [default value]
            """

            if len(dom_charges) > 0:
                # define default values
                parameter_values = [1500.0, 1.0, 0.02, -1.]

                splits = self._config[
                                'relative_time_method'].lower()[20:].split('_')

                index = 0
                for split in splits:
                    if split:
                        parameter_values[index] = float(split)
                        index += 1

                # get local DOM time offset given the pulses at the DOM
                sorted_indices = np.argsort(dom_times)
                local_time_offset = get_time_range(
                                    dom_charges[sorted_indices],
                                    dom_times[sorted_indices],
                                    time_window_size=parameter_values[0],
                                    step=parameter_values[1],
                                    rel_charge_threshold=parameter_values[2],
                                    rel_diff_threshold=parameter_values[3],
                                    )[0]
            else:
                local_time_offset = 0.

        else:
            local_time_offset = 0.

        return local_time_offset
