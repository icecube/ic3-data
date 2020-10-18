from __future__ import print_function, division
import numpy as np

from ic3_data.ext_boost import fill_reduced_summary_statistics_data

"""All data format functions must have the following signature:

    Parameters
    ----------
    container : DNNDataContainer
        The data container that will be filled.
    batch_index : int
        The batch index.
    write_to_frame : bool
        Whether or not the DNN data should be written to the frame.
    frame : I3Frame
        The current frame.
    pulses : I3RecoPulseSeriesMap
        The pulse series map from which to calculate the DNN input data.
    config : dict
        A dictionary that contains all configuration settings.
    dom_exclusions : list of str, None
        List of frame keys that define DOMs or TimeWindows that should be
        excluded. Typical values for this are:
        ['BrightDOMs','SaturationWindows', 'BadDomsList', 'CalibrationErrata']
    partial_exclusion : bool, None
        If True, partially exclude DOMS, e.g. only omit pulses from
        excluded TimeWindows defined in 'dom_exclusions'.
        If False, all pulses from a DOM will be excluded if the omkey
        exists in the dom_exclusions.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """


def total_dom_charge(container, batch_index, write_to_frame, frame, pulses,
                     config, dom_exclusions, partial_exclusion,
                     *args, **kwargs):
    """Get the total DOM charge per DOM

    Parameters
    ----------
    container : DNNDataContainer
        The data container that will be filled.
    batch_index : int
        The batch index.
    write_to_frame : bool
        Whether or not the DNN data should be written to the frame.
    frame : I3Frame
        The current frame.
    pulses : I3RecoPulseSeriesMap
        The pulse series map from which to calculate the DNN input data.
    config : dict
        A dictionary that contains all configuration settings.
    dom_exclusions : list of str, None
        List of frame keys that define DOMs or TimeWindows that should be
        excluded. Typical values for this are:
        ['BrightDOMs','SaturationWindows', 'BadDomsList', 'CalibrationErrata']
    partial_exclusion : bool, None
        If True, partially exclude DOMS, e.g. only omit pulses from
        excluded TimeWindows defined in 'dom_exclusions'.
        If False, all pulses from a DOM will be excluded if the omkey
        exists in the dom_exclusions.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    add_total_charge = True
    add_t_first = False
    add_t_std = False

    fill_reduced_summary_statistics_data(
        container=container,
        pulse_key=pulses,
        add_total_charge=add_total_charge,
        add_t_first=add_t_first,
        add_t_std=add_t_std,
        write_to_frame=write_to_frame,
        batch_index=batch_index,
    )


def reduced_summary_statistics_data(container, batch_index, write_to_frame,
                                    frame, pulses, config, dom_exclusions,
                                    partial_exclusion, *args, **kwargs):
    """Get a reduced set of summary statistics per DOM

    These include: total dom charge, time of first pulse, std. dev of pulse
    times. The pulse times are calculated relative to the charge weighted
    mean time of all pulses.

    Parameters
    ----------
    container : DNNDataContainer
        The data container that will be filled.
    batch_index : int
        The batch index.
    write_to_frame : bool
        Whether or not the DNN data should be written to the frame.
    frame : I3Frame
        The current frame.
    pulses : I3RecoPulseSeriesMap
        The pulse series map from which to calculate the DNN input data.
    config : dict
        A dictionary that contains all configuration settings.
    dom_exclusions : list of str, None
        List of frame keys that define DOMs or TimeWindows that should be
        excluded. Typical values for this are:
        ['BrightDOMs','SaturationWindows', 'BadDomsList', 'CalibrationErrata']
    partial_exclusion : bool, None
        If True, partially exclude DOMS, e.g. only omit pulses from
        excluded TimeWindows defined in 'dom_exclusions'.
        If False, all pulses from a DOM will be excluded if the omkey
        exists in the dom_exclusions.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    add_total_charge = True
    add_t_first = True
    add_t_std = True

    fill_reduced_summary_statistics_data(
        container=container,
        pulse_key=pulses,
        add_total_charge=add_total_charge,
        add_t_first=add_t_first,
        add_t_std=add_t_std,
        write_to_frame=write_to_frame,
        batch_index=batch_index,
    )
