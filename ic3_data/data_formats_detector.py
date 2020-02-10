from __future__ import print_function, division
import numpy as np

from ic3_data.ext_boost import get_cascade_classification_data
from ic3_data.utils.mc_tree_input import get_mc_tree_input_data_dict

"""All data format functions must have the following signature:

    Parameters
    ----------
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

    Returns
    -------
    float
        Global time offset. Time variables will be shifted by this global
        offset. E.g. if the network predicts a vertex time based on input
        data relative to the vertex time, the predicted time will be shifted
        by this global offset.
    dict
        A dictionary with the structure
        I3OMKey: (list of float, list of int, list of int)
        The first list of the tuple are the bin values and the second list
        are the bin indices. The last list is a list of bin exclusions.
        list
            Bin values: values of non-zero data bins.
        list
            Bin indices: indices to which the returned values belong to.
        list
            Bin exclusions: indices which define bins that will be excluded.
    """


def cascade_classification_data(frame, pulses, config, dom_exclusions,
                                partial_exclusion, *args, **kwargs):
    """Get pulses on DOMs in time bins relative to expected arrival times
    based on a given cascade hypothesis. This can be useful to detect incoming
    or outgoing muons.

    The cascade hypothesis is given by the I3MapStringDouble 'cascade_key'.
    It must contain the following values:

        VertexX: The x-position of the cascade vertex in meter.
        VertexY: The y-position of the cascade vertex in meter.
        VertexZ: The z-position of the cascade vertex in meter.
        VertexTime:  The time of the cascade vertex in ns.
        VertexX_unc: The uncertainty on the x-position in meter.
        VertexY_unc: The uncertainty on the y-position in meter.
        VertexZ_unc: The uncertainty on the z-position in meter.
        VertexTime_unc:  The uncertainty on the time in ns.


    Parameters
    ----------
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

    Returns
    -------
    float
        Global time offset. Time variables will be shifted by this global
        offset. E.g. if the network predicts a vertex time based on input
        data relative to the vertex time, the predicted time will be shifted
        by this global offset.
    dict
        A dictionary with the structure
        I3OMKey: (list of float, list of int, list of int)
        The first list of the tuple are the bin values and the second list
        are the bin indices. The last list is a list of bin exclusions.
        list
            Bin values: values of non-zero data bins.
        list
            Bin indices: indices to which the returned values belong to.
        list
            Bin exclusions: indices which define bins that will be excluded.
    """

    add_dom_info = True
    time_quantiles = config['time_quantiles']

    # fill in vertex time
    global_time_offset = frame[config['cascade_key']]['VertexTime']

    data_dict = get_cascade_classification_data(
            frame, pulses, config['cascade_key'], time_quantiles, add_dom_info)

    return global_time_offset, data_dict


def mc_tree_input_data(frame, pulses, config, dom_exclusions,
                       partial_exclusion, *args, **kwargs):
    """Get input data based on I3MCTree.
    This can, for example, be used to create a model for biased simulation.

    For each DOM, the following values are calculated:

        1.) Distance to closest energy loss in meter.
        2.) Charge within 0 to 30 meter and 0 to 45 degree
        3.) Charge within 0 to 30 meter and 45 to 90 degree
        4.) Charge within 0 to 30 meter and 90 to 180 degree
        5.) Charge within 30 to 60 meter and 0 to 45 degree
        6.) Charge within 30 to 60 meter and 45 to 90 degree
        7.) Charge within 30 to 60 meter and 90 to 180 degree
        8.) Charge within 60 to 150 meter and 0 to 45 degree
        9.) Charge within 60 to 150 meter and 45 to 90 degree
        10.) Charge within 60 to 150 meter and 90 to 180 degree
        11.) Accumulated charge from energy losses (within 500m) that are
             closer to this DOM than to any other DOM.

    Parameters
    ----------
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

    Returns
    -------
    float
        Global time offset. Time variables will be shifted by this global
        offset. E.g. if the network predicts a vertex time based on input
        data relative to the vertex time, the predicted time will be shifted
        by this global offset.
    dict
        A dictionary with the structure
        I3OMKey: (list of float, list of int, list of int)
        The first list of the tuple are the bin values and the second list
        are the bin indices. The last list is a list of bin exclusions.
        list
            Bin values: values of non-zero data bins.
        list
            Bin indices: indices to which the returned values belong to.
        list
            Bin exclusions: indices which define bins that will be excluded.
    """
    global_time_offset = 0.
    angle_bins = [0., 45., 90., 180.]  # degree
    distance_bins = [0., 30., 60., 150.]  # meter
    distance_cutoff = 500.  # meter
    energy_cutoff = 1  # GeV

    data_dict = get_mc_tree_input_data_dict(
        frame=frame, angle_bins=angle_bins, distance_bins=distance_bins,
        distance_cutoff=distance_cutoff, energy_cutoff=energy_cutoff)

    return global_time_offset, data_dict
