from __future__ import print_function, division
import numpy as np

from ic3_data.ext_boost import get_cascade_classification_data

"""All data format functions must have the following signature:

    Parameters
    ----------
    frame : I3Frame
        The current frame.
    pulses : I3RecoPulseSeriesMap
        The pulse series map from which to calculate the DNN input data.
    config : dict
        A dictionary that contains all configuration settings.
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
        A dictionary with the structure I3OMKey: (list of float, list of int)
        The first list of the tuple are the bin values and the second list
        are the bin indices
        list
            Bin values: values of non-zero data bins.
        list
            Bin indices: indices to which the returned values belong to.
    """


def cascade_classification_data(frame, pulses, config, *args, **kwargs):
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
        A dictionary with the structure I3OMKey: (list of float, list of int)
        The first list of the tuple are the bin values and the second list
        are the bin indices
        list
            Bin values: values of non-zero data bins.
        list
            Bin indices: indices to which the returned values belong to.
    """

    add_dom_info = True
    time_quantiles = config['time_quantiles']

    # fill in vertex time
    global_time_offset = frame[config['cascade_key']]['VertexTime']

    data_dict = get_cascade_classification_data(
            frame, pulses, config['cascade_key'], time_quantiles, add_dom_info)

    return global_time_offset, data_dict
