from __future__ import print_function, division

import numpy as np
from icecube import dataclasses


def get_wf_quantile(times, charges, quantile=0.1):
    """Get charge weighted time quantile of pulses.

    Parameters
    ----------
    times : list or np.ndarray
        Time of pulses.
        Assumed that the times are already SORTED!
        This is usually the case for I3 PulseSeries.
    charges : list or np.ndarray
        Charges of pulses.
    quantile : float, optional
        Cumulative charge quantile.

    Returns
    -------
    float
        The time will be returned at which a fraction of the total charge
        equal to 'quantile' is collected.
    """

    assert 0 <= quantile and quantile <= 1., \
        '{!r} not in [0,1]'.format(quantile)

    # calculate total charge
    total_charge = np.sum(charges)

    # compute cumulative sum
    cum_charge = np.cumsum(charges) / total_charge

    # small constant:
    epsilon = 1e-6

    # get time of pulse at which the cumulative charge is first >= q:
    mask = cum_charge >= quantile - epsilon
    return times[mask][0]


def get_time_of_first_light(dom_pos, vertex_pos, vertex_time):
    """Get the time when unscattered light, emitted at the vertex position
    and time, arrives at the dom position.

    Parameters
    ----------
    dom_pos : I3Position
        I3Position of DOM
    vertex_pos : I3Position
        I3Position of cascade vertex.
    vertex_time : float
        Time of cascade vertex.

    Returns
    -------
    float
        Time when first unscattered light from cascade arrives at DOM
    """
    distance = (dom_pos - vertex_pos).magnitude
    photon_time = distance / dataclasses.I3Constants.c_ice
    return vertex_time + photon_time


def get_time_range(charges, times, time_window_size=6000):
    """Get time range in which the most charge is detected.

    A time window of size 'time_window_size' is shifted across the
    event in 500 ns time steps. The time range, at which the maximum
    charge is enclosed within the time window, is returned

    Parameters
    ----------
    charges : list of float
        List of measured pulse charges.
    times : list of float
        List of measured pulse times
    time_window_size : int, optional
        Defines the size of the time window.

    Returns
    -------
    (float, float)
        Time rane in which the maximum charge is detected.
    """
    max_charge_sum = 0
    start_t = 9000
    for t in range(int(np.nanmin(times)//1000)*1000,
                   int(np.nanmax(times)//1000)*1000 - time_window_size,
                   500):
        indices_smaller = t < times
        indices_bigger = times < t + time_window_size
        indices = np.logical_and(indices_smaller, indices_bigger)
        charge_sum = np.sum(charges[indices])
        if charge_sum > max_charge_sum:
            max_charge_sum = charge_sum
            start_t = t
    return [start_t, start_t + time_window_size]
