from __future__ import print_function, division
import numpy as np
from scipy.special import erf

from icecube import icetray, dataclasses

from ic3_data.utils.detector import get_dom_coords


class CascasdePulseVisualizer(icetray.I3ConditionalModule):
    '''Module to visualize pulses that are relevant for cascade classification,
    i.e. finding ingoing and outgoing muons.
    '''
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("PulseKey",
                          "Pulses to use",
                          'InIceDSTPulses')
        self.AddParameter("OutputKey",
                          "Base name to which the frames will be stored",
                          'CascasdePulseVisualizer')
        self.AddParameter("CascadeKey",
                          "Key from which the vertex and uncertainties will "
                          "be obtained. If this is an I3Particle, of if "
                          "uncertainties as 'VerteX_unc' are not provided, "
                          "They will default to 30m and 100 ns",
                          'L3_MonopodFit4')
        self.AddParameter("TimeQuantiles",
                          "Time quantiles as a list of tuples of (rel, abs).",
                          # [(-1, -2000), (-1, -1000), (-1, 0),
                          #  (-0.5, 0), (0, 0), (1, 0), (1.35634, 0),
                          #  (1.35634, 100), (1.35634, 500), (1.35634, 5000)]
                          # [(-1, -1000), (-1, 0),
                          #  (-0.5, 0), (0, 0), (0.5, 0), (1, 0), (1.35634, 0),
                          #  (1.5, 0), (2.0, 0), (3.0, 0), (3.0, + 2000)]
                          [(-1, -1000), (-1, 0),
                           (-0.5, 0), (0, 0), (0.25, 0), (0.50, 0), (0.75, 0),
                           (1, 0), (1.35634, 0), (1.7, 0), (2.0, 0), (3.0, 0),
                           (3.0, + 2000)]
                          )
        self.AddParameter("ProbabilityThreshold",
                          "Threshold under which to discard pulses",
                          0.2)

    def Configure(self):
        """Configure DNN data handler

        Raises
        ------
        ValueError
            Description
        """
        self._pulse_key = self.GetParameter("PulseKey")
        self._output_key = self.GetParameter("OutputKey")
        self._cascade_key = self.GetParameter("CascadeKey")
        self._time_quantiles = self.GetParameter("TimeQuantiles")
        self._threshold = self.GetParameter("ProbabilityThreshold")

        self._dom_pos_dict = None

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
        if self._dom_pos_dict is None:
            print('No Geometry provided. Limiting to main detector array!')

        pulses = frame[self._pulse_key]

        if isinstance(pulses, dataclasses.I3RecoPulseSeriesMapMask) or \
                isinstance(pulses, dataclasses.I3RecoPulseSeriesMapUnion):
            pulses = pulses.apply(frame)

        # get cascade position and uncertainty
        cascade_obj = frame[self._cascade_key]

        if isinstance(cascade_obj, dataclasses.I3Particle):
            x = cascade_obj.pos.x
            y = cascade_obj.pos.y
            z = cascade_obj.pos.z
            t = cascade_obj.time
            unc_x = 10.
            unc_y = 10.
            unc_z = 10.
            unc_t = 50.
        else:
            x = cascade_obj['VertexX']
            y = cascade_obj['VertexY']
            z = cascade_obj['VertexZ']
            t = cascade_obj['VertexTime']

            if 'VertexX_unc' in cascade_obj:
                unc_x = cascade_obj['VertexX_unc']
            else:
                unc_x = 10.

            if 'VertexY_unc' in cascade_obj:
                unc_y = cascade_obj['VertexY_unc']
            else:
                unc_y = 10.

            if 'VertexZ_unc' in cascade_obj:
                unc_z = cascade_obj['VertexZ_unc']
            else:
                unc_z = 10.

            if 'VertexTime_unc' in cascade_obj:
                unc_t = cascade_obj['VertexTime_unc']
            else:
                unc_t = 50.

        c = dataclasses.I3Constants.c
        c_ice = dataclasses.I3Constants.c_ice

        unc_pos = np.sqrt(unc_x**2 + unc_y**2 + unc_z**2)
        unc_time = np.sqrt((unc_pos / c)**2 + unc_t**2)

        num_bins = len(self._time_quantiles) - 1
        cascade_pulse_maps = [dataclasses.I3RecoPulseSeriesMap()
                              for i in range(num_bins)]

        # loop through hit DOMs
        for om_key, dom_pulses in pulses:

            if self._dom_pos_dict is None:
                if om_key.string > 78:
                    # skip DeepCore DOMs
                    continue
                pos = dataclasses.I3Position(*get_dom_coords(om_key.string,
                                                             om_key.om))
            else:
                pos = self._dom_pos_dict[om_key]

            # distance to DOM
            distance = np.sqrt((x - pos.x)**2 + (y - pos.y)**2 +
                               (z - pos.z)**2)
            delta_t = distance / c

            time_bins = []
            for rel_t, abs_t in self._time_quantiles:
                time_bins.append(t + rel_t*delta_t + abs_t)

            max_t = time_bins[-1] + 3*unc_time

            # Now loop through pulses and keep pulses above the threshold
            cascade_pulses = [dataclasses.I3RecoPulseSeries()
                              for i in range(num_bins)]
            for p in dom_pulses:
                if p.time > max_t:
                    break

                # get cumulative pdf at first time edge
                sigma = (time_bins[0] - p.time) / unc_time
                current_cdf = 0.5*(1 + erf(sigma/np.sqrt(2)))

                # loop through the rest of the time windows
                for i, edge in enumerate(time_bins[1:]):

                    # calculate cdf at this time edge
                    sigma = (edge - p.time) / unc_time
                    new_cdf = 0.5*(1 + erf(sigma/np.sqrt(2)))

                    prob = new_cdf - current_cdf

                    # save pulse if above threshold
                    if prob > self._threshold:
                        new_pulse = dataclasses.I3RecoPulse(p)
                        new_pulse.charge = prob * p.charge
                        new_pulse.time = prob * p.time
                        cascade_pulses[i].append(new_pulse)

                    # update current CDF value
                    current_cdf = new_cdf

            for i in range(num_bins):
                if len(cascade_pulses[i]) > 0:
                    cascade_pulse_maps[i][om_key] = cascade_pulses[i]

        # write to frame
        for i in range(num_bins):
            frame[self._output_key + '_bin_{:03d}'.format(i)] = \
                cascade_pulse_maps[i]

        self.PushFrame(frame)
