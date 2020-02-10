from __future__ import print_function, division

import numpy as np
from icecube import dataclasses, icetray
from icecube.phys_services import I3Calculator


def get_mc_tree_input_data_dict(frame, angle_bins, distance_bins,
                                distance_cutoff, energy_cutoff):

    allowed_types = [dataclasses.I3Particle.NuclInt,
                     dataclasses.I3Particle.PairProd,
                     dataclasses.I3Particle.Brems,
                     dataclasses.I3Particle.DeltaE,
                     dataclasses.I3Particle.EMinus,
                     dataclasses.I3Particle.EPlus,
                     dataclasses.I3Particle.Hadrons,
                     ]

    num_dist_bins = len(distance_bins) - 1
    num_angle_bins = len(angle_bins) - 1
    num_bins = 2 + num_dist_bins * num_angle_bins

    # create empty data array for DOMs
    dom_data = np.zeros(shape=(86, 60, num_bins))
    dom_data[..., 0] = float('inf')

    # walk through energy losses and calculate data
    for loss in frame['I3MCTree']:

        # skip energy loss if it is not one of the allowed types
        if loss.type not in allowed_types:
            continue

        # walk through DOMs and calculate data
        min_distance = float('inf')
        min_omkey = None
        for string in range(1, 87):
            for dom in range(1, 61):
                om_key = icetray.OMKey(string, dom)

                # get position of DOM
                dom_pos = frame['I3Geometry'].omgeo[om_key].position

                # calculate distance and opening angle to DOM
                diff = dom_pos - loss.pos
                diff_p = dataclasses.I3Particle()
                diff_p.dir = dataclasses.I3Direction(diff.x, diff.y, diff.z)
                angle = I3Calculator.angle(diff_p, loss)
                distance = diff.magnitude

                # sort loss energy to correct bin index
                index_angle = np.searchsorted(angle_bins_rad, angle) - 1
                index_dist = np.searchsorted(distance_bins, distance) - 1

                # check if it is out of bounds
                out_of_bounds = False
                if index_angle < 0 or index_angle >= num_angle_bins:
                    out_of_bounds = True
                if index_dist < 0 or index_dist >= num_dist_bins:
                    out_of_bounds = True

                if not out_of_bounds:
                    # calculate index
                    index = 1 + index_dist * num_angle_bins + index_angle
                    assert index > 0 and index < num_bins - 1

                    # accumulate energy
                    dom_data[om_key.string - 1, om_key.om - 1, index] += \
                        loss.energy

                # check if distance is closest so far
                if distance < dom_data[om_key.string - 1, om_key.om - 1, 0]:
                    dom_data[om_key.string - 1, om_key.om - 1, 0] = distance

                if distance < min_distance:
                    min_distance = distance
                    min_omkey = om_key

        # add energy loss to closest DOM within distance_cutoff
        if loss.energy > energy_cutoff:
            if min_distance < distance_cutoff:
                dom_data[min_omkey.string - 1, min_omkey.om - 1, -1] += \
                    loss.energy

    # create constant lists
    bin_indices = range(num_bins)
    bin_exclusions = []

    # create data dict
    data_dict = {}
    for string in range(1, 87):
        for dom in range(1, 61):
            om_key = icetray.OMKey(string, dom)
            bin_values = dom_data[om_key.string - 1, om_key.om - 1].tolist()
            data_dict[om_key] = (bin_values, bin_indices, bin_exclusions)

    return data_dict
