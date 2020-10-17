/* Reduced Summary Statistics Data Functions
*/
#ifndef REDUCED_SUMMARY_STATISTICS_CPP
#define REDUCED_SUMMARY_STATISTICS_CPP

#include "icetray/I3Frame.h"
#include "icetray/OMKey.h"
#include "dataclasses/physics/I3RecoPulse.h"
#include "dataclasses/I3Map.h"
#include "dataclasses/I3Double.h"
#include "dataclasses/I3Vector.h"

// include necessary boost headers
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "utils.cpp"

namespace bn = boost::python::numpy;


// Reduced Summary Statistics Data
// -------------------------------

template <typename T>
inline boost::python::tuple get_reduced_summary_statistics_data(
                                  const boost::python::object pulse_map_obj,
                                  const bool add_total_charge,
                                  const bool add_t_first,
                                  const bool add_t_std
                                ) {

    // Get pulse map
    I3RecoPulseSeriesMap& pulse_map = boost::python::extract<I3RecoPulseSeriesMap&>(pulse_map_obj);

    // create a dict for the output data
    boost::python::dict data_dict;
    T global_offset_time = 0.;

    // Iterate over pulses once to obtain global time offset
    if (add_t_first){
        MeanVarianceAccumulator<T> acc_total;
        for (auto const& dom_pulses : pulse_map){
            for (auto const& pulse : dom_pulses.second){
                acc_total.add_element(pulse.GetTime(), pulse.GetCharge());
            }
        }
        global_offset_time = acc_total.mean();
    }

    // now iterate over DOMs and pulses to fill data_dict
    for (auto const& dom_pulses : pulse_map){

        // check if pulses are present
        unsigned int n_pulses = dom_pulses.second.size();
        if (n_pulses == 0){
            continue;
        }

        // create and initialize variables
        T dom_charge_sum = 0.0;
        MeanVarianceAccumulator<T> acc;

        // loop through pulses
        for (auto const& pulse : dom_pulses.second){

            // total DOM charge
            dom_charge_sum += pulse.GetCharge();

            // weighted mean and std
            if (add_t_std){
                acc.add_element(pulse.GetTime(), pulse.GetCharge());
            }
        }

        // add data
        int counter = 0;
        boost::python::list bin_exclusions_list; // empty dummy exclusions
        boost::python::list bin_indices_list;
        boost::python::list bin_values_list;

        // Total DOM charge
        if (add_total_charge){
            bin_indices_list.append(counter);
            bin_values_list.append(dom_charge_sum);
            counter += 1;
        }

        // time of first pulse
        if (add_t_first){
            bin_indices_list.append(counter);
            bin_values_list.append(
                dom_pulses.second[0].GetTime() - global_offset_time);
            counter += 1;
        }

        // time std deviation of pulses at DOM
        if (add_t_std){
            bin_indices_list.append(counter);
            if (n_pulses == 1){
                bin_values_list.append(0.);
            } else{
                bin_values_list.append(acc.std());
            }
            counter += 1;
        }

        // add to data_dict
        data_dict[dom_pulses.first] = boost::python::make_tuple(
                bin_values_list, bin_indices_list, bin_exclusions_list);
    }

    return  boost::python::make_tuple(global_offset_time, data_dict);
}


template <typename T>
inline void fill_reduced_summary_statistics_data(
                                  boost::python::object container,
                                  const boost::python::object pulse_map_obj,
                                  const bool add_total_charge,
                                  const bool add_t_first,
                                  const bool add_t_std,
                                  const int batch_index
                                ) {

    // Get pulse map
    I3RecoPulseSeriesMap& pulse_map = boost::python::extract<I3RecoPulseSeriesMap&>(pulse_map_obj);

    // collect settings of container
    const bool is_str_dom_format =  boost::python::extract<bool>(
        container.attr("config")["is_str_dom_format"]);

    // create references to the data fields that need to be modified

    I3MapKeyVectorInt& bin_indices = boost::python::extract<I3MapKeyVectorInt&>(
        container.attr("bin_indices"));
    I3MapKeyVectorInt& bin_exclusions = boost::python::extract<I3MapKeyVectorInt&>(
        container.attr("bin_exclusions"));
    I3MapKeyVectorDouble& bin_values = boost::python::extract<I3MapKeyVectorDouble&>(
        container.attr("bin_values"));
    I3Double& global_time_offset = boost::python::extract<I3Double&>(
        container.attr("global_time_offset"));

    if (is_str_dom_format){
        // auto np_x_dom =
        // .get_dtype()
        // x_dom =
        // x_dom_exclusions =
    }

    // create a dict for the output data
    boost::python::dict data_dict;
    T global_offset_time = 0.;

    // Iterate over pulses once to obtain global time offset
    if (add_t_first){
        MeanVarianceAccumulator<T> acc_total;
        for (auto const& dom_pulses : pulse_map){
            for (auto const& pulse : dom_pulses.second){
                acc_total.add_element(pulse.GetTime(), pulse.GetCharge());
            }
        }
        global_offset_time = acc_total.mean();
    }

    // now iterate over DOMs and pulses to fill data_dict
    for (auto const& dom_pulses : pulse_map){

        // check if pulses are present
        unsigned int n_pulses = dom_pulses.second.size();
        if (n_pulses == 0){
            continue;
        }

        // create and initialize variables
        T dom_charge_sum = 0.0;
        MeanVarianceAccumulator<T> acc;

        // loop through pulses
        for (auto const& pulse : dom_pulses.second){

            // total DOM charge
            dom_charge_sum += pulse.GetCharge();

            // weighted mean and std
            if (add_t_std){
                acc.add_element(pulse.GetTime(), pulse.GetCharge());
            }
        }

        // add data
        int counter = 0;
        boost::python::list bin_exclusions_list; // empty dummy exclusions
        boost::python::list bin_indices_list;
        boost::python::list bin_values_list;

        // Total DOM charge
        if (add_total_charge){
            bin_indices_list.append(counter);
            bin_values_list.append(dom_charge_sum);
            counter += 1;
        }

        // time of first pulse
        if (add_t_first){
            bin_indices_list.append(counter);
            bin_values_list.append(
                dom_pulses.second[0].GetTime() - global_offset_time);
            counter += 1;
        }

        // time std deviation of pulses at DOM
        if (add_t_std){
            bin_indices_list.append(counter);
            if (n_pulses == 1){
                bin_values_list.append(0.);
            } else{
                bin_values_list.append(acc.std());
            }
            counter += 1;
        }

        // add to data_dict
        data_dict[dom_pulses.first] = boost::python::make_tuple(
                bin_values_list, bin_indices_list, bin_exclusions_list);
    }
}

#endif