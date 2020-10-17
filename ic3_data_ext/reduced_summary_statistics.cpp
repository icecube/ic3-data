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

// --------------------------------------------
// Define Detector Constants for Hex-Conversion
// --------------------------------------------
const int STRING_TO_HEX_A[78] = {
    -4, -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2,
    -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4,
    4,  4,  4,  4,  4,  4,  5,  5,  5,  5
};
const int STRING_TO_HEX_B[78] = {
    -1,  0,  1,  2,  3,  4, -2, -1,  0,  1,  2,  3,  4, -3, -2, -1,  0,
    1,  2,  3,  4, -4, -3, -2, -1,  0,  1,  2,  3,  4, -5, -4, -3, -2,
    -1,  0,  1,  2,  3,  4, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4, -5,
    -4, -3, -2, -1,  0,  1,  2,  3, -5, -4, -3, -2, -1,  0,  1,  2, -5,
    -4, -3, -2, -1,  0,  1, -5, -4, -3, -2
};


// -------------------------------
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

    // -------------------------------------------------------------
    // create references to the data fields that need to be modified
    // -------------------------------------------------------------
    I3MapKeyVectorInt& bin_indices = boost::python::extract<I3MapKeyVectorInt&>(
        container.attr("bin_indices"));
    I3MapKeyVectorInt& bin_exclusions = boost::python::extract<I3MapKeyVectorInt&>(
        container.attr("bin_exclusions"));
    I3MapKeyVectorDouble& bin_values = boost::python::extract<I3MapKeyVectorDouble&>(
        container.attr("bin_values"));
    I3Double& global_time_offset = boost::python::extract<I3Double&>(
        container.attr("global_time_offset"));

    bn::ndarray global_time_offset_batch = boost::python::extract<bn::ndarray>(
        container.attr("global_time_offset_batch"));

    // x_dom, x_dom_exclusions, x_ic78, x_ic78_exclusions, x_deepcore
    // and x_deepcore_exclusions depend on is_str_dom_format and do not
    // always exist. Therefore we will get them further below when needed
    // -------------------------------------------------------------

    // create a dict for the output data
    boost::python::dict data_dict;

    // Iterate over pulses once to obtain global time offset
    if (add_t_first){
        MeanVarianceAccumulator<T> acc_total;
        for (auto const& dom_pulses : pulse_map){
            for (auto const& pulse : dom_pulses.second){
                acc_total.add_element(pulse.GetTime(), pulse.GetCharge());
            }
        }

        // set global time offset values
        global_time_offset = acc_total.mean();
        global_time_offset_batch[batch_index] = global_time_offset.value;
    }

    // now iterate over DOMs and pulses to fill data_dict
    for (auto const& dom_pulses : pulse_map){

        // check if pulses are present
        unsigned int n_pulses = dom_pulses.second.size();
        if (n_pulses == 0){
            continue;
        }

        // get om_key for conenience
        const OMKey om_key = dom_pulses.first;
        const int om_num = om_key.GetOM() - 1;
        const int string_num = om_key.GetString() - 1;

        // only take real in-ice DOMs
        if (om_num >= 60){
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

        // -------------------------------------
        // Collect values and update data fields
        // -------------------------------------
        int counter = 0;
        std::vector<int> bin_exclusions_list; // empty dummy exclusions
        std::vector<int> bin_indices_list;
        std::vector<double> bin_values_list;

        // Total DOM charge
        if (add_total_charge){

            // update
            bin_indices_list.push_back(counter);
            bin_values_list.push_back(dom_charge_sum);
            counter += 1;
        }

        // time of first pulse
        if (add_t_first){
            bin_indices_list.push_back(counter);
            bin_values_list.push_back(
                dom_pulses.second[0].GetTime() - global_time_offset.value);
            counter += 1;
        }

        // time std deviation of pulses at DOM
        if (add_t_std){
            bin_indices_list.push_back(counter);
            if (n_pulses == 1){
                bin_values_list.push_back(0.);
            } else{
                bin_values_list.push_back(acc.std());
            }
            counter += 1;
        }

        // Update fields
        bin_indices[om_key] = bin_indices_list;
        bin_exclusions[om_key] = bin_exclusions_list;
        bin_values[om_key] = bin_values_list;

        // add data values
        for (int i=0; i < bin_indices_list.size(); i++){
            if (is_str_dom_format){

                // Get reference to data field
                bn::ndarray x_dom = boost::python::extract<bn::ndarray>(
                    container.attr("x_dom"));
                std::cout << "batch_index: " << batch_index << std::endl;
                std::cout << "string_num: " << string_num << std::endl;
                std::cout << "om_num: " << om_num << std::endl;
                std::cout << "i: " << i << std::endl;

                x_dom[batch_index, string_num, om_num,
                    bin_indices_list[i]] = bin_values_list[i];

            }else{

                // DeepCore
                if (string_num >= 78){

                    // Get reference to data field
                    bn::ndarray x_deepcore = boost::python::extract<bn::ndarray>(
                        container.attr("x_deepcore"));

                    x_deepcore[batch_index, string_num - 78, om_num,
                        bin_indices_list[i]] = bin_values_list[i];

                // Main Array (Hex-Structure)
                }else{

                    // Get reference to data field
                    bn::ndarray x_ic78 = boost::python::extract<bn::ndarray>(
                        container.attr("x_ic78"));

                    const int hex_a = STRING_TO_HEX_A[string_num];
                    const int hex_b = STRING_TO_HEX_B[string_num];

                    // Center of Detector is hex_a, hex_b = 0, 0
                    // hex_a goes from -4 to 5
                    // hex_b goes from -5 to 4
                    x_ic78[batch_index, hex_a + 4, hex_b + 5, om_num,
                        bin_indices_list[i]] = bin_values_list[i];
                }

            }
        }

        // Normally we would have to add exclusions for:
        // x_dom_exclusions
        // x_ic78_exclusions
        // x_deepcore_exclusions
        // But for this data metho, there are no exclusions,
        // so we can skip this

        // -------------------------------------
    }
}

#endif