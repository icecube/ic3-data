/* Reduced Summary Statistics Data Functions

This file is for older boost versions < 106500 (1.65)
which do not have boost/python/numpy.hpp
*/
#ifndef REDUCED_SUMMARY_STATISTICS__OLD_BOOST_VERSION_CPP
#define REDUCED_SUMMARY_STATISTICS__OLD_BOOST_VERSION_CPP

#include "icetray/I3Frame.h"
#include "icetray/OMKey.h"
#include "icetray/I3Logging.h"
#include "dataclasses/physics/I3RecoPulse.h"
#include "dataclasses/I3Map.h"
#include "dataclasses/I3Double.h"
#include "dataclasses/I3Vector.h"

// include necessary boost headers
#include <boost/python.hpp>

#include "utils.cpp"

typedef typename boost::python::numeric::array pyndarray;

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


// -------------------------------------------
// Helper Functions to update Container Fields
// -------------------------------------------
template <typename T>
inline void update_time_offset(
                              boost::python::object container,
                              const T global_offset_time,
                              const int batch_index
                            ) {

    // create references to the data fields that need to be modified
    I3Double& global_time_offset = boost::python::extract<I3Double&>(
        container.attr("global_time_offset"));
    pyndarray global_time_offset_batch = boost::python::extract<pyndarray>(
        container.attr("global_time_offset_batch"));

    // update fields
    global_time_offset = global_offset_time;
    global_time_offset_batch[batch_index] = global_time_offset.value;
}

inline void update_i3_map_data_fields(
                              boost::python::object container,
                              const std::vector<OMKey>& om_keys,
                              const std::vector<I3VectorInt>& bin_indices,
                              const std::vector<I3VectorInt>& bin_exclusions,
                              const std::vector<I3VectorDouble>& bin_values
                            ) {

    // create references to the data fields that need to be modified
    I3MapKeyVectorInt& bin_indices_map = boost::python::extract<I3MapKeyVectorInt&>(
        container.attr("bin_indices"));
    I3MapKeyVectorInt& bin_exclusions_map = boost::python::extract<I3MapKeyVectorInt&>(
        container.attr("bin_exclusions"));
    I3MapKeyVectorDouble& bin_values_map = boost::python::extract<I3MapKeyVectorDouble&>(
        container.attr("bin_values"));

    for (int i = 0; i < om_keys.size(); i++){

        // get om_key
        const OMKey om_key = om_keys[i];

        // assign values to map
        bin_indices_map[om_key] = bin_indices[i];
        bin_values_map[om_key] = bin_values[i];

        if (bin_exclusions[i].size() > 0){
            bin_exclusions_map[om_key] = bin_exclusions[i];
        }
    }
}

inline void update_str_dom_data_fields(
                              boost::python::object container,
                              const int batch_index,
                              const std::vector<OMKey>& om_keys,
                              const std::vector<I3VectorInt>& bin_indices,
                              const std::vector<I3VectorDouble>& bin_values
                            ) {

    // create references to the data fields that need to be modified
    pyndarray x_dom = boost::python::extract<pyndarray>(
        container.attr("x_dom"));

    // check data type of numpy arrays
    // if (bn::dtype::get_builtin<double>() != x_dom.get_dtype()){
    //     log_fatal("Numpy array x_dom in container is not np.float64!");
    // }

    // get a pointer to the input data
    double* x_dom_ptr = reinterpret_cast<double*>(x_dom.getflat());

    // compute helper variables for offset calculation
    const int n_strings = 86;
    const int n_doms = 60;
    const int n_bins =  boost::python::extract<int>(
        container.attr("config")["num_bins"]);
    const int batch_offset = n_strings*n_doms*n_bins*batch_index;

    for (int counter = 0; counter < om_keys.size(); counter++){

        // get data
        const I3VectorInt bin_indices_list = bin_indices[counter];
        const I3VectorDouble bin_values_list = bin_values[counter];

        // get om_key
        const OMKey om_key = om_keys[counter];
        const int om_num = om_key.GetOM() - 1;
        const int string_num = om_key.GetString() - 1;

        const int dom_offset = batch_offset + n_doms*n_bins*string_num + n_bins*om_num;

        // add data values
        for (int i=0; i < bin_indices_list.size(); i++){
            // x_dom[batch_index][string_num][om_num][bin_indices_list[i]]
            //     = bin_values_list[i];

            int offset = dom_offset + bin_indices_list[i];
            x_dom_ptr[offset] = bin_values_list[i];
        }
    }
}

inline void update_hex_data_fields(
                              boost::python::object container,
                              const int batch_index,
                              const std::vector<OMKey>& om_keys,
                              const std::vector<I3VectorInt>& bin_indices,
                              const std::vector<I3VectorDouble>& bin_values
                            ) {

    // create references to the data fields that need to be modified
    pyndarray x_deepcore = boost::python::extract<pyndarray>(
        container.attr("x_deepcore"));
    pyndarray x_ic78 = boost::python::extract<pyndarray>(
        container.attr("x_ic78"));

    // check data type of numpy arrays
    // if (bn::dtype::get_builtin<double>() != x_ic78.get_dtype()){
    //     log_fatal("Numpy array x_ic78 in container is not np.float64!");
    // }
    // if (bn::dtype::get_builtin<double>() != x_deepcore.get_dtype()){
    //     log_fatal("Numpy array x_deepcore in container is not np.float64!");
    // }

    // get a pointer to the input data
    double* x_deepcore_ptr = reinterpret_cast<double*>(x_deepcore.getflat());
    double* x_ic78_ptr = reinterpret_cast<double*>(x_ic78.getflat());

    // compute helper variables for offset calculation
    const int n_bins =  boost::python::extract<int>(
        container.attr("config")["num_bins"]);
    const int dc_batch_offset = 8*60*n_bins*batch_index;
    const int ic_batch_offset = 10*10*60*n_bins*batch_index;

    for (int counter = 0; counter < om_keys.size(); counter++){

        // get data
        const I3VectorInt bin_indices_list = bin_indices[counter];
        const I3VectorDouble bin_values_list = bin_values[counter];

        // get om_key
        const OMKey om_key = om_keys[counter];
        const int om_num = om_key.GetOM() - 1;
        const int string_num = om_key.GetString() - 1;

        // add data values
        if (string_num >= 78){

            const int dom_offset = dc_batch_offset +
                                    60*n_bins*(string_num - 78)
                                    + n_bins*om_num;

            // DeepCore
            for (int i=0; i < bin_indices_list.size(); i++){
                // x_deepcore[batch_index][string_num - 78][om_num]
                //     [bin_indices_list[i]] = bin_values_list[i];

                int offset = dom_offset + bin_indices_list[i];
                x_deepcore_ptr[offset] = bin_values_list[i];
            }

        }else{

            // Main Array (Hex-Structure)

            // Center of Detector is hex_a, hex_b = 0, 0
            // hex_a goes from -4 to 5
            // hex_b goes from -5 to 4
            const int hex_a = STRING_TO_HEX_A[string_num] + 4;
            const int hex_b = STRING_TO_HEX_B[string_num] + 5;

            const int dom_offset =
                ic_batch_offset + 10*60*n_bins*hex_a
                + 60*n_bins*hex_b + n_bins*om_num;

            for (int i=0; i < bin_indices_list.size(); i++){
                // x_ic78[batch_index][hex_a][hex_b][om_num]
                //     [bin_indices_list[i]] = bin_values_list[i];

                int offset = dom_offset + bin_indices_list[i];
                x_ic78_ptr[offset] = bin_values_list[i];
            }
        }
    }
}

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
                                  const bool write_to_frame,
                                  const int batch_index
                                ) {

    // Get pulse map
    I3RecoPulseSeriesMap& pulse_map = boost::python::extract<I3RecoPulseSeriesMap&>(pulse_map_obj);

    // collect settings of container
    const bool is_str_dom_format =  boost::python::extract<bool>(
        container.attr("config")["is_str_dom_format"]);

    // create variables for the output data
    const int num_hit_doms = pulse_map.size();

    T global_offset_time = 0;

    int vector_counter = 0;
    std::vector<OMKey> om_keys(num_hit_doms);
    std::vector<I3VectorInt> bin_indices(num_hit_doms);
    std::vector<I3VectorInt> bin_exclusions(num_hit_doms);
    std::vector<I3VectorDouble> bin_values(num_hit_doms);

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
        I3VectorInt bin_exclusions_list; // empty dummy exclusions
        I3VectorInt bin_indices_list;
        I3VectorDouble bin_values_list;

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
                dom_pulses.second[0].GetTime() - global_offset_time);
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

        // collect lists for DOM
        om_keys[vector_counter] = om_key;
        bin_indices[vector_counter] = bin_indices_list;
        bin_exclusions[vector_counter] = bin_exclusions_list;
        bin_values[vector_counter] = bin_values_list;
        vector_counter += 1;

        // -------------------------------------
    }

    // --------------------------------
    // Now update Container data fields
    // --------------------------------

    // update global offset time
    update_time_offset<T>(container, global_offset_time, batch_index);

    // udate I3Map data fields if results are to be written to the frame
    if (write_to_frame){
        update_i3_map_data_fields(
            container, om_keys, bin_indices, bin_exclusions, bin_values);
    }

    // Update container data
    if (is_str_dom_format){
        update_str_dom_data_fields(
            container, batch_index, om_keys, bin_indices, bin_values);
    }else{
        update_hex_data_fields(
            container, batch_index, om_keys, bin_indices, bin_values);
    }

    // Normally we would have to add exclusions for:
    // x_dom_exclusions
    // x_ic78_exclusions
    // x_deepcore_exclusions
    // But for this data method, there are no exclusions, so we can skip this

}


#endif