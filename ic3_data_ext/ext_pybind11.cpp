#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <string>
#include <math.h>

#include "utils.cpp"


/******************************************************
Time critical functions for the Deep Learning-based
reconstruction (DNN_reco) are written in C++ and
wrapped with pybind11.
******************************************************/

// Actual C++ snippets. See docstrings at the bottom for argument info.
namespace py = pybind11;

template <typename T>
inline py::list get_summary_data( const py::array_t<T> dom_charges,
                                  const py::array_t<T> rel_dom_times,
                                  const bool perform_log = false
                                ) {
    auto charges = dom_charges.template unchecked<1>();
    auto times = rel_dom_times.template unchecked<1>();

    int num_pulses = dom_charges.shape(0);

    T dom_charge_sum = 0.0;
    T dom_charge_sum_500ns = 0.0;
    T dom_charge_sum_100ns = 0.0;
    T charge_weighted_quantile20_time = 0.0;
    T charge_weighted_quantile50_time = 0.0;
    T charge_weighted_mean_time = 0.0;
    T charge_weighted_std_time = 0.0;

    //----------------------
    // calculate values
    //----------------------
    MeanVarianceAccumulator<T> acc;

    // loop through pulses
    for(int i=0; i < num_pulses; i++){

        // overall charge
        dom_charge_sum += charges(i);

        // charge in first 100/500 ns
        if( times(i) - times(0) < 500.){
            dom_charge_sum_500ns += charges(i);
            if( times(i) - times(0) < 100.){
                dom_charge_sum_100ns += charges(i);
            }
        }

        // weighted mean and std
        acc.add_element(times(i), charges(i));

    }

    // weighted mean and std
    charge_weighted_mean_time = acc.mean();
    if( num_pulses > 1){
        charge_weighted_std_time = acc.std();
    }

    T quantile_charge_sum = 0.0;
    bool found_quanitle20 = false;
    bool found_quanitle50 = false;

    // second loop through pulses
    for(int i=0; i < num_pulses && !found_quanitle50; i++){

        // weighted quantiles
        quantile_charge_sum += charges(i);

        T rel_charge_factor = quantile_charge_sum / dom_charge_sum;

        if( !found_quanitle20 ){
            if( rel_charge_factor >= 0.2){
                charge_weighted_quantile20_time = times(i);
                found_quanitle20 = true;
            }
        }

        if( !found_quanitle50 ){
            if( rel_charge_factor >= 0.5){
                charge_weighted_quantile50_time = times(i);
                found_quanitle50 = true;
            }
        }

    }

    //----------------------
    // create and fill summary data list
    //----------------------
    py::list summary_data;

    if(perform_log){
        summary_data.append( log10(1.0 + dom_charge_sum) );
        summary_data.append( log10(1.0 + dom_charge_sum_500ns) );
        summary_data.append( log10(1.0 + dom_charge_sum_100ns) );
    }
    else{
        summary_data.append( dom_charge_sum);
        summary_data.append( dom_charge_sum_500ns);
        summary_data.append( dom_charge_sum_100ns);
    }
    summary_data.append( times(0)); // first pulse time
    summary_data.append( charge_weighted_quantile20_time);
    summary_data.append( charge_weighted_quantile50_time);
    summary_data.append( times(num_pulses -1)); // last pulse time
    summary_data.append( charge_weighted_mean_time);
    summary_data.append( charge_weighted_std_time);

    return summary_data;
}


template <typename T>
inline py::list get_time_range(const py::array_t<T> charges,
                               const py::array_t<T> times,
                               const T time_window_size = 6000.0,
                               const T step = 1.0,
                               const T rel_charge_threshold = 0.02,
                               const T rel_diff_threshold = -8.){

    // unchecked: do not check array bounds
    // <1> charges must have ndim = 1
    auto c_charges = charges.template unchecked<1>();
    auto c_times = times.template unchecked<1>();

    // get number of pulses
    int num_pulses = charges.shape(0);

    T start_t = 9000.0;
    if (num_pulses > 0) {

        start_t = 0.0;

        // average noise rate for detector in hits / ns
        T noise_rate = 0.003;
        T total_charge = 0;
        for (unsigned int j = 0; j < num_pulses; j++){
            total_charge += c_charges[j];
        }

        // threshold for which a shift of the time window is still allowed
        T charge_threshold = -total_charge * rel_charge_threshold;

        T max_charge_sum = 0.0;

        // floor to nearest 1000th
        // (Not really necessesary, but done
        //  for consistency between python
        //  implementation)
        //std::hash<T>{}(variable)
        T min_time = T( int(c_times(0) / 1000) *1000);
        T max_time = T(int((c_times(num_pulses - 1)
                            + time_window_size) / 1000.) *1000.);

        const int window_bin_size = int(time_window_size / step);
        const int num_bins = int( (max_time - min_time) / step) + 1;

        // create vector for cumulative sum in each time bin
        std::vector<T> cum_sum(num_bins, 0.0);

        T charge_sum = 0.0;
        int bin_index = 0;
        int pulse_counter = 0;

        for(int time = min_time; time < max_time - step; time += step){ // for consistency to python implementation:
        // for(int time = min_time; time <= max_time; time += step){ // inconsistent with python implementation

            //-----------------------
            // Get Charge in Time bin
            //-----------------------
            T bin_charge_sum = 0.0;

            // loop through pulses in time bin
            while( pulse_counter < num_pulses &&
                    c_times(pulse_counter) < time + step){

                // update charge sum for current time bin
                bin_charge_sum = bin_charge_sum + c_charges(pulse_counter);

                // next pulse
                pulse_counter += 1;
            }

            //-----------------------
            // Update Values
            //-----------------------

            // save charge sum for time bin
            cum_sum[bin_index] = bin_charge_sum;

            // update charge sum
            charge_sum += bin_charge_sum;
            if( bin_index - window_bin_size >= 0 ){
                // subtract charge of time bin not
                // included anymore in current time window
                charge_sum -= cum_sum[bin_index - window_bin_size];
            }

            // calculate the relative difference and compare it to
            // fluctuations expected from the noise rate
            T current_start_t = time - time_window_size + step;
            T uncorrelated_time_window =
                std::min(current_start_t - start_t, time_window_size);

            // Ensure that noise is at least noise for 1ns
            T noise =
                std::max(noise_rate, uncorrelated_time_window * noise_rate);
            T sqrt_noise = sqrt(noise);
            T diff = charge_sum - max_charge_sum;
            T rel_diff = diff / sqrt_noise;

            // update new time window
            if(rel_diff >= rel_diff_threshold && diff >= charge_threshold){
                max_charge_sum = charge_sum;
                start_t = current_start_t;
                if( start_t < min_time ){
                    start_t = min_time;
                }
            }

            // next time window
            bin_index += 1;
        }
    }

    py::list time_range;
    time_range.append(start_t);
    time_range.append(start_t + time_window_size);

    return  time_range;
}


PYBIND11_PLUGIN(ext_pybind11) {
    py::module m("ext_pybind11", R"pbdoc(
        Pybind11 C++ backend for DNN_reco
        -------------------------------

        .. currentmodule:: ext_pybind11

        .. autosummary::
           :toctree: _generate

           get_time_range
    )pbdoc");

    auto get_time_range_docstr = R"pbdoc(
                    Get Time Range of Event:
                    Picks time range of length
                    time_window_size with the most
                    accumulated charge.

                    Parameters
                    ----------
                    charges : array-like, shape (n_pulses)
                        Array of pulse charges
                        Assumes these are ordered in time: tmin -> tmax
                    times : array-like, shape (n_pulses)
                        Array of pulse times.
                        Assumes these are ordered in time: tmin -> tmax
                    time_window_size : double
                        Size of time window in which to
                        look for the most accumulated charge
                    step : double
                        Step size for shifting the time window.
                    rel_charge_threshold : double
                        This defines the maximum relative charge deficit that
                        the new time window may have, e.g. the new time window
                        must have:
                            charge_new <= charge_old * rel_charge_threshold
                    rel_diff_threshold : double
                        This defines the maximum relative difference of charge
                        (delta charge / noise expectation) that the new time
                        window may have.

                    Returns
                    -------
                    time_range : array-like, shape (2)
                        Beginning and end time of the
                        time range: (beginning, end)
                  )pbdoc";


    auto get_summary_data_docstr = R"pbdoc(
                    Calculates summary data
                    for a given list of dom_pulses

                    Parameters
                    ----------
                    dom_charges : array-like, shape (n_pulses)
                        Array of pulse charges
                        Assumes these are ordered in time: tmin -> tmax
                    rel_dom_times : array-like, shape (n_pulses)
                        Array of pulse times relative to first pulse.
                        Assumes these are ordered in time: tmin -> tmax
                    perform_log : bool
                        Whether to perform log or not.
                        True: perform log
                        False: do not perform log

                    Returns
                    -------
                    summary_data : array-like, shape (1)
                        Summary data
                  )pbdoc";


    // Define the actual template types
    m.def("get_time_range", &get_time_range<double>,
          get_time_range_docstr,
          py::arg("charges"), py::arg("times"),
          py::arg("time_window_size") = 6000.0,
          py::arg("step") = 1.0,
          py::arg("rel_charge_threshold") = 0.02,
          py::arg("rel_diff_threshold") = -8.);


    m.def("get_summary_data", &get_summary_data<double>,
          get_summary_data_docstr,
          py::arg("dom_charges"),
          py::arg("rel_dom_times"),
          py::arg("perform_log") = false);


#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif
    return m.ptr();
}
