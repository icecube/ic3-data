#include <iostream>
#include <string>
#include <math.h>

#include "icetray/I3Frame.h"
#include "dataclasses/physics/I3RecoPulse.h"
#include "dataclasses/physics/I3Particle.h"
#include "dataclasses/geometry/I3Geometry.h"
#include "dataclasses/geometry/I3OMGeo.h"
#include "dataclasses/I3Map.h"
#include "dataclasses/I3TimeWindow.h"
#include <boost/python.hpp>
#include "numpy/ndarrayobject.h"


/******************************************************
Time critical functions for the Deep Learning-based
reconstruction (DNN_reco) are written in C++ and
wrapped with boost python.
******************************************************/

// Actual C++ snippets. See docstrings at the bottom for argument info.

/*template <typename T>
inline boost::python::tuple restructure_pulses(
                                      const boost::python::object& pulse_map_obj
                                    ) {

    // Get pulse map
    I3RecoPulseSeriesMap& pulse_map = boost::python::extract<I3RecoPulseSeriesMap&>(pulse_map_obj);

    boost::python::list charges;
    boost::python::list times;
    boost::python::dict dom_times_dict;
    boost::python::dict dom_charges_dict;

    for (auto const& dom_pulses : pulse_map){
        boost::python::list dom_charges;
        boost::python::list dom_times;
        for (int i=0; i < dom_pulses.second.size(); i++ ){
            dom_charges.append(dom_pulses.second.at(i).GetCharge());
            dom_times.append(dom_pulses.second.at(i).GetTime());
        }
        dom_times_dict[dom_pulses.first] = dom_times;
        dom_charges_dict[dom_pulses.first] = dom_charges;
        charges.extend(dom_charges);
        times.extend(dom_times);
    }

    return  boost::python::make_tuple(
                            charges, times, dom_times_dict, dom_charges_dict );
}*/

template <typename T>
struct select_npy_type
{};

template <>
struct select_npy_type<double>
{
    const static NPY_TYPES type = NPY_DOUBLE;
};

template <>
struct select_npy_type<float>
{
    const static NPY_TYPES type = NPY_FLOAT;
};

template <>
struct select_npy_type<int>
{
    const static NPY_TYPES type = NPY_INT;
};

// https://stackoverflow.com/questions/10701514/how-to-return-numpy-array-from-boostpython
template <typename T>
boost::python::object stdVecToNumpyArray( std::vector<T> const& vec )
{
      npy_intp size = vec.size();

     /* const_cast is rather horrible but we need a writable pointer
        in C++11, vec.data() will do the trick
        but you will still need to const_cast
      */

      T * data = size ? const_cast<T *>(&vec[0])
        : static_cast<T *>(NULL);

    // create a PyObject * from pointer and data
      PyObject * pyObj = PyArray_SimpleNewFromData( 1, &size, select_npy_type<T>::type, data );
      boost::python::handle<> handle( pyObj );
      boost::python::numeric::array arr( handle );

    /* The problem of returning arr is twofold: firstly the user can modify
      the data which will betray the const-correctness
      Secondly the lifetime of the data is managed by the C++ API and not the
      lifetime of the numpy array whatsoever. But we have a simple solution..
     */

       return arr.copy(); // copy the object. numpy owns the copy now.
  }
// todo: change this to templated function and remove duplicate

template <std::size_t N, typename T>
boost::python::object stdArrayToNumpyArray( std::array<T, N> const& vec )
{
      npy_intp size = vec.size();

     /* const_cast is rather horrible but we need a writable pointer
        in C++11, vec.data() will do the trick
        but you will still need to const_cast
      */

      T * data = size ? const_cast<T *>(&vec[0])
        : static_cast<T *>(NULL);

    // create a PyObject * from pointer and data
      PyObject * pyObj = PyArray_SimpleNewFromData( 1, &size, select_npy_type<T>::type, data );
      boost::python::handle<> handle( pyObj );
      boost::python::numeric::array arr( handle );

    /* The problem of returning arr is twofold: firstly the user can modify
      the data which will betray the const-correctness
      Secondly the lifetime of the data is managed by the C++ API and not the
      lifetime of the numpy array whatsoever. But we have a simple solution..
     */

       return arr.copy(); // copy the object. numpy owns the copy now.
  }

template <typename T>
inline boost::python::dict get_cascade_classification_data(
                                  boost::python::object& frame_obj,
                                  const boost::python::object& pulse_map_obj,
                                  const boost::python::object& cascade_key_obj,
                                  boost::python::object& time_quantiles_obj,
                                  const bool add_dom_info
                                ) {

    // extract objects
    I3Frame& frame = boost::python::extract<I3Frame&>(frame_obj);
    const std::string cascade_key =
        boost::python::extract<std::string>(cascade_key_obj);

    const unsigned int n_bins =  len(time_quantiles_obj) - 1;

    std::vector<double> time_quantiles_abs;
    std::vector<double> time_quantiles_rel;
    for (int i = 0; i < n_bins + 1; ++i){
        time_quantiles_rel.push_back(
                boost::python::extract<double>(time_quantiles_obj[i][0]));
        time_quantiles_abs.push_back(
                boost::python::extract<double>(time_quantiles_obj[i][1]));
    }

    // Get geometry map
    const I3OMGeoMap& omgeo = frame.Get<I3Geometry>("I3Geometry").omgeo;
    // Get pulse map
    I3RecoPulseSeriesMap& pulse_map = boost::python::extract<I3RecoPulseSeriesMap&>(pulse_map_obj);

    // get parameters
    const I3MapStringDouble& params = frame.Get<I3MapStringDouble>(cascade_key);

    double x = params.at("VertexX");
    double y = params.at("VertexY");
    double z = params.at("VertexZ");
    double t = params.at("VertexTime");
    double unc_x = params.at("VertexX_unc");
    double unc_y = params.at("VertexY_unc");
    double unc_z = params.at("VertexZ_unc");
    double unc_t = params.at("VertexTime_unc");
    double c = 0.299792458; // m/ns from dataclasses.I3Constants.c
    double c_ice = 0.221030462863; // m/ns from dataclasses.I3Constants.c_ice
    double sqrt2 = std::sqrt(2);

    double unc_pos = std::sqrt(unc_x*unc_x + unc_y*unc_y + unc_z*unc_z);
    double unc_time = std::sqrt(std::pow(unc_pos / c, 2) + unc_t*unc_t);

    boost::python::dict data_dict;

    for (auto const& dom_pulses : pulse_map){

        const I3Position& pos = omgeo.at(dom_pulses.first).position;

        // distance to DOM
        double distance = std::sqrt(std::pow(x - pos.GetX(), 2) +
                                    std::pow(y - pos.GetY(), 2) +
                                    std::pow(z - pos.GetZ(), 2));

        double delta_t = distance / c;

        std::vector<double> time_edges;

        double max_edge = 0;
        for (unsigned int i=0; i < n_bins + 1; i++){
            max_edge =
                t + time_quantiles_rel[i] * delta_t + time_quantiles_abs[i];
            time_edges.push_back(max_edge);
        }

        std::vector<double> bin_values_list;
        for (unsigned int i=0; i < n_bins; i++){
            bin_values_list.push_back(0);
        }

        // loop through pulses
        for (auto const& pulse : dom_pulses.second){
            //dom_charges.push_back(pulse.GetCharge());
            //dom_times.push_back(pulse.GetTime());

            // check if we need to keep going
            if (pulse.GetTime() > max_edge + 4*unc_time){
                break;
            }

            // get cumulative pdf at first time edge
            double sigma = (time_edges[0] - pulse.GetTime()) / unc_time;
            double current_cdf = 0.5*(1 + std::erf(sigma/sqrt2));

            // get iterator for vector
            std::vector<double>::iterator values_iterator =
                bin_values_list.begin();

            // loop through time windows
            for (unsigned int i=1; i<n_bins+1; i++){

                // calculate cdf at this time edge
                double sigma = (time_edges[i] - pulse.GetTime()) / unc_time;
                double new_cdf = 0.5*(1 + std::erf(sigma/sqrt2));

                // accumulate charge in bin
                /*bin_values_list[i-1] +=
                    (new_cdf - current_cdf) * pulse.GetCharge();*/
                *values_iterator +=
                    (new_cdf - current_cdf) * pulse.GetCharge();
                values_iterator += 1;

                current_cdf = new_cdf;
            }

        }

        unsigned int n_total =  n_bins;
        if (add_dom_info){
            n_total += 2;
            bin_values_list.push_back(distance);
            bin_values_list.push_back(unc_time);
        }

        // compress output: throw out extremely small values
        std::vector<int> bin_indices_list;
        std::vector<double> bin_values_list_compressed;
        std::vector<double>::iterator values_iterator =
                bin_values_list.begin();
        for (unsigned int i=0; i < n_total; i++){
            if (*values_iterator > 1e-7){
                bin_values_list_compressed.push_back(*values_iterator);
                bin_indices_list.push_back(i);
            }
            values_iterator += 1;
        }

        if (bin_values_list_compressed.size() > 0){
            data_dict[dom_pulses.first] = boost::python::make_tuple(
                    stdVecToNumpyArray<double>(bin_values_list_compressed),
                    stdVecToNumpyArray<int>(bin_indices_list));
        }
    }

    return  data_dict;
}

template <typename T>
inline boost::python::tuple restructure_pulses(
                                  const boost::python::object& pulse_map_obj
                                ) {

    // Get pulse map
    I3RecoPulseSeriesMap& pulse_map = boost::python::extract<I3RecoPulseSeriesMap&>(pulse_map_obj);

    boost::python::dict dom_times_dict;
    boost::python::dict dom_charges_dict;

    std::vector<double> charges;
    std::vector<double> times;

    for (auto const& dom_pulses : pulse_map){
        std::vector<double> dom_charges;
        std::vector<double> dom_times;

        for (auto const& pulse : dom_pulses.second){
            dom_charges.push_back(pulse.GetCharge());
            dom_times.push_back(pulse.GetTime());
        }
        dom_times_dict[dom_pulses.first] = stdVecToNumpyArray<double>(dom_times);
        dom_charges_dict[dom_pulses.first] = stdVecToNumpyArray<double>(dom_charges);

        // extend total charges and times
        // reserve() is optional - just to improve performance
        charges.reserve(charges.size() + dom_charges.size());
        charges.insert(charges.end(), dom_charges.begin(), dom_charges.end());

        times.reserve(times.size() + dom_times.size());
        times.insert(times.end(), dom_times.begin(), dom_times.end());

    }
    boost::python::object charges_numpy = stdVecToNumpyArray<double>(charges);
    boost::python::object times_numpy = stdVecToNumpyArray<double>(times);

    return  boost::python::make_tuple(
                charges_numpy, times_numpy, dom_times_dict, dom_charges_dict );
}

I3RecoPulseSeriesMapPtr get_valid_pulse_map(
                            boost::python::object& frame_obj,
                            const boost::python::object& pulse_key_obj,
                            const boost::python::list& excluded_doms_obj,
                            const boost::python::object& partial_exclusion_obj,
                            const boost::python::object& verbose_obj){
    /*
    This method creates a new pulse series based on the given pulse series
    name, but excludes pulses and DOMs as specified with the excluded_doms
    and partial_exclusion.
    */

    // extract c++ data types from python objects
    I3Frame& frame = boost::python::extract<I3Frame&>(frame_obj);
    const std::string pulse_key =
        boost::python::extract<std::string>(pulse_key_obj);
    const bool partial_exclusion =
        boost::python::extract<bool>(partial_exclusion_obj);
    const bool verbose = boost::python::extract<bool>(verbose_obj);

    std::vector<std::string> excluded_doms;
    for (int i = 0; i < len(excluded_doms_obj); ++i){
        excluded_doms.push_back(
                boost::python::extract<std::string>(excluded_doms_obj[i]));
    }

    // keep track of removed number of pulses and DOMs
    unsigned int removed_doms = 0;
    unsigned int removed_pulses = 0;

    // get pulses
    const I3RecoPulseSeriesMap& pulses =
        frame.Get<I3RecoPulseSeriesMap>(pulse_key);

    // copy pulses so that we can modify them
    I3RecoPulseSeriesMap pulses_masked = I3RecoPulseSeriesMap(pulses);

    // ------------------------------------------------
    // remove all excluded DOMs (!= I3TimeWindowSeries)
    // ------------------------------------------------
    I3TimeWindowSeriesMap exclusions;
    for (const std::string& mapname: excluded_doms) {

            I3TimeWindowSeriesMapConstPtr exclusions_segment =
                frame.Get<I3TimeWindowSeriesMapConstPtr>(mapname);

            I3VectorOMKeyConstPtr excludedoms =
                frame.Get<I3VectorOMKeyConstPtr>(mapname);

            if (exclusions_segment && partial_exclusion) {
                /* These are TimeWindowSeries from which we have to remove
                pulses that lie within, since partial exclusion is true and
                we want to keep the rest of the pulses of a DOM.

                For now we will combine all of the provided TimeWindowSeries
                into a single I3TimeWindowSeriesMap: exclusions.
                */
                    for (I3TimeWindowSeriesMap::const_iterator i =
                        exclusions_segment->begin(); i !=
                        exclusions_segment->end(); i++){
                            exclusions[i->first] = exclusions[i->first] |
                                i->second;
                        }
            } else if (exclusions_segment && !partial_exclusion) {
                /* These are TimeWindowSeries, but since partial exclusion
                is false, we will remove all DOMs that have exclusion time
                windows defined.
                */
                    for (I3TimeWindowSeriesMap::const_iterator i =
                        exclusions_segment->begin(); i !=
                        exclusions_segment->end(); i++){
                            //exclusions[i->first].push_back(I3TimeWindow());
                            removed_doms += pulses_masked.erase(i->first);
                        }

            } else if (excludedoms) {
                /* These are whole DOMs to be ommited.
                Examples can be BrightDOMs which is a list of DOM OMKeys that
                will be completely removed.
                */
                    for (const OMKey& key: *excludedoms){
                            //exclusions[key].push_back(I3TimeWindow());
                            removed_doms += pulses_masked.erase(key);
                        }
            }
    }

    for (I3TimeWindowSeriesMap::const_iterator tws = exclusions.begin();
                tws != exclusions.end(); tws++){

        // ----------------------------
        // Get effective readout window
        // ----------------------------
        I3TimeWindowSeries effective_readouts;
        effective_readouts.push_back(I3TimeWindow());
        {
            // The effective readout windows are the set difference of the
            // global readout window and exclusion windows
            effective_readouts = effective_readouts & (~(tws->second));
        }
        // ----------------------------

        I3RecoPulseSeriesMap::const_iterator dom_pulse_ptr =
            pulses_masked.find(tws->first);

        // make sure key exists in pulses
        if (dom_pulse_ptr != pulses_masked.end()){

            // make iterator point at pulse series for the given OMKey
            I3RecoPulseSeries rps = dom_pulse_ptr->second;
            I3RecoPulseSeries masked_pulse_series = I3RecoPulseSeries();

            // reserve memory (Note: this reserves more than necessary)
            masked_pulse_series.reserve(rps.size());

            // define pointer to begin and end of pulse series
            auto begin(rps.cbegin()), end(rps.cend());
            for (auto &readout : effective_readouts) {
                auto pulse_time_sort = [](const I3RecoPulse &p, double t) {
                                                     return p.GetTime() < t; };
                begin = std::lower_bound(begin, end, readout.GetStart(),
                                         pulse_time_sort);
                end = std::lower_bound(begin, end, readout.GetStop(),
                                       pulse_time_sort);

                // insert valid pulses
                masked_pulse_series.insert(masked_pulse_series.end(),
                                           begin, end);

                // update new begin and end position
                begin = end;
                end = rps.cend();
            }

            // delete old pulse series and update with masked pulse series
            //pulses_masked.erase(tws->first);
            removed_pulses += rps.size() - masked_pulse_series.size();
            pulses_masked[tws->first] = masked_pulse_series;
        }
    }
    // ------------------------------------------------

    // write pulses to frame
    I3RecoPulseSeriesMapPtr fr_pulses =
        boost::make_shared<I3RecoPulseSeriesMap>(pulses_masked);
    //frame.Put(pulse_key + "_masked", fr_pulses);

    if (verbose){
        /*log_info(
            "[MaskPulses] Removed %d DOMs and %d additional pulses from %s",
            removed_doms, removed_pulses, pulse_key.c_str());*/
        std::cout << "[MaskPulses] Removed " << removed_doms <<" DOMs and "
                  << removed_pulses << " additional pulses from "
                  << pulse_key << std::endl;
    }

    return fr_pulses;
}


BOOST_PYTHON_MODULE(ext_boost)
{
    // numpy requires this
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    import_array();

    boost::python::def("restructure_pulses",
                       &restructure_pulses<double>);

    boost::python::def("get_valid_pulse_map",
                       &get_valid_pulse_map);

    boost::python::def("get_cascade_classification_data",
                       &get_cascade_classification_data<double>);
}
