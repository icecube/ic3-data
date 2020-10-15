/******************************************************
Time critical functions for the Deep Learning-based
reconstruction (DNN_reco) are written in C++ and
wrapped with boost python.
******************************************************/
#include <iostream>
#include <string>
#include <math.h>

#include "icetray/I3Frame.h"
#include "icetray/OMKey.h"
#include "dataclasses/physics/I3RecoPulse.h"
#include "dataclasses/physics/I3Particle.h"
#include "dataclasses/physics/I3MCTree.h"
#include "dataclasses/geometry/I3Geometry.h"
#include "dataclasses/geometry/I3OMGeo.h"
#include "dataclasses/I3Map.h"
#include "dataclasses/I3TimeWindow.h"
#include "phys-services/I3Calculator.h"
#include <boost/version.hpp>
#include <boost/python.hpp>

#include "utils.cpp"

/*
Depending on the boost version, we need to use numpy differently.
Prior to Boost 1.63 (BOOST_VERSION == 106300) numpy was not directly
included in boost/python

See answers and discussion provided here:
https://stackoverflow.com/questions/10701514/how-to-return-numpy-array-from-boostpython
*/
#if BOOST_VERSION < 106500
    #include "numpy/npy_3kcompat.h"
    typedef typename boost::python::numeric::array pyndarray;
    namespace arrayFunc = boost::python::numeric;
#else
    #include <boost/python/numpy.hpp>
    typedef typename boost::python::numpy::ndarray pyndarray;
    namespace arrayFunc = boost::python::numpy;
    namespace bn = boost::python::numpy;
#endif



/******************************************************
 Helper-Functions to return numpy arrays
******************************************************/

// https://stackoverflow.com/questions/10701514/how-to-return-numpy-array-from-boostpython
#if BOOST_VERSION < 106500

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
          pyndarray arr( handle );

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
          pyndarray arr( handle );

        /* The problem of returning arr is twofold: firstly the user can modify
          the data which will betray the const-correctness
          Secondly the lifetime of the data is managed by the C++ API and not the
          lifetime of the numpy array whatsoever. But we have a simple solution..
         */

           return arr.copy(); // copy the object. numpy owns the copy now.
      }
#else
    template <typename T>
    boost::python::object stdVecToNumpyArray( std::vector<T> const& vec )
    {
        Py_intptr_t shape[1] = { vec.size() };
        bn::ndarray result = bn::zeros(1, shape, bn::dtype::get_builtin<double>());
        std::copy(vec.begin(), vec.end(),
                  reinterpret_cast<double*>(result.get_data()));
        return result;
    }
    // todo: change this to templated function and remove duplicate
    template <std::size_t N, typename T>
    boost::python::object stdArrayToNumpyArray( std::array<T, N> const& vec )
    {
        Py_intptr_t shape[1] = { vec.size() };
        bn::ndarray result = bn::zeros(1, shape, bn::dtype::get_builtin<double>());
        std::copy(vec.begin(), vec.end(),
                  reinterpret_cast<double*>(result.get_data()));
        return result;
    }
#endif

/******************************************************
 Functions with pybinding for python-based usage
******************************************************/
template <typename T>
inline boost::python::tuple get_reduced_summary_statistics_data(
                                  const boost::python::object& pulse_map_obj,
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

        // check if pulses are present
        if (dom_pulses.second.size() == 0){
            continue;
        }

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

        unsigned int n_total = n_bins;
        if (add_dom_info){
            n_total += 2;
            bin_values_list.push_back(distance);
            bin_values_list.push_back(unc_time);
        }

        // compress output: throw out extremely small values
        boost::python::list bin_exclusions_list; // empty dummy exclusions
        boost::python::list bin_indices_list;
        boost::python::list bin_values_list_compressed;
        std::vector<double>::iterator values_iterator =
                bin_values_list.begin();
        bool found_at_least_one_element = false;
        for (unsigned int i=0; i < n_total; i++){
            if (*values_iterator > 1e-7){
                bin_values_list_compressed.append(*values_iterator);
                bin_indices_list.append(i);
                found_at_least_one_element = true;
            }
            values_iterator += 1;
        }

        if (found_at_least_one_element){
            data_dict[dom_pulses.first] = boost::python::make_tuple(
                    bin_values_list_compressed, bin_indices_list,
                    bin_exclusions_list);
        }
    }

    return  data_dict;
}


// equivalent to numpy searchsorted
template <typename T>
unsigned int find_index(std::vector<T> array, T value){
    unsigned int index = 0;
    while (value > array[index] && index < array.size()){
        index++;
    }
    return index;
}

template <typename T>
inline boost::python::dict get_mc_tree_input_data_dict(
                                  boost::python::object& frame_obj,
                                  boost::python::object& angle_bins_obj,
                                  boost::python::object& distance_bins_obj,
                                  const double distance_cutoff,
                                  const double energy_cutoff,
                                  const bool add_distance
                                ) {

    // extract objects
    I3Frame& frame = boost::python::extract<I3Frame&>(frame_obj);

    const unsigned int num_dist_bins =  len(distance_bins_obj) - 1;
    const unsigned int num_angle_bins =  len(angle_bins_obj) - 1;
    const unsigned int num_bins =
        1 + add_distance + num_dist_bins * num_angle_bins;

    std::vector<double> angle_bins;
    std::vector<double> distance_bins;
    for (int i = 0; i < num_angle_bins + 1; ++i){
        angle_bins.push_back(
                boost::python::extract<double>(angle_bins_obj[i]));
    }
    for (int i = 0; i < num_dist_bins + 1; ++i){
        distance_bins.push_back(
                boost::python::extract<double>(distance_bins_obj[i]));
    }

    // Get geometry map
    const I3OMGeoMap& omgeo = frame.Get<I3Geometry>("I3Geometry").omgeo;

    // Get I3MCTree map
    const I3MCTree& mctree = frame.Get<I3MCTree>("I3MCTree");

    // create data array for DOMs [shape: (86, 60, num_bins)]
    double init_dist_value = 0.;
    if (add_distance){
        init_dist_value = std::numeric_limits<double>::max();
    }
    std::vector<double>  dom_data[86][60];
    for(size_t string = 1; string <= 86; string++){
        for(size_t dom = 1; dom <= 60; dom++){
            dom_data[string-1][dom-1] = std::vector<double>();

            // intialize values
            dom_data[string-1][dom-1].push_back(init_dist_value);
            for(size_t i=1; i < num_bins; i++){
                dom_data[string-1][dom-1].push_back(0);
            }
        }
    }

    // walk through energy losses and calculate data
    for (auto const& loss : mctree){

        // skip energy loss if it is not one of the allowed types
        bool is_allowed_type = false;
        switch(loss.GetType()){
            case I3Particle::ParticleType::NuclInt:
            case I3Particle::ParticleType::PairProd:
            case I3Particle::ParticleType::Brems:
            case I3Particle::ParticleType::DeltaE:
            case I3Particle::ParticleType::EMinus:
            case I3Particle::ParticleType::EPlus:
            case I3Particle::ParticleType::Hadrons: is_allowed_type=true;
                                                    break;
            default: is_allowed_type=false;
        }
        if (!is_allowed_type || loss.GetEnergy() < energy_cutoff ||
            loss.GetPos().Magnitude() > 2000){
            continue;
        }

        double min_distance = std::numeric_limits<double>::max();
        OMKey min_omkey = OMKey();

        // walk through DOMs and calculate data
        for(size_t string = 1; string <= 86; string++){
            for(size_t dom = 1; dom <= 60; dom++){

                const OMKey& om_key = OMKey(string, dom);

                // get position of DOM
                const I3Position& pos = omgeo.at(om_key).position;

                // calculate distance and opening angle to DOM
                const I3Position& diff = pos - loss.GetPos();
                I3Particle diff_p = I3Particle();
                diff_p.SetDir(diff.GetX(), diff.GetY(), diff.GetZ());
                const double angle = I3Calculator::Angle(diff_p, loss);
                const double distance = diff.Magnitude();

                // sort loss energy to correct bin index
                unsigned int index_angle =
                    find_index<double>(angle_bins, angle) - 1;
                unsigned int index_dist =
                    find_index<double>(distance_bins, distance) - 1;

                // check if it is out of bounds
                bool out_of_bounds = false;
                if (index_angle < 0 || index_angle >= num_angle_bins){
                    out_of_bounds = true;
                }
                if (index_dist < 0 || index_dist >= num_dist_bins){
                    out_of_bounds = true;
                }

                if (!out_of_bounds){
                    // calculate index
                    unsigned int index = add_distance +
                        index_dist * num_angle_bins + index_angle;

                    // accumulate energy
                    dom_data[string - 1][dom - 1][index] += loss.GetEnergy();
                }

                // check if distance is closest so far
                if (add_distance && distance < dom_data[string - 1][dom - 1][0]){
                    dom_data[string - 1][dom - 1][0] = distance;
                }

                if (distance < min_distance){
                    min_distance = distance;
                    min_omkey = om_key;
                }
            }
        }

        // add energy loss to closest DOM within distance_cutoff
        if (loss.GetEnergy() > energy_cutoff){
            if (min_distance < distance_cutoff){
                dom_data[min_omkey.GetString() - 1]
                         [min_omkey.GetOM() - 1]
                         [num_bins-1] += loss.GetEnergy();
            }
        }
    }

    // empty dummy exclusions
    boost::python::list bin_exclusions_list;

    boost::python::dict data_dict;
    for(size_t string = 1; string <= 86; string++){
        for(size_t dom = 1; dom <= 60; dom++){
            const OMKey& om_key = OMKey(string, dom);

            // create value and bin indices lists
            boost::python::list bin_values_list;
            boost::python::list bin_indices_list;
            boost::python::list bin_values_list_compressed;

            std::vector<double>::iterator values_iterator =
                dom_data[string - 1][dom - 1].begin();
            bool found_at_least_one_element = false;

            for (unsigned int i=0; i < num_bins; i++){
                if (*values_iterator > 1e-7){
                    bin_values_list_compressed.append(*values_iterator);
                    bin_indices_list.append(i);
                    found_at_least_one_element = true;
                }
                values_iterator += 1;
            }

            if (found_at_least_one_element){
                data_dict[om_key] = boost::python::make_tuple(
                        bin_values_list_compressed, bin_indices_list,
                        bin_exclusions_list);
            }
        }
    }
    return  data_dict;
}


boost::python::list get_time_bin_exclusions(
                            boost::python::object& frame_obj,
                            boost::python::object& om_key_obj,
                            const boost::python::list& time_bins,
                            const boost::python::list& excluded_doms_obj,
                            const boost::python::object& partial_exclusion_obj,
                            const double time_offset
                            ){
    /*
    This method creates a list of bin indices of a given DOM that are excluded
    as specified with the excluded_doms and partial_exclusion.
    */

    // extract c++ data types from python objects
    I3Frame& frame = boost::python::extract<I3Frame&>(frame_obj);
    const OMKey& om_key = boost::python::extract<OMKey&>(om_key_obj);
    const bool partial_exclusion =
        boost::python::extract<bool>(partial_exclusion_obj);

    std::vector<std::string> excluded_doms;
    for (int i = 0; i < len(excluded_doms_obj); ++i){
        excluded_doms.push_back(
                boost::python::extract<std::string>(excluded_doms_obj[i]));
    }

    // create a list for the bin exclusions
    boost::python::list bin_exclusions_list;
    std::set<int> bin_exclusions_set;

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
                        if (i->first == om_key){
                            exclusions[i->first] = exclusions[i->first] |
                                i->second;
                        }
                    }
            } else if (exclusions_segment && !partial_exclusion) {
                /* These are TimeWindowSeries, but since partial exclusion
                is false, we will remove all DOMs that have exclusion time
                windows defined.
                */
                for (I3TimeWindowSeriesMap::const_iterator i =
                    exclusions_segment->begin(); i !=
                    exclusions_segment->end(); i++){
                        if (i->first == om_key) {
                            // remove complete DOM
                            bin_exclusions_list.append(-1);
                            return bin_exclusions_list;
                        }
                    }

            } else if (excludedoms) {
                /* These are whole DOMs to be ommited.
                Examples can be BrightDOMs which is a list of DOM OMKeys that
                will be completely removed.
                */
                for (const OMKey& key: *excludedoms){
                        if (key == om_key) {
                            // remove complete DOM
                            bin_exclusions_list.append(-1);
                            return bin_exclusions_list;
                        }
                    }
            }
    }

    for (I3TimeWindowSeriesMap::const_iterator tws = exclusions.begin();
                tws != exclusions.end(); tws++){

        for (auto &readout : tws->second) {
            for (int index = 0; index < len(time_bins) - 1; ++index){
                if (readout.GetStart() - time_offset < time_bins[index+1] &&
                    readout.GetStop() - time_offset > time_bins[index] ){
                    bin_exclusions_set.insert(index);
                }
            }
        }

    }

    for(auto const index : bin_exclusions_set){
        bin_exclusions_list.append(index);
    }

    return bin_exclusions_list;
}


template <typename T>
inline boost::python::tuple restructure_pulses(
                                  const boost::python::object& pulse_map_obj
                                ) {

    // Get pulse map
    const I3RecoPulseSeriesMap& pulse_map = boost::python::extract<I3RecoPulseSeriesMap&>(pulse_map_obj);

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

/* Benchmark test:
Create an input tensor of shape [86, 60] which holds the total charge per DOM.
*/
inline bn::ndarray get_charge_input_data(
            const boost::python::object& pulse_map_obj
        ) {

    // Get pulse map
    const I3RecoPulseSeriesMap& pulse_map = boost::python::extract<I3RecoPulseSeriesMap&>(pulse_map_obj);

    boost::python::tuple shape = boost::python::make_tuple(86, 60);
    bn::dtype dtype = bn::dtype::get_builtin<float>();
    bn::ndarray matrix = bn::zeros(shape, dtype);

    // loop over pulses and accumulate charge
    for (auto const& dom_pulses : pulse_map){

        int om_num = dom_pulses.first.GetOM() - 1;
        int string_num = dom_pulses.first.GetString() - 1;

        if (om_num < 60){
            for (auto const& pulse : dom_pulses.second){
                matrix[string_num][om_num] += pulse.GetCharge();
            }
        }
    }
    return  matrix;
}

inline bn::ndarray get_charge_input_data2(
            boost::python::object& frame_obj,
            const boost::python::object& pulse_key_obj
        ) {

    // extract c++ data types from python objects
    I3Frame& frame = boost::python::extract<I3Frame&>(frame_obj);
    const std::string pulse_key =
        boost::python::extract<std::string>(pulse_key_obj);

    // get pulses
    const I3RecoPulseSeriesMap& pulse_map =
        frame.Get<I3RecoPulseSeriesMap>(pulse_key);

    boost::python::tuple shape = boost::python::make_tuple(86, 60);
    bn::dtype dtype = bn::dtype::get_builtin<float>();
    bn::ndarray matrix = bn::zeros(shape, dtype);

    // loop over pulses and accumulate charge
    for (auto const& dom_pulses : pulse_map){

        int om_num = dom_pulses.first.GetOM() - 1;
        int string_num = dom_pulses.first.GetString() - 1;

        if (om_num < 60){
            for (auto const& pulse : dom_pulses.second){
                matrix[string_num][om_num] += pulse.GetCharge();
            }
        }
    }
    return  matrix;
}

static boost::python::list  get_charge_input_data3(
            boost::python::object& frame_obj,
            const boost::python::object& pulse_key_obj
        ) {

    // extract c++ data types from python objects
    I3Frame& frame = boost::python::extract<I3Frame&>(frame_obj);
    const std::string pulse_key =
        boost::python::extract<std::string>(pulse_key_obj);

    // get pulses
    const I3RecoPulseSeriesMap& pulse_map =
        frame.Get<I3RecoPulseSeriesMap>(pulse_key);

    // create matrix which is a list of lists
    boost::python::list matrix;

    for (unsigned int s = 0; s < 86; s++){
        boost::python::list string_list;
        for (unsigned int d = 0; d < 60; d++){
            string_list.append(0.);
        }
        matrix.append(string_list);
    }

    // loop over pulses and accumulate charge
    for (auto const& dom_pulses : pulse_map){

        int om_num = dom_pulses.first.GetOM() - 1;
        int string_num = dom_pulses.first.GetString() - 1;

        if (om_num < 60){
            for (auto const& pulse : dom_pulses.second){
                matrix[string_num][om_num] += pulse.GetCharge();
            }
        }
    }
    return  matrix;
}

static bn::ndarray  get_charge_input_data4(
            boost::python::object& frame_obj,
            const boost::python::object& pulse_key_obj
        ) {

    // extract c++ data types from python objects
    I3Frame& frame = boost::python::extract<I3Frame&>(frame_obj);
    const std::string pulse_key =
        boost::python::extract<std::string>(pulse_key_obj);

    // get pulses
    const I3RecoPulseSeriesMap& pulse_map =
        frame.Get<I3RecoPulseSeriesMap>(pulse_key);

    // create matrix
    float matrix[86][60];

    for (unsigned int s = 0; s < 86; s++){
        for (unsigned int d = 0; d < 60; d++){
            matrix[s][d] = 0.;
        }
    }

    // loop over pulses and accumulate charge
    for (auto const& dom_pulses : pulse_map){

        int om_num = dom_pulses.first.GetOM() - 1;
        int string_num = dom_pulses.first.GetString() - 1;

        if (om_num < 60){
            for (auto const& pulse : dom_pulses.second){
                matrix[string_num][om_num] += pulse.GetCharge();
            }
        }
    }

    // create numpy array
    bn::ndarray py_array = bn::from_data(
        matrix,
        bn::dtype::get_builtin<float>(),
        boost::python::make_tuple(86, 60),
        boost::python::make_tuple(sizeof(float), 86*sizeof(float)),
        boost::python::object());

    return  py_array.copy(); // python owns the copy now
}

static void fill_charge_input_data(
            boost::python::object& frame_obj,
            boost::python::numpy::ndarray& input,
            const boost::python::object& pulse_key_obj
        ) {

    // extract c++ data types from python objects
    I3Frame& frame = boost::python::extract<I3Frame&>(frame_obj);
    const std::string pulse_key =
        boost::python::extract<std::string>(pulse_key_obj);

    // get a pointer to the input data
    float* input_ptr = reinterpret_cast<float*>(input.get_data());


    // get pulses
    const I3RecoPulseSeriesMap& pulse_map =
        frame.Get<I3RecoPulseSeriesMap>(pulse_key);

    /*// initialize with zeros
    for (unsigned int s = 0; s < 86; s++){
        for (unsigned int d = 0; d < 60; d++){
            input_ptr[s][d] = 0.;
        }
    }*/

    // loop over pulses and accumulate charge
    for (auto const& dom_pulses : pulse_map){

        unsigned int om_num = dom_pulses.first.GetOM() - 1;
        unsigned int string_num = dom_pulses.first.GetString() - 1;
        unsigned int offset = 60*string_num + om_num;

        if (om_num < 60){
            float dom_charge = 0;
            for (auto const& pulse : dom_pulses.second){
                //input_ptr[string_num][om_num] += pulse.GetCharge();
                dom_charge += pulse.GetCharge();
            }
            input_ptr[offset] += dom_charge;
        }
    }
}

/* Combine DOM exclusions into a single vector of DOMs and a single
TimeWindowsSeriesMap
Returns:
    I3VectorOMKey: the combined DOM exclusions
    I3TimeWindowSeriesMap: the combined time window exclusions
*/
boost::python::tuple combine_exclusions(
                    boost::python::object& frame_obj,
                    const boost::python::list& excluded_doms_obj,
                    const boost::python::object& partial_exclusion_obj){

    // extract c++ data types from python objects
    I3Frame& frame = boost::python::extract<I3Frame&>(frame_obj);
    const bool partial_exclusion =
        boost::python::extract<bool>(partial_exclusion_obj);

    std::vector<std::string> exclusion_series;
    for (int i = 0; i < len(excluded_doms_obj); ++i){
        exclusion_series.push_back(
                boost::python::extract<std::string>(excluded_doms_obj[i]));
    }

    I3VectorOMKey exclusion_doms;
    I3TimeWindowSeriesMap exclusion_tws;
    for (const std::string& mapname: exclusion_series) {

        I3TimeWindowSeriesMapConstPtr exclusions_segment =
            frame.Get<I3TimeWindowSeriesMapConstPtr>(mapname);

        I3VectorOMKeyConstPtr excludedoms =
            frame.Get<I3VectorOMKeyConstPtr>(mapname);

        if (exclusions_segment && partial_exclusion) {
            /* These are TimeWindowSeries from which we have to remove
            pulses that lie within, since partial exclusion is true and
            we want to keep the rest of the pulses of a DOM.

            For now we will combine all of the provided TimeWindowSeries
            into a single I3TimeWindowSeriesMap: exclusion_tws.
            */
            for (I3TimeWindowSeriesMap::const_iterator i =
                exclusions_segment->begin(); i !=
                exclusions_segment->end(); i++){
                    exclusion_tws[i->first] = exclusion_tws[i->first] |
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
                    exclusion_doms.push_back(i->first);
            }

        } else if (excludedoms) {
            /* These are whole DOMs to be ommited.
            Examples can be BrightDOMs which is a list of DOM OMKeys that
            will be completely removed.
            */
            for (const OMKey& key: *excludedoms){
                    exclusion_doms.push_back(key);
            }
        }
    }
    return  boost::python::make_tuple(exclusion_doms, exclusion_tws);
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
    #if BOOST_VERSION < 106500
        // Specify that py::numeric::array should refer to the Python type
        // numpy.ndarray (rather than the older Numeric.array).
        boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
        // numpy requires this
        import_array();

    #else
        bn::initialize();

    #endif

    boost::python::def("restructure_pulses",
                       &restructure_pulses<double>);

    boost::python::def("get_charge_input_data",
                       &get_charge_input_data);

    boost::python::def("get_charge_input_data2",
                       &get_charge_input_data2);

    boost::python::def("get_charge_input_data3",
                       &get_charge_input_data3);

    boost::python::def("get_charge_input_data4",
                       &get_charge_input_data4);

    boost::python::def("fill_charge_input_data",
                       &fill_charge_input_data);

    boost::python::def("get_valid_pulse_map",
                       &get_valid_pulse_map);

    boost::python::def("combine_exclusions",
                       &combine_exclusions);

    boost::python::def("get_time_bin_exclusions",
                       &get_time_bin_exclusions);

    boost::python::def("get_reduced_summary_statistics_data",
                       &get_reduced_summary_statistics_data<double>);

    boost::python::def("get_cascade_classification_data",
                       &get_cascade_classification_data<double>);

    boost::python::def("get_mc_tree_input_data_dict",
                       &get_mc_tree_input_data_dict<double>);
}
