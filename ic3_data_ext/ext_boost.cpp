#include <iostream>
#include <string>
#include <math.h>

#include "icetray/I3Frame.h"
#include "dataclasses/physics/I3RecoPulse.h"
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

// https://stackoverflow.com/questions/10701514/how-to-return-numpy-array-from-boostpython
boost::python::object stdVecToNumpyArray( std::vector<double> const& vec )
{
      npy_intp size = vec.size();

     /* const_cast is rather horrible but we need a writable pointer
        in C++11, vec.data() will do the trick
        but you will still need to const_cast
      */

      double * data = size ? const_cast<double *>(&vec[0])
        : static_cast<double *>(NULL);

    // create a PyObject * from pointer and data
      PyObject * pyObj = PyArray_SimpleNewFromData( 1, &size, NPY_DOUBLE, data );
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
        dom_times_dict[dom_pulses.first] = stdVecToNumpyArray(dom_times);
        dom_charges_dict[dom_pulses.first] = stdVecToNumpyArray(dom_charges);

        // extend total charges and times
        // reserve() is optional - just to improve performance
        charges.reserve(charges.size() + dom_charges.size());
        charges.insert(charges.end(), dom_charges.begin(), dom_charges.end());

        times.reserve(times.size() + dom_times.size());
        times.insert(times.end(), dom_times.begin(), dom_times.end());

    }
    boost::python::object charges_numpy = stdVecToNumpyArray(charges);
    boost::python::object times_numpy = stdVecToNumpyArray(times);

    return  boost::python::make_tuple(
                charges_numpy, times_numpy, dom_times_dict, dom_charges_dict );
}

void get_valid_pulse_map(boost::python::object& frame_obj,
                         const boost::python::object& pulse_key_obj,
                         const boost::python::list& excluded_doms_obj,
                         const boost::python::object& partial_exclusion_obj,
                         const boost::python::object& verbose_obj){

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
                            pulses_masked.erase(i->first);
                        }

            } else if (excludedoms) {
                /* These are whole DOMs to be ommited.
                Examples can be BrightDOMs which is a list of DOM OMKeys that
                will be completely removed.
                */
                    for (const OMKey& key: *excludedoms){
                            //exclusions[key].push_back(I3TimeWindow());
                            pulses_masked.erase(key);
                        }
            }
    }

    //for (const auto& tws: exclusions){
    for (I3TimeWindowSeriesMap::const_iterator tws = exclusions.begin();
                tws != exclusions.end(); tws++){
        std::cout << "Exclusion windows for OMKey " << tws->first << std::endl;

        // ----------------------------
        // Get effective readout window
        // ----------------------------
        I3TimeWindowSeries effective_readouts;
        effective_readouts.push_back(I3TimeWindow());
        {
            //I3TimeWindowSeriesMap::const_iterator tws =
            //    exclusions.find(i->first);
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
                auto pulse_time_sort = [](const I3RecoPulse &p, double t) { return p.GetTime() < t; };
                begin = std::lower_bound(begin, end, readout.GetStart(), pulse_time_sort);
                end = std::lower_bound(begin, end, readout.GetStop(), pulse_time_sort);

                std::cout << "\tChosen start: " << begin->GetTime()
                          << " end: " << end->GetTime() << std::endl;

                masked_pulse_series.insert(masked_pulse_series.end(), begin, end);

                // Now go through all valid pulses and add them
                for (auto pulse_iterator = begin; pulse_iterator != end;
                     pulse_iterator++){
                    std::cout << "\t\tPulse Charge: "
                              << pulse_iterator->GetCharge() << " Time: "
                              << pulse_iterator->GetTime() << std::endl;

                    //masked_pulse_series.push_back(pulse_iterator);
                    //masked_pulse_series.push_back(I3RecoPulse(pulse_iterator));
                }

                begin = end;
                end = rps.cend();
            }

            // delete old pulse series and update with masked pulse series
            pulses_masked.erase(tws->first);
            pulses_masked[tws->first] = masked_pulse_series;
        }

        for (const auto& tw : tws->second){
            std::cout << "\tStart: " << tw.GetStart()
                      << " End: " << tw.GetStop() << std::endl;
        }
    }
    // ------------------------------------------------

    // write pulses to frame
    I3RecoPulseSeriesMapPtr fr_pulses =
        boost::make_shared<I3RecoPulseSeriesMap>(pulses_masked);
    frame.Put(pulse_key + "_masked", fr_pulses);

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
}
