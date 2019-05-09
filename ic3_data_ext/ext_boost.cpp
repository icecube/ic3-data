#include <iostream>
#include <string>
#include <math.h>

#include "dataclasses/physics/I3RecoPulse.h"
#include "dataclasses/I3Map.h"
#include <boost/python.hpp>


/******************************************************
Time critical functions for the Deep Learning-based
reconstruction (DNN_reco) are written in C++ and
wrapped with boost python.
******************************************************/

// Actual C++ snippets. See docstrings at the bottom for argument info.

template <typename T>
inline boost::python::tuple restructure_pulse_map(
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
}


BOOST_PYTHON_MODULE(ext_boost)
{
    boost::python::def("restructure_pulse_map",
                       &restructure_pulse_map<double>);
}
