#include <iostream>
#include <string>
#include <math.h>

#include "dataclasses/physics/I3RecoPulse.h"
#include "dataclasses/I3Map.h"
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

        for (unsigned int i=0; i < dom_pulses.second.size(); i++ ){
            dom_charges.push_back(dom_pulses.second.at(i).GetCharge());
            dom_times.push_back(dom_pulses.second.at(i).GetTime());

            charges.push_back(dom_pulses.second.at(i).GetCharge());
            times.push_back(dom_pulses.second.at(i).GetTime());
        }
        dom_times_dict[dom_pulses.first] = stdVecToNumpyArray(dom_times);
        dom_charges_dict[dom_pulses.first] = stdVecToNumpyArray(dom_charges);

    }
    boost::python::object charges_numpy = stdVecToNumpyArray(charges);
    boost::python::object times_numpy = stdVecToNumpyArray(times);

    return  boost::python::make_tuple(
                charges_numpy, times_numpy, dom_times_dict, dom_charges_dict );
}


BOOST_PYTHON_MODULE(ext_boost)
{
    // numpy requires this
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    import_array();

    boost::python::def("restructure_pulses",
                       &restructure_pulses<double>);
}
