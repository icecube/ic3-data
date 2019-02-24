import importlib
import numpy as np


def print_warning(msg):
    """Print Warning in yellow color.

    Parameters
    ----------
    msg : str
        String to print.
    """
    print('\033[93m' + msg + '\033[0m')


def load_class(full_class_string):
    """
    dynamically load a class from a string

    Parameters
    ----------
    full_class_string : str
        The full class string to the given python clas.
        Example:
            my_project.my_module.my_class

    Returns
    -------
    python class
        PYthon class defined by the 'full_class_string'
    """

    class_data = full_class_string.split(".")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]

    module = importlib.import_module(module_path)
    # Finally, we retrieve the Class
    return getattr(module, class_str)


def weighted_quantile(x, weights, quantile=0.68):
    """Compute weighted quantile

    Parameters
    ----------
    x : list or numpy.ndarray
        The data for which to compute the quantile
    weights : list or numpy.ndarray
        The weights for x.
    quantile : float, optional
        The quantile to compute.

    Returns
    -------
    float
        The weighted quantile
    """
    if weights is None:
        weights = np.ones_like(x)

    x = np.asarray(x)
    weights = np.asarray(weights)

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    weights_sorted = weights[sorted_indices]
    cum_weights = np.cumsum(weights_sorted) / np.sum(weights)
    mask = cum_weights >= quantile

    return x_sorted[mask][0]


def weighted_std(x, weights=None):
    """"
    Weighted std deviation.

    Source
    ------
    http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf

    returns 0 if len(x)==1

    Parameters
    ----------
    x : list or numpy.ndarray
        Description
    weights : None, optional
        Description

    Returns
    -------
    float
        Weighted standard deviation
    """
    if len(x) == 1:
        return 0

    if weights is None:
        return np.std(x, ddof=1)

    x = np.asarray(x)
    weights = np.asarray(weights)

    w_mean_x = np.average(x, weights=weights)
    n = len(weights[weights != 0])

    s = n * np.sum(weights*(x - w_mean_x)*(x - w_mean_x)) / ((n - 1) *
                                                             np.sum(weights))
    return np.sqrt(s)
