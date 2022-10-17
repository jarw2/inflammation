"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.
"""

import numpy as np
import inflammation
import os
import inspect

def get_data_dir():
    """get default directory holding data files"""
    return os.path.dirname(inspect.getfile(inflammation)) + '/data'

def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load. If it is not an absolute path the
    file is assumed to be in the default data directory
    """
    if not os.path.isabs(filename):
        filename = get_data_dir() + '/' + filename

    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2d inflammation data array."""
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2d inflammation data array."""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2d inflammation data array."""
    return np.min(data, axis=0)

def patient_normalise(data):
    """
    Normalise patient data between 0 and 1 of a 2D inflammation data array.
    
    Inf values are set to zero
    NaN are ignored, and normalised to zero
    Negative values are clipped to zero
    """

    if not isinstance(data, np.ndarray):
        raise TypeError('Data should be an ndarray')

    if np.any(data < 0):
        raise ValueError('Inflammation values should be non-negative')

    data[data<0] = 0 # Clip negative values to zero
    data[np.isinf(data)] = 0 # Replace infinite values by zeros
    max_for_each_patient = np.max(data, axis=1) # Changed for axis=0 to axis=1 in debugging

    with np.errstate(divide='ignore', invalid='ignore'): # Ignoring division by zero errors
        normalised_data = data / max_for_each_patient[:, np.newaxis]
    normalised_data[np.isnan(normalised_data)] = 0 # Replace NaNs by zeros
    return normalised_data



# TODO(lesson-design) Add Patient class
# TODO(lesson-design) Implement data persistence
# TODO(lesson-design) Add Doctor class
