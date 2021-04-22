import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # suppress FutureWarnings from TensorFlow/Numpy compatibility
from EMCCDCalibration import CalibrateGainIS
from config import *  # load the default configuration settings, provides variable params
from tifffile import imread

"""
CalibrateGainIS has one required argument: a Python list containing a dictionary for each data set. Each dictionary 
should contain keys that specify.

samples:    Array of the samples. Each pixel will be analyzed individually with the ordering (t, x, y).
S0_map:     Value or array the same size as 'samples' (x, y) containing the fixed offset values S0 of each pixel. 
sigma_map:  Value or array the same size as 'samples' (x, y) containing the fixed readout noise of each pixel. 
ADU_map:    Value or array the same size as 'samples' (x, y) containing the fixed ADU conversion factor of each pixel.
gain_est:   Initial estimate of the gain value. Value or array size of 'samples' (x, y).
E_est:      (optional) Initial estimate of the intensity for the data set. Will be estimated from samples if not 
            specified. Value or array the same size as 'samples' (x, y). 

S0_map and sigma_map are determined from the mean and standard deviation of an image sequence acquired without signal at
the same exposure time and gain set-point as the samples. Multiple data sets may share the same S0_map, sigma_map, and 
ADU_est, but they should be specified in each entry. Each data set must have the same (x, y) dimensions, but may contain 
different numbers of samples (t).

An optional dictionary specifying non-default parameters (e.x. parameter ranges) can be passed to the CalibrateGainIS
class through the params keyword. See config.py for the required fields and default vales.
"""

"load samples"
data_set = [
    {  # samples for intensity level #1
        'samples': imread('sample_data/gain-IS/100x-gain_level-01.tif'),
        'S0_map': imread('sample_data/gain-IS/100x-gain_no-signal_S0.tif'),
        'sigma_map': imread('sample_data/gain-IS/100x-gain_no-signal_sigma.tif'),
        'ADU_map': imread('sample_data/gain-IS/ADU_map.tif'),
        'gain_est': 100.0,
    },

    {  # samples for intensity level #2
        'samples': imread('sample_data/gain-IS/100x-gain_level-02.tif'),
        'S0_map': imread('sample_data/gain-IS/100x-gain_no-signal_S0.tif'),
        'sigma_map': imread('sample_data/gain-IS/100x-gain_no-signal_sigma.tif'),
        'ADU_map': imread('sample_data/gain-IS/ADU_map.tif'),
        'gain_est': 100.0,
    },

    {  # samples for intensity level #3
        'samples': imread('sample_data/gain-IS/100x-gain_level-03.tif'),
        'S0_map': imread('sample_data/gain-IS/100x-gain_no-signal_S0.tif'),
        'sigma_map': imread('sample_data/gain-IS/100x-gain_no-signal_sigma.tif'),
        'ADU_map': imread('sample_data/gain-IS/ADU_map.tif'),
        'gain_est': 100.0,
    },

    {  # samples for intensity level #4
        'samples': imread('sample_data/gain-IS/100x-gain_level-04.tif'),
        'S0_map': imread('sample_data/gain-IS/100x-gain_no-signal_S0.tif'),
        'sigma_map': imread('sample_data/gain-IS/100x-gain_no-signal_sigma.tif'),
        'ADU_map': imread('sample_data/gain-IS/ADU_map.tif'),
        'gain_est': 100.0,
    },

    {  # samples for intensity level #5
        'samples': imread('sample_data/gain-IS/100x-gain_level-05.tif'),
        'S0_map': imread('sample_data/gain-IS/100x-gain_no-signal_S0.tif'),
        'sigma_map': imread('sample_data/gain-IS/100x-gain_no-signal_sigma.tif'),
        'ADU_map': imread('sample_data/gain-IS/ADU_map.tif'),
        'gain_est': 100.0,
    },

    {  # samples for intensity level #6
        'samples': imread('sample_data/gain-IS/100x-gain_level-06.tif'),
        'S0_map': imread('sample_data/gain-IS/100x-gain_no-signal_S0.tif'),
        'sigma_map': imread('sample_data/gain-IS/100x-gain_no-signal_sigma.tif'),
        'ADU_map': imread('sample_data/gain-IS/ADU_map.tif'),
        'gain_est': 100.0,
    },
]

"Initialize calibration class."
calib = CalibrateGainIS(data_set, params=params)

"Run ADU calibration on each pixel."
calib.run(output='sample_data/gain-IS/results.npz')
# calib.run(px_list=[[0, 0], [0, 1]], debug=True)  # example call for debugging
