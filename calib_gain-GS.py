import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # suppress FutureWarnings from TensorFlow/Numpy compatibility
from EMCCDCalibration import CalibrateGainGS
from config import *  # load the default configuration settings, provides variable params
from tifffile import imread

"""
CalibrateGainGS has one required argument: a Python list containing a dictionary for each data set. Each dictionary 
should contain keys that specify.

samples:    Array of the samples. Each pixel will be analyzed individually with the ordering (t, x, y).
S0_map:     Value or array the same size as 'samples' (x, y) containing the fixed offset values S0 of each pixel. 
sigma_map:  Value or array the same size as 'samples' (x, y) containing the fixed readout noise of each pixel. 
ADU_map:    Value or array the same size as 'samples' (x, y) containing the fixed ADU conversion factor of each pixel.
gain_est:   Initial estimate of the gain value. Value or array size of 'samples' (x, y). If gain_est is 1, a Poisson
            noise model will be used and the gain will be considered fixed.

S0_map and sigma_map are determined from the mean and standard deviation of an image sequence acquired without signal at
the same exposure time and gain set-point as the samples. Each data set must have the same (x, y) dimensions, but may 
contain different numbers of samples (t). The initial intensity estimate will be based on sample data. 

An optional dictionary specifying non-default parameters (e.x. parameter ranges) can be passed to the CalibrateGainGS
class through the params keyword. See config.py for the required fields and default vales.
"""

"load samples"
data_set = [
    {  # samples for gain level #1
        'samples': imread('sample_data/gain-GS/1x-gain.tif'),
        'S0_map': imread('sample_data/gain-GS/1x-gain_no-signal_S0.tif'),
        'sigma_map': imread('sample_data/gain-GS/1x-gain_no-signal_sigma.tif'),
        'ADU_map': imread('sample_data/gain-GS/ADU_map.tif'),
        'gain_est': 1.0,
    },

    {  # samples for gain level #2
        'samples': imread('sample_data/gain-GS/5x-gain.tif'),
        'S0_map': imread('sample_data/gain-GS/5x-gain_no-signal_S0.tif'),
        'sigma_map': imread('sample_data/gain-GS/1x-gain_no-signal_sigma.tif'),
        'ADU_map': imread('sample_data/gain-GS/ADU_map.tif'),
        'gain_est': 5.0,
    },

    {  # samples for gain level #3
        'samples': imread('sample_data/gain-GS/25x-gain.tif'),
        'S0_map': imread('sample_data/gain-GS/25x-gain_no-signal_S0.tif'),
        'sigma_map': imread('sample_data/gain-GS/25x-gain_no-signal_sigma.tif'),
        'ADU_map': imread('sample_data/gain-GS/ADU_map.tif'),
        'gain_est': 25.0,
    },

    {  # samples for gain level #4
        'samples': imread('sample_data/gain-GS/100x-gain.tif'),
        'S0_map': imread('sample_data/gain-GS/100x-gain_no-signal_S0.tif'),
        'sigma_map': imread('sample_data/gain-GS/100x-gain_no-signal_sigma.tif'),
        'ADU_map': imread('sample_data/gain-GS/ADU_map.tif'),
        'gain_est': 100.0,
    },

    {  # samples for gain level #5
        'samples': imread('sample_data/gain-GS/300x-gain.tif'),
        'S0_map': imread('sample_data/gain-GS/300x-gain_no-signal_S0.tif'),
        'sigma_map': imread('sample_data/gain-GS/300x-gain_no-signal_sigma.tif'),
        'ADU_map': imread('sample_data/gain-GS/ADU_map.tif'),
        'gain_est': 300.0,
    },
]

"Initialize calibration class."
calib = CalibrateGainGS(data_set, params=params)

"Run ADU calibration on each pixel."
calib.run('MLE', output='sample_data/gain-GS/results.npz')
# calib.run('hierarchical', px_list=[[0, 0], [0, 1]], debug=True)  # example call for debugging
