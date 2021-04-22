Essential files:

config.py:  Default settings. Primarily defining parameter limits. 
NoiseModel_TFv1.py:  TensorFlow implementation of EMCCD noise model. 
NoiseModelWrapper.py:  Theanos wrapper of TensorFlow model. Used to interface PyMC3 with TF for MAP or MCMC. 

Example file:
calib_ADU.py:  Example setup for running ADU calibration.
calib_gain-IS.py:  Example setup for running gain calibration using the intensity series approach.
calib_gain-GS.py:  Example setup for running gain calibration using the gain series approach.
