"""
Default PyMC3 parameter ranges
"""

params = {
    'E_MIN': 1.0,  # minimum value of photoelectron estimate
    'E_MAX': 500000.0,  # maximum value of photoelectron estimate
    'ADU_MIN': 1.0,  # minimum value of ADU estimate
    'ADU_MAX': 15.0,  # maximum value of ADU estimate
    'GAIN_MIN': 2.0,  # minimum value of gain estimate
    'GAIN_MAX': 500.0,  # maximum value of gain estimate

    'MU_SIGMA': 25.0,  # non-central hierarchical parameters
    'SIGMA_SIGMA': 25.0,
}
