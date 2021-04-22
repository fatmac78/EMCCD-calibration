import tensorflow as tf
from NoiseModelWrapper import LogLikelihood
from NoiseModel_TFv1 import NoiseModel  # Tensorflow v2, using v1 compatibility (eager execution disabled)
import pymc3 as pm
import theano.tensor as tt
import numpy as np
from config import *  # load the default configuration settings, provides variable params


class CalibrateBase:
    """
    Base class for calibration. Primarily supplies a common __init__()
    """

    def __init__(self, data_set, params=params):
        """
        Initialize class.
        """

        "Store data set"
        self.data_set = data_set
        self.params = params

        "Convert params specified as values into arrays."
        sx, sy = self.data_set[0]['samples'].shape[1], self.data_set[0]['samples'].shape[2]
        for sample_set in self.data_set:
            for key in ['S0_map', 'sigma_map', 'ADU_map', 'ADU_est', 'gain_est', 'E_est']:
                if key in sample_set.keys():
                    if not isinstance(sample_set[key], np.ndarray):
                        sample_set[key] = np.ones((sx, sy)) * sample_set[key]

        "Initialize TensorFlow session"
        self.session = tf.compat.v1.InteractiveSession()


class CalibrateADU(CalibrateBase):
    """
    Calibrate analogue to digital unit (ADU) conversion factor using an intensity-series data set.
    """

    def run(self, output='results.npz', px_list=None, debug=False):
        """
        Calibrate ADU for each pixel in data set and save results.

        px_list:    List of pixels to analyze. Overrides cycling over all pixels.
        debug:      Boolean. Toggles save results.
        """

        "Setup noise models for each data set"
        noise_models = list()
        for sample_set in self.data_set:

            "Use intensity estimate, if provided; otherwise, calculate from samples."
            if 'E_est' not in sample_set.keys():
                sample_set['E_est'] = (np.nanmean(sample_set['samples'], 0) - sample_set['S0_map']) * sample_set['ADU_est']

            "Initialize a TensorFlow model for data set. (0, 0) pixel values used only for initialization."
            noise_models.append(NoiseModel(sample_set['S0_map'][0, 0],
                                           sample_set['sigma_map'][0, 0],
                                           1,  # gain = 1 for ADU calibration
                                           sample_set['ADU_est'][0, 0],
                                           sample_set['E_est'][0, 0],
                                           sample_set['samples'][:, 0, 0],
                                           diff_var=['p_e', 'ADU'], session=self.session))

        sx, sy = self.data_set[0]['samples'].shape[1], self.data_set[0]['samples'].shape[2]

        """
        PyMC3 section.
        """

        with pm.Model() as m:

            "Initialize fit variables (E_i and ADU)."
            pm_Es = []
            for idx, sample_set in enumerate(self.data_set):
                pm_Es.append(pm.Uniform('E_{:d}'.format(idx), self.params['E_MIN'], self.params['E_MAX'],
                                        testval=sample_set['E_est'][0, 0]))
            pm_ADU = pm.Uniform('ADU', self.params['ADU_MIN'], self.params['ADU_MAX'],
                                testval=self.data_set[0]['ADU_est'][0, 0])

            "Construct likelihood objects."
            thetas = []  # list holding the PyMC3 variables required for each individual noise model
            noise_models_wrapped = []  # list of wrapped noise models
            pm_logps = []  # list of PyMC3 logp functions
            for idx, noise_model in enumerate(noise_models):
                thetas.append(tt.as_tensor_variable([pm_Es[idx], pm_ADU]))
                noise_models_wrapped.append(LogLikelihood(noise_model, [noise_model.p_e, noise_model.ADU]))
                pm_logps.append(pm.Potential('likelihood_{:d}'.format(idx), noise_models_wrapped[idx](thetas[idx])))

            "Collate variable names (only those of interest for results)."
            var_names = [val.name for val in m.free_RVs if '__' not in val.name] + [val.name for val in m.deterministics
                                                                                    if '__' not in val.name]

            "Initialize results."
            results = dict()
            for var in var_names:
                results[var] = np.zeros((sx, sy)) * np.nan

            "Loop over pixels."
            if not px_list:  # if px_list not specified, loop over all pixels
                px_list = list(np.ndindex(sx, sy))
            for [idx_i, idx_j] in px_list:

                "Load values for current pixel."
                for sample_set, noise_model in zip(self.data_set, noise_models):
                    noise_model.samples.load(sample_set['samples'][:, idx_i, idx_j])
                    noise_model.S0.load(sample_set['S0_map'][idx_i, idx_j])
                    noise_model.sigma.load(sample_set['sigma_map'][idx_i, idx_j])
                    noise_model.p_e.load(sample_set['E_est'][idx_i, idx_j])
                    noise_model.ADU.load(sample_set['ADU_est'][idx_i, idx_j])
                    self.session.run(noise_model.ln_likelihood)

                "Optimize parameters."
                map_results = pm.find_MAP()

                "Report MAP results."
                print('({:d}, {:d})'.format(idx_i, idx_j))
                for key in sorted(var_names):
                    print('{}: {:.5f}'.format(key, map_results[key]))
                    results[key][idx_i, idx_j] = map_results[key]

                "save results"
                if not debug:
                    np.savez(output, **results)


class CalibrateGainIS(CalibrateBase):
    """
    Calibrate a single gain set-point using an intensity-series data set.
    """

    def run(self, output='results.npz', px_list=None, debug=False):
        """
        Calibrate gain for each pixel in data set and save results.

        px_list:    List of pixels to analyze. Overrides cycling over all pixels.
        debug:      Boolean. Toggles save results.
        """

        "Setup noise models for each data set"
        noise_models = list()
        for sample_set in self.data_set:

            "Use intensity estimate, if provided; otherwise, calculate from samples."
            if 'E_est' not in sample_set.keys():
                sample_set['E_est'] = (np.nanmean(sample_set['samples'], 0) - sample_set['S0_map']) \
                                      * sample_set['ADU_map'] / sample_set['gain_est']

            "Initialize a TensorFlow model for data set. (0, 0) pixel values used only for initialization."
            noise_models.append(NoiseModel(sample_set['S0_map'][0, 0],
                                           sample_set['sigma_map'][0, 0],
                                           sample_set['gain_est'][0, 0],
                                           sample_set['ADU_map'][0, 0],
                                           sample_set['E_est'][0, 0],
                                           sample_set['samples'][:, 0, 0],
                                           diff_var=['p_e', 'gain'], session=self.session))

        sx, sy = self.data_set[0]['samples'].shape[1], self.data_set[0]['samples'].shape[2]

        """
        PyMC3 section.
        """

        with pm.Model() as m:

            "Initialize fit variables (E_i and gain)."
            pm_Es = []
            for idx, sample_set in enumerate(self.data_set):
                pm_Es.append(pm.Uniform('E_{:d}'.format(idx), self.params['E_MIN'], self.params['E_MAX'],
                                        testval=sample_set['E_est'][0, 0]))
            pm_gain = pm.Uniform('gain', self.params['GAIN_MIN'], self.params['GAIN_MAX'],
                                 testval=self.data_set[0]['gain_est'][0, 0])

            "Construct likelihood objects."
            thetas = []  # list holding the PyMC3 variables required for each individual noise model
            noise_models_wrapped = []  # list of wrapped noise models
            pm_logps = []  # list of PyMC3 logp functions
            for idx, noise_model in enumerate(noise_models):
                thetas.append(tt.as_tensor_variable([pm_Es[idx], pm_gain]))
                noise_models_wrapped.append(LogLikelihood(noise_model, [noise_model.p_e, noise_model.gain]))
                pm_logps.append(pm.Potential('likelihood_{:d}'.format(idx), noise_models_wrapped[idx](thetas[idx])))

            "Collate variable names (only those of interest for results)."
            var_names = [val.name for val in m.free_RVs if '__' not in val.name] + [val.name for val in m.deterministics
                                                                                    if '__' not in val.name]

            "Initialize results."
            results = dict()
            for var in var_names:
                results[var] = np.zeros((sx, sy)) * np.nan

            "Loop over pixels."
            if not px_list:  # if px_list not specified, loop over all pixels
                px_list = list(np.ndindex(sx, sy))
            for [idx_i, idx_j] in px_list:

                "Load values for current pixel."
                for sample_set, noise_model in zip(self.data_set, noise_models):
                    noise_model.samples.load(sample_set['samples'][:, idx_i, idx_j])
                    noise_model.S0.load(sample_set['S0_map'][idx_i, idx_j])
                    noise_model.sigma.load(sample_set['sigma_map'][idx_i, idx_j])
                    noise_model.ADU.load(sample_set['ADU_map'][idx_i, idx_j])
                    noise_model.p_e.load(sample_set['E_est'][idx_i, idx_j])
                    noise_model.gain.load(sample_set['gain_est'][idx_i, idx_j])
                    self.session.run(noise_model.ln_likelihood)

                "Optimize parameters."
                map_results = pm.find_MAP()

                "Report MAP results."
                print('({:d}, {:d})'.format(idx_i, idx_j))
                for key in sorted(var_names):
                    print('{}: {:.5f}'.format(key, map_results[key]))
                    results[key][idx_i, idx_j] = map_results[key]

                "save results"
                if not debug:
                    np.savez(output, **results)


class CalibrateGainGS(CalibrateBase):
    """
    Calibrate multiple gain set-points using a gain-series data set.
    """

    def run(self, mode, output='results.npz', px_list=None, debug=False):
        """
        Calibrate gain for each pixel in data set and save results.

        mode:       'MLE', 'hierarchical', 'non-centered'. Type of model to use for parameter estimation.
        px_list:    List of pixels to analyze. Overrides cycling over all pixels.
        debug:      Boolean. Toggles save results.
        """

        "Sanity check of 'mode' specification"
        if mode not in ['MLE', 'hierarchical', 'non-centered']:
            raise ValueError('Must specify mode as MLE, hierarchical, or non-centered.')

        "Estimate intensity from all data sets."
        E_est = [(np.nanmean(sample_set['samples'], 0) - sample_set['S0_map'])
                 * sample_set['ADU_map'] / sample_set['gain_est'] for sample_set in self.data_set]

        "Setup noise models for each data set"
        noise_models = list()
        for sample_set in self.data_set:
            "Initialize a TensorFlow model for data set. (0, 0) pixel values used only for initialization."
            diff_vars = ['p_e'] if sample_set['gain_est'][0, 0] == 1 else ['p_e', 'gain']  # if data set is 1x gain, gain is a fixed variable
            noise_models.append(NoiseModel(sample_set['S0_map'][0, 0],
                                           sample_set['sigma_map'][0, 0],
                                           sample_set['gain_est'][0, 0],
                                           sample_set['ADU_map'][0, 0],
                                           np.nanmean(E_est, 0)[0, 0],
                                           sample_set['samples'][:, 0, 0],
                                           diff_var=diff_vars, session=self.session))

        sx, sy = self.data_set[0]['samples'].shape[1], self.data_set[0]['samples'].shape[2]

        """
        PyMC3 section.
        """

        with pm.Model() as m:

            "Initialize fit variables (E_i and gains)."
            if mode == 'hierarchical':
                pm_mu = pm.Normal('mu', mu=np.nanmean(E_est, 0)[0, 0], sigma=self.params['MU_SIGMA'])
                pm_sigma = pm.HalfNormal('sigma', sigma=self.params['SIGMA_SIGMA'])
                pm_E = [pm.Normal('E_{:d}'.format(idx), mu=pm_mu, sigma=pm_sigma, testval=val[0, 0]) for idx, val in
                            enumerate(E_est)]

            elif mode == 'non-centered':
                pm_mu = pm.Normal('mu', mu=np.nanmean(E_est, 0)[0, 0], sigma=self.params['MU_SIGMA'])
                pm_sigma = pm.HalfNormal('sigma', sigma=self.params['SIGMA_SIGMA'])
                pm_mu_offset = [pm.Normal('mu_offset_{:d}'.format(idx), mu=0, sigma=1) for idx, val in
                                enumerate(E_est)]
                pm_E = [pm.Deterministic('E_{:d}'.format(idx), pm_mu + pm_mu_offset[idx] * pm_sigma) for idx, val in
                        enumerate(E_est)]

            else:  # default to MLE
                pm_E = pm.Uniform('E', self.params['E_MIN'], self.params['E_MAX'], testval=np.nanmean(E_est, 0)[0, 0])

            pm_gains = []
            for idx, sample_set in enumerate(self.data_set):
                if sample_set['gain_est'][0, 0] == 1:
                    pm_gains.append(None)  # placeholder for 1x gain noise models
                else:
                    pm_gains.append(pm.Uniform('gain_{:d}'.format(idx), self.params['GAIN_MIN'], self.params['GAIN_MAX'],
                                               testval=sample_set['gain_est'][0, 0]))

            "Construct likelihood objects."
            thetas = []  # list holding the PyMC3 variables required for each individual noise model
            noise_models_wrapped = []  # list of wrapped noise models
            pm_logps = []  # list of PyMC3 logp functions
            for idx, [noise_model, sample_set, gain] in enumerate(zip(noise_models, self.data_set, pm_gains)):

                "Construct thetas holding all free variables."
                if mode in ['hierarchical', 'non-centered']:
                    if sample_set['gain_est'][0, 0] == 1:
                        thetas.append(tt.as_tensor_variable([pm_E[idx]]))  # E is only free parameter for 1x gain data sets
                    else:
                        thetas.append(tt.as_tensor_variable([pm_E[idx], gain]))
                else:  # default to MLE
                    if sample_set['gain_est'][0, 0] == 1:
                        thetas.append(tt.as_tensor_variable([pm_E]))  # E is only free parameter for 1x gain data sets
                    else:
                        thetas.append(tt.as_tensor_variable([pm_E, gain]))

                "Construct wrapped noise models."
                if sample_set['gain_est'][0, 0] == 1:
                    noise_models_wrapped.append(LogLikelihood(noise_model, [noise_model.p_e]))
                else:
                    noise_models_wrapped.append(LogLikelihood(noise_model, [noise_model.p_e, noise_model.gain]))

                "Construct global likelihood variable."
                pm_logps.append(pm.Potential('likelihood_{:d}'.format(idx), noise_models_wrapped[idx](thetas[idx])))

            "Collate variable names (only those of interest for results)."
            var_names = [val.name for val in m.free_RVs if '__' not in val.name] + [val.name for val in m.deterministics
                                                                                    if '__' not in val.name]

            "Initialize results."
            results = dict()
            for var in var_names:
                results[var] = np.zeros((sx, sy)) * np.nan

            "Loop over pixels."
            if not px_list:  # if px_list not specified, loop over all pixels
                px_list = list(np.ndindex(sx, sy))
            for [idx_i, idx_j] in px_list:

                "Load values for current pixel."
                for sample_set, noise_model, e in zip(self.data_set, noise_models, E_est):
                    noise_model.samples.load(sample_set['samples'][:, idx_i, idx_j])
                    noise_model.S0.load(sample_set['S0_map'][idx_i, idx_j])
                    noise_model.sigma.load(sample_set['sigma_map'][idx_i, idx_j])
                    noise_model.ADU.load(sample_set['ADU_map'][idx_i, idx_j])
                    noise_model.p_e.load(e[idx_i, idx_j])
                    noise_model.gain.load(sample_set['gain_est'][idx_i, idx_j])
                    self.session.run(noise_model.ln_likelihood)

                "Optimize parameters."
                map_results = pm.find_MAP()

                "Report MAP results."
                print('({:d}, {:d})'.format(idx_i, idx_j))
                for key in sorted(var_names):
                    print('{}: {:.5f}'.format(key, map_results[key]))
                    results[key][idx_i, idx_j] = map_results[key]

                "save results"
                if not debug:
                    np.savez(output, **results)
