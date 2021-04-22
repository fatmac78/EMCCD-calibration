import theano.tensor as tt
import numpy as np
import os
os.environ['MKL_NUM_THREADS'] = '36'
os.environ['OMP_NUM_THREADS'] = '36'


class LogLikelihood(tt.Op):
    """
    Wrapper class for making the TensorFlow implementation of the camera noise model available to PyMC3 (Theano)
    """

    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, noise_model, free_vars):
        """
        Initialize wrapper class.
        :param noise_model:
        :param free_vars:
        """

        "store inputs"
        self.noise_model = noise_model
        self.free_vars = free_vars
        self.session = self.noise_model.session

        "gradient"
        self.logpgrad = LogLikelihoodGrad(self.noise_model, self.free_vars)

    def perform(self, node, inputs, output_storage, params=None):
        """
        Return the value of the likelihood function.
        :param node:
        :param inputs:
        :param output_storage:
        :param params:
        :return:
        """

        "set free variables of the TensorFlow noise model"
        theta, = inputs
        for var, val in zip(self.free_vars, theta):
            var.load(val)

        "calculate likelihood value"
        ll_val = self.session.run(self.noise_model.ln_likelihood)

        "store the likelihood value to the output register"
        output_storage[0][0] = np.array(ll_val)

    def grad(self, inputs, g):
        """
        Return the Jacobian of the likelihood function
        :param inputs:
        :param g:
        :return:
        """

        theta, = inputs
        # return [g[0] * self.logpgrad(theta)]  # form according to Pitkin (PyMC3 "black box likelihood" example)
        return [self.logpgrad(theta)]  # correct?


class LogLikelihoodGrad(tt.Op):
    """
    Wrapper class of the gradient for making the TensorFlow implementation of the camera noise model available to PyMC3 (Theano)
    """

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, noise_model, free_vars):
        """
        Initialize gradient wrapper class.
        :param noise_model:
        :param free_vars:
        """

        "store inputs"
        self.noise_model = noise_model
        self.free_vars = free_vars
        self.session = self.noise_model.session

    def perform(self, node, inputs, output_storage, params=None):
        """
        Return the Jacobian of the likelihood function
        :param node:
        :param inputs:
        :param output_storage:
        :param params:
        :return:
        """

        "set free variables of the TensorFlow noise model"
        theta, = inputs
        for var, val in zip(self.free_vars, theta):
            var.load(val)

        "calculate the gradient of the likelihood function"
        if len(self.free_vars) == 1:
            d_ll_val = [self.session.run(self.noise_model.d_ln_likelihood[0])]
        else:
            d_ll_val = self.session.run(self.noise_model.d_ln_likelihood)

        "store the gradient of the likelihood function to the output register"
        output_storage[0][0] = np.array(d_ll_val)
