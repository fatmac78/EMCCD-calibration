import tensorflow as tf
import numpy as np
import os
from scipy.stats import rv_discrete
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress messages
os.environ['MKL_NUM_THREADS'] = '36'
os.environ['OMP_NUM_THREADS'] = '36'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # suppress messages


class NoiseModel:

    def __init__(self, S0, sigma, gain, ADU, p_e, samples, s_bit=17, thresh=700, p_lim=1e-16, diff_var=['p_e', 'ADU'],
                 session=None, norm=False):
        """
        Initialize NoiseModel class.
        :param S0:
        :param sigma:
        :param gain:
        :param ADU:
        :param p_e:
        :param samples:
        :param s_bit:
        :param thresh:
        :param p_lim:
        :param session:
        :param norm:
        """

        "if no TensorFlow session is specified, create one"
        if session is None:
            self.session = tf.compat.v1.InteractiveSession()
        else:
            self.session = session

        "store fixed variables"
        self.S0 = tf.Variable(S0, dtype=tf.float64)
        self.sigma = tf.Variable(sigma, dtype=tf.float64)
        self.samples = tf.Variable(samples, dtype=tf.int32)

        "define free variables"
        self.gain = tf.Variable(gain, name='gain', dtype=tf.float64)
        self.p_e = tf.Variable(p_e, name='p_e', dtype=tf.float64)
        self.ADU = tf.Variable(ADU, name='ADU', dtype=tf.float64)
        tf_0 = tf.constant(0.0, dtype=tf.float64, name='tf_0')  # placeholder constant

        "misc. settings"
        self.s_bit = s_bit
        self.thresh = thresh
        self.p_lim = p_lim

        "setup readout noise PDF"
        self.s = tf.range(0, 2 ** self.s_bit, name='s', dtype=tf.float64)
        self.g = tf.math.exp(-(self.s - self.S0) ** 2 / (2 * self.sigma ** 2)) / tf.math.sqrt(2 * np.pi * self.sigma ** 2)

        "construct probability distribution function q(S | E, gain, ADU)"
        if gain == 1:  # Poisson only

            ln_q = (self.s * self.ADU) * tf.math.log(self.p_e) - self.p_e - tf.math.lgamma(self.s * self.ADU + 1)
            self.q = tf.exp(ln_q) * self.ADU  # perform change fo variables normalization here

        else:  # non-central chi-squared distribution

            q1 = tf.math.sqrt(self.p_e / (self.s[1:] * self.ADU * self.gain)) * \
                     tf.math.exp(-self.s[1:] * self.ADU / self.gain - self.p_e + 2 * tf.math.sqrt(self.s[1:] * self.ADU * self.p_e / self.gain)) * \
                     tf.math.bessel_i1e(2 * tf.math.sqrt(self.s[1:] * self.ADU * self.p_e / self.gain))
            self.q = self.ADU * tf.concat([tf.convert_to_tensor(value=[tf.exp(-self.p_e) * (1 + self.p_e / self.gain)]), q1], 0)

        "convole Poisson/gamma with Gaussian using FFTs"
        conv = tf.math.real(tf.signal.ifft(tf.signal.fft(tf.cast(self.q, tf.complex128)) * tf.signal.fft(tf.cast(self.g, tf.complex128))))

        "complete forms of pdf and log(pdf)"
        pdf_clip = tf.where(tf.less(conv, self.p_lim), tf.ones_like(conv) * self.p_lim, conv)
        if norm:  # enforce normalization
            self.pdf = pdf_clip / tf.reduce_sum(pdf_clip)
        else:
            self.pdf = pdf_clip

        self.ln_pdf = tf.math.log(self.pdf)

        "likelihood function and derivative of the likelihood function"
        self.ln_likelihood = tf.reduce_sum(input_tensor=tf.gather(self.ln_pdf, self.samples))

        "collect variables requested for differentiation"
        xs = []
        if 'p_e' in diff_var:
            xs.append(self.p_e)
        if 'ADU' in diff_var:
            xs.append(self.ADU)
        if 'gain' in diff_var:
            xs.append(self.gain)

        if len(xs) > 0:
            self.d_ln_likelihood = tf.gradients(ys=self.ln_likelihood, xs=xs)

        "finalize TensorFlow session with variable initialization"
        self.session.run(tf.compat.v1.global_variables_initializer())

    def rv_pdf(self, n_samples):
        """
        Generate random samples from pdf
        """

        # print('initializing new random variable sampler')
        pdf_sampler = rv_discrete(values=(self.s.eval(), self.pdf.eval()))

        # print('generating random samples')
        samples = pdf_sampler.rvs(size=n_samples)

        return samples
