import tensorflow as tf
import numpy as np


def logli_normal_bivariate(x, mu, sigma, rho, reduce_sum=False):
    """
    Bivariate Gaussian log-likelihood. Rank of arguments is expected to be 3.

    Args:
        x: data samples with shape (batch_size, num_time_steps, data_size).
        mu:
        sigma: standard deviation.
        rho:
        reduce_sum: False, None or list of axes.
    Returns:

    """
    last_axis = tf.rank(x)-1
    x1, x2 = tf.split(x, 2, axis=last_axis)
    mu1, mu2 = tf.split(mu, 2, axis=last_axis)
    sigma1, sigma2 = tf.split(sigma, 2, axis=last_axis)

    with tf.name_scope('logli_normal_bivariate'):
        x_mu1 = tf.subtract(x1, mu1)
        x_mu2 = tf.subtract(x2, mu2)
        Z = tf.square(tf.div(x_mu1, tf.maximum(1e-9, sigma1))) + \
            tf.square(tf.div(x_mu2, tf.maximum(1e-9, sigma2))) - \
            2*tf.div(tf.multiply(rho, tf.multiply(x_mu1, x_mu2)), tf.maximum(1e-9, tf.multiply(sigma1, sigma2)))

        rho_square_term = tf.maximum(1e-9, 1-tf.square(rho))
        log_regularize_term = tf.log(tf.maximum(1e-9, 2*np.pi*tf.multiply(tf.multiply(sigma1, sigma2), tf.sqrt(rho_square_term)) ))
        log_power_e = tf.div(Z, 2*rho_square_term)
        result = -(log_regularize_term + log_power_e)

        if reduce_sum is False:
            return result
        else:
            return tf.reduce_sum(result, reduce_sum)


def logli_normal_diag_cov(x, mu, sigma, reduce_sum=False):
    """
    Log-likelihood of Gaussian with diagonal covariance matrix.

    Args:
        x:
        mu:
        sigma: standard deviation.
        reduce_sum:

    Returns:

    """
    with tf.name_scope('logli_normal_diag_cov'):
        ssigma2 = tf.maximum(1e-6, tf.square(sigma)*2)
        denom_log = tf.log(tf.sqrt(np.pi * ssigma2))
        norm = tf.square(tf.subtract(x, mu))
        z = tf.div(norm, ssigma2)
        result = -(z + denom_log)

        if reduce_sum is False:
            return result
        else:
            return tf.reduce_sum(result, reduce_sum)


def logli_bernoulli(x, theta, reduce_sum=False):
    """
    Bernoulli log-likelihood.

    Args:
        x:
        theta:
        reduce_sum:

    Returns:

    """
    with tf.name_scope('logli_bernoulli'):
        result = (tf.multiply(x, tf.log(tf.maximum(1e-9, theta))) + tf.multiply((1 - x), tf.log(tf.maximum(1e-9, 1 - theta))))

        if reduce_sum is False:
            return result
        else:
            return tf.reduce_sum(result, reduce_sum)


def kld_normal_isotropic(mu1, sigma1, mu2, sigma2, reduce_sum=False):
    """
    Kullback-Leibler divergence between two isotropic Gaussian distributions.

    Args:
        mu1:
        sigma1: standard deviation.
        mu2:
        sigma2: standard deviation.
        reduce_sum:

    Returns:

    """
    with tf.name_scope("kld_normal_isotropic"):
        result = tf.reduce_sum(0.5 * (
            2 * tf.log(tf.maximum(1e-9, sigma2))
            - 2 * tf.log(tf.maximum(1e-9, sigma1))
            + (tf.square(sigma1) + tf.square(mu1 - mu2)) / tf.maximum(1e-9, (tf.square(sigma2))) - 1), keepdims=True, axis=-1)

        if reduce_sum is False:
            return result
        else:
            return tf.reduce_sum(result, reduce_sum)