from __future__ import annotations

import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased
            estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not
            been fitted. To be set as True in `UnivariateGaussian.fit`
            function.

        mu_: float
            Estimated expectation initialized as None. To be set in
            `UnivariateGaussian.fit` function.

        var_: float
            Estimated variance initialized as None. To be set in
            `UnivariateGaussian.fit` function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated
        estimation (where estimator is either biased or unbiased). Then sets
        `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X)
        self.var_ = X.var(ddof=0 if self.biased_ else 1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted
        estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")

        pdfs = []

        denominator = (2 * np.pi * self.var_) ** 0.5

        for x in X:
            exp_arg = - ((x - self.mu_) ** 2) / (2 * self.var_)
            numerator = np.exp(exp_arg)

            x_pdf = numerator / denominator
            pdfs.append(x_pdf)

        return np.asarray(pdfs)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian
        model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        n_samples = X.shape[0]

        left_operand = n_samples * np.log((1 / (2 * np.pi * sigma) ** 0.5))

        right_operand = 0
        for x in X:
            right_operand += - ((x - mu) ** 2) / (2 * sigma)

        return left_operand + right_operand


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X.T, ddof=1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")

        pdfs = []
        mn = multivariate_normal(self.mu_, self.cov_)
        for x in X:
            pdfs.append(mn.pdf(x))

        return np.ndarray(pdfs)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray,
                       X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)

        s1 = 0
        for sample in X:
            diff = (sample - mu)
            s1 += -0.5 * diff.T.dot(cov_inv).dot(diff)

        m = X.shape[0]
        d = X.shape[1]

        log_param = 1 / (2 * np.pi * (cov_det ** (1 / d)))
        s2 = d * m * np.log(log_param) / 2

        return s1 + s2
