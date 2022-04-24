from typing import NoReturn

import scipy.stats

from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in
            `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        self.pi_ = counts / y.shape[0]

        self.mu_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        self.vars_ = np.zeros((self.classes_.shape[0], X.shape[1]))

        for i, sample in enumerate(X):
            self.mu_[y[i]] += sample

        self.mu_ /= counts[:, None]

        for i, sample in enumerate(X):
            diff = sample - self.mu_[y[i]]
            self.vars_[y[i]] += diff ** 2

        self.vars_ /= counts[:, None]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        z = np.ndarray((X.shape[0], self.classes_.shape[0]))

        for i, sample in enumerate(X):
            for k in self.classes_:
                p_vec = np.array(
                    [np.log(scipy.stats.norm.pdf(
                        sample[j], mean=self.mu_[k, j], var=self.vars_[k, j]
                    )
                            for j in range(X.shape[1]))]
                )
                z[i, k] = np.log(self.pi_[k]) + np.sum(p_vec)

        return np.argmax(z, axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.ndarray((X.shape[0], self.classes_.shape[0]))
        for k in range(self.classes_.shape[0]):
            n = scipy.stats.multivariate_normal(mean=self.mu_[k],
                                                cov=self.cov_)
            for i in range(X.shape[0]):
                likelihoods[i, k] = n.pdf(X[i]) * self.pi_[k]

        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))

