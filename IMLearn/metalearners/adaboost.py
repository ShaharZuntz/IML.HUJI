import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn

from ..metrics import loss_functions


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.models_ = list()
        self.D_ = list()
        self.weights_ = list()

        num_samples = X.shape[0]

        S = np.c_[X, y]

        # set initial distribution to i=uniform
        D_t = np.full(1 / num_samples)

        for t in range(self.iterations_):
            self.D_.append(D_t)

            # invoke base learner on sample drawn according to D_t
            S_t = np.random.choice(S, num_samples, p=D_t)
            h_t = self.wl_().fit(S_t[:, :-1], S_t[:, -1])
            y_t = h_t.predict(S_t[:, :-1])

            # update and normalize sample weights
            e_t = np.sum(y_t != S_t[:, -1])
            w_t = 0.5 * np.log(1 / e_t - 1)
            D_t = D_t * np.exp(-y * w_t * y_t)
            D_t /= np.sum(D_t)

            self.models_.append(h_t)
            self.weights_.append(w_t)

    def _predict(self, X):
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
        return self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under mis-classification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under mis-classification loss function
        """
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y_pred = np.full(X.shape[0], 0)
        for i in range(T):
            y_pred += np.multiply(self.models_[i].predict(X).T,
                                  self.weights_[i])

        return np.sign(y_pred)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under mis-classification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under mis-classification loss function
        """
        return loss_functions.misclassification_error(
            y, self.partial_predict(X, T)
        )
