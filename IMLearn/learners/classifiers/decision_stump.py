from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART
    algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is
        about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples = X.shape[0]
        n_features = X.shape[1] if len(X.shape) > 1 else 1

        thr = X[0, 0]
        thr_err = 1
        opt_j = 0
        sign = 1

        for j, col in enumerate(X.transpose()):
            thr_p, thr_err_p = self._find_threshold(col, y, 1)
            thr_m, thr_err_m = self._find_threshold(col, y, -1)

            if thr_err_p <= thr_err_m:
                thr_j, thr_err_j, sign_j = thr_p, thr_err_p, 1
            else:
                thr_j, thr_err_j, sign_j = thr_m, thr_err_m, -1

            if thr_err_j <= thr_err:
                thr, opt_j, sign = thr_j, j, sign_j

        self.threshold_, self.j_, self.sign_ = thr, opt_j, sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign`
        whereas values which equal to or above the threshold are predicted
        as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_,
                        self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform
        a split. The threshold is found according to the value minimizing
        the mis-classification error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Mis-classification error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are
        predicted as `-sign` whereas values which equal to or above the
        threshold are predicted as `sign`
        """
        matrix = np.c_[values, labels]

        thr = values[0]
        thr_err = 1

        for t in values:
            values_p = matrix[matrix[:, 0] >= t]
            values_m = matrix[matrix[:, 0] < t]

            loss_p = misclassification_error(
                values_p[:, 1], np.full(values_p.shape[0], sign)
            ) if values_p.shape[0] > 0 else 1
            loss_m = misclassification_error(
                values_m[:, 1], np.full(values_m.shape[0], -sign)
            ) if values_m.shape[0] > 0 else 1

            loss = loss_p + loss_m

            if loss < thr_err:
                thr, thr_err = t, thr_err

        return thr, thr_err

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
        return misclassification_error(y, self._predict(X))
