from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated
        model. When called, the scoring function receives the true- and
        predicted values for each sample and potentially additional
        arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    perm = np.arange(len(X))
    subset_size = int(np.ceil(len(X) / cv))
    perm_split = np.split(perm, np.arange(subset_size, len(X), subset_size))

    train_scores = 0
    validation_scores = 0

    for validation_idx in perm_split:
        X_train = np.delete(X, validation_idx, axis=0)
        y_train = np.delete(y, validation_idx)
        estimator.fit(X_train, y_train)
        y_train_pred = estimator.predict(X_train)
        train_scores += scoring(y_train_pred, y_train)

        X_validation = X[validation_idx]
        y_validation = y[validation_idx]
        y_validation_pred = estimator.predict(X_validation)
        validation_scores += scoring(y_validation_pred, y_validation)

    return train_scores / cv, validation_scores / cv
