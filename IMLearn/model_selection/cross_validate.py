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
    perm = np.random.permutation(len(X))
    subset_size = len(X) // cv

    # TODO: split might return an over-sized partition
    perm_split = np.split(perm, np.arange(subset_size, len(X), subset_size))

    train_scores = 0
    validation_scores = 0

    for i, validation_idx in enumerate(perm_split):
        X_train = X[~np.isin(validation_idx)]
        y_train = y[~np.isin(validation_idx)]
        estimator.fit(X_train, y_train)
        train_scores += scoring(y_train,
                                estimator.predict(X_train))

        X_validation = X[validation_idx]
        y_validation = y[validation_idx]
        validation_scores += scoring(y_validation,
                                     estimator.predict(X_validation))

    return train_scores / cv, validation_scores / cv
