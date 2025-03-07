from math import ceil
from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(
        X: pd.DataFrame, y: pd.Series, train_proportion: float = .75
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    n_samples = y.shape[0]
    train_size = ceil(train_proportion * n_samples)
    train_indices = np.random.choice(n_samples, train_size, replace=False)

    train_x = X.iloc[train_indices]
    train_y = y.iloc[train_indices]

    test_x = X.drop(train_indices)
    test_y = y.drop(train_indices)



    # y_name = y.name
    # Xy = pd.concat([X, y], axis=1)
    #
    # n_samples = Xy.shape[0]
    # n_samples_train = ceil(train_proportion * n_samples)
    #
    # train_xy = Xy.sample(n_samples_train)
    # test_xy = Xy.drop(train_xy.index)
    #
    # train_x = pd.DataFrame(train_xy.drop([y_name], axis=1))
    # test_x = pd.DataFrame(test_xy.drop([y_name], axis=1))
    #
    # train_y = train_xy[y_name]
    # test_y = test_xy[y_name]

    return train_x, train_y, test_x, test_y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
