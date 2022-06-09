from __future__ import annotations

import numpy as np
import sklearn.linear_model
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, RidgeRegression, \
    LinearRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select
    the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2)
    # + eps for eps Gaussian noise and split into training- and testing
    # portions
    X, y, y_true = generate_data(n_samples, noise, -1.2, 2)

    X_train, y_train, X_test, y_test = split_train_test(
        pd.DataFrame(X), pd.Series(y), 2 / 3)

    plot_Q1_figure(X, X_test, X_train, y_test, y_train, y_true, noise)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_scores, validation_scores = list(), list()
    for k in range(11):
        base_estimator = PolynomialFitting(k)
        train_score, validation_score = cross_validate(
            base_estimator, np.array(X_train[0]), np.array(y_train),
            mean_square_error)
        train_scores.append(train_score)
        validation_scores.append(validation_score)

    plot_Q2_figure(train_scores, validation_scores)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and
    # report test error
    k = np.argmin(validation_scores)
    model = PolynomialFitting(k)
    model.fit(np.array(X_train[0]), np.array(y_train))


def plot_Q2_figure(train_scores, validation_scores):
    go.Figure(
        [go.Scatter(x=np.arange(11), y=train_scores,
                    name="training error", mode="lines+markers"),
         go.Scatter(x=np.arange(11), y=validation_scores,
                    name="validation error", mode="lines+markers")],
        layout=go.Layout(title="Training and Validation Error as a function "
                               "of the Polynomial Degree")
    ).show()


def plot_Q1_figure(X, X_test, X_train, y_test, y_train, y_true, noise):
    go.Figure(
        [
            go.Scatter(x=X, y=y_true, name="noiseless",
                       mode="markers", marker=dict(color="black")),
            go.Scatter(x=X_train[0], y=y_train, name="train set",
                       mode="markers", marker=dict(color="red")),
            go.Scatter(x=X_test[0], y=y_test, name="test set",
                       mode="markers", marker=dict(color="blue"))
        ], layout=go.Layout(title="Generated Data, without (black) and "
                                  "with (red for train and blue for test) "
                                  f"noise level of {noise}.")
    ).show()


def generate_data(n_samples, noise, start, end):
    X = np.linspace(start, end, n_samples)
    y_true = f(X)
    epsilon = np.random.normal(np.zeros(n_samples), noise)
    y = y_true + epsilon
    return X, y, y_true


def f(X):
    return (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best
    fitting regularization parameter values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the
        algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing
    # portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train = X[:n_samples], y[:n_samples]
    X_test, y_test = X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization
    # parameter for Ridge and Lasso regressions
    l_train_scores, l_validation_scores = list(), list()
    r_train_scores, r_validation_scores = list(), list()

    lasso_range = [2 * i / (n_evaluations) for i in range(n_evaluations)]
    ridge_range = [2 * i / (n_evaluations) for i in range(n_evaluations)]

    for l_lam, r_lam in zip(lasso_range, ridge_range):
        lasso_est = Lasso(l_lam)

        l_train_score, l_validation_score = cross_validate(
            lasso_est, X_train, y_train, mean_square_error)
        l_train_scores.append(l_train_score)
        l_validation_scores.append(l_validation_score)

        ridge_est = RidgeRegression(r_lam)
        r_train_score, r_validation_score = cross_validate(
            ridge_est, X_train, y_train, mean_square_error)
        r_train_scores.append(r_train_score)
        r_validation_scores.append(r_validation_score)

    go.Figure(
        [go.Scatter(x=lasso_range, y=l_train_scores,
                    name="training error", mode="lines+markers"),
         go.Scatter(x=lasso_range, y=l_validation_scores,
                    name="validation error", mode="lines+markers")],
        layout=go.Layout(title="Lasso Regression, training- and "
                               "validation-error as a function of lambda")
    ).show()

    go.Figure(
        [go.Scatter(x=ridge_range, y=r_train_scores,
                    name="training error", mode="lines+markers"),
         go.Scatter(x=ridge_range, y=r_validation_scores,
                    name="validation error", mode="lines+markers")],
        layout=go.Layout(title="Ridge Regression, training- and "
                               "validation-error as a function of lambda")
    ).show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least
    # Squares model
    l_lam = lasso_range[np.argmin(l_validation_scores)]
    r_lam = ridge_range[np.argmin(r_validation_scores)]

    lasso_model = Lasso(l_lam)
    lasso_model.fit(X_train, y_train)
    ridge_model = RidgeRegression(r_lam)
    ridge_model.fit(X_train, y_train)
    ls_model = LinearRegression()
    ls_model.fit(X_train, y_train)

    lasso_test_error = mean_square_error(y_test, lasso_model.predict(X_test))
    ridge_test_error = ridge_model.loss(X_test, y_test)
    ls_test_error = ls_model.loss(X_test, y_test)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()

