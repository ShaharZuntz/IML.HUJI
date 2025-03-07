import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import sklearn.metrics
from sklearn.metrics import roc_curve

import utils
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2, BaseModule, LogisticModule
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None],
                                              List[np.ndarray],
                                              List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's
    value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding
        the objective's value and parameters at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values: List[np.ndarray] = list()
    weights: List[np.ndarray] = list()

    def callback(val: np.ndarray, weight: np.ndarray, **kwargs):
        values.append(val)
        weights.append(weight)

    return callback, values, weights


def minimize_module(init: np.ndarray, eta: float, M: Type[BaseModule],
                    module_name: str):
    callback, values, weights = get_gd_state_recorder_callback()

    gd = GradientDescent(FixedLR(eta), callback=callback)
    gd.fit(M(init), None, None)

    # plot descent path
    plot_descent_path(
        module=M,
        descent_path=np.array(weights),
        title=f"Descent Trajectory; Module: {module_name}, eta={eta}"
    ).show()

    num_iter = len(values)

    # plot convergence rate
    go.Figure(
        [go.Scatter(x=np.arange(num_iter),
                    y=np.array(values).reshape(num_iter),
                    mode="markers+lines", marker_color="black")],
        layout=go.Layout(
            title=f"Convergence Rate, Module: {module_name}, eta={eta}")
    ).show()


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)
):
    for eta in etas:
        # minimize L1 module
        minimize_module(init, eta, L1, "L1")

        # minimize L2 module
        minimize_module(init, eta, L2, "L2")


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the
    # exponentially decaying learning rate
    fig = go.Figure(
        layout=go.Layout(title=f"Convergence Rate of Exponential learning "
                               f"rate with different Gammas, Module: L1, "
                               f"eta={eta}")
    )
    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(ExponentialLR(eta, gamma), callback=callback)
        gd.fit(L1(init), None, None)
        num_iter = len(values)
        fig.add_trace(
            go.Scatter(
                x=np.arange(num_iter), y=np.array(values).reshape(num_iter),
                mode="lines", name=f"{gamma}"
            ))

    # Plot algorithm's convergence for the different values of gamma
    fig.show()

    # Plot descent path for gamma=0.95
    callback, values, weights = get_gd_state_recorder_callback()
    gd = GradientDescent(ExponentialLR(eta, .95), callback=callback)
    gd.fit(L1(init), None, None)
    plot_descent_path(
        module=L1,
        descent_path=np.array(weights),
        title=f"Descent Trajectory of Exponential learning rate; Module: L1, "
              f"eta={eta}, gamma=0.95"
    ).show()


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8
              ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
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
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X = np.array(X_train)
    train_mean = np.mean(X, axis=0)
    X -= train_mean

    y = np.array(y_train)

    GD = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
    model = LogisticRegression(solver=GD).fit(X, y)

    # model = LogisticRegression(solver=GD).fit(X, y)
    y_prob = model.predict_proba(X)

    fpr, tpr, thresholds = roc_curve(y, y_prob)
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title="ROC curve of Logistic Regression (non-Regularized) over "
                  "the South Africa Heart Disease Dataset",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    opt_alpha = thresholds[np.argmax(tpr - fpr)]
    model.alpha_ = opt_alpha
    test = np.array(X_test) - train_mean
    # print(f"alpha={opt_alpha}, test error={model.loss(test, np.array(y_test))}")


    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    train_scores, validation_scores = list(), list()

    for lam in lambdas:
        GD = GradientDescent(learning_rate=FixedLR(base_lr=1e-4), max_iter=20000)
        model = LogisticRegression(solver=GD, penalty="l1", alpha=0.5,
                                   lam=lam)
        train_score, validation_score = cross_validate(
            model, X, np.array(y_train), misclassification_error
        )
        train_scores.append(train_score)
        validation_scores.append(validation_score)

    opt_lam = lambdas[np.argmin(validation_scores)]
    GD = GradientDescent(learning_rate=FixedLR(base_lr=1e-4), max_iter=20000)
    model = LogisticRegression(solver=GD, penalty="l1", alpha=0.5,
                               lam=opt_lam).fit(X, np.array(y_train))
    # print(f"l1: lam={opt_lam}, test error"
    #       f"={model.loss(test, np.array(y_test))}")

    train_scores, validation_scores = list(), list()

    for lam in lambdas:
        GD = GradientDescent(learning_rate=FixedLR(base_lr=1e-4), max_iter=20000)
        model = LogisticRegression(solver=GD, penalty="l2", alpha=0.5,
                                   lam=lam)
        train_score, validation_score = cross_validate(
            model, X, np.array(y_train), misclassification_error
        )
        train_scores.append(train_score)
        validation_scores.append(validation_score)

    opt_lam = lambdas[np.argmin(validation_scores)]
    GD = GradientDescent(learning_rate=FixedLR(base_lr=1e-4), max_iter=20000)
    model = LogisticRegression(solver=GD, penalty="l2", alpha=0.5,
                               lam=opt_lam).fit(X, np.array(y_train))
    # print(f"l2: lam={opt_lam}, test error"
    #       f"={model.loss(test, np.array(y_test))}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
