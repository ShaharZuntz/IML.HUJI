import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250,
                              train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = (generate_data(train_size, noise),
                                            generate_data(test_size, noise))

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ABLearner = Q1(n_learners, train_X, train_y, test_X, test_y)

    # Question 2: Plotting decision surfaces
    Q2(ABLearner, train_X, test_X, test_y)

    # Question 3: Decision surface of best performing ensemble
    Q3(ABLearner, n_learners, train_X, test_X, test_y)

    # Question 4: Decision surface with weighted samples
    Q4(ABLearner, train_X, train_y, test_X)


def Q1(n_learners: int,
       train_X: np.ndarray, train_y: np.ndarray,
       test_X: np.ndarray, test_y: np.ndarray) -> AdaBoost:
    # Fit the AdaBoost learner
    ABLearner = AdaBoost(wl=lambda: DecisionStump(), iterations=n_learners)
    ABLearner.fit(train_X, train_y)

    # calculate partial training and test error
    training_error = [ABLearner.partial_loss(train_X, train_y, T)
                      for T in range(1, n_learners + 1)]
    test_error = [ABLearner.partial_loss(test_X, test_y, T)
                  for T in range(1, n_learners + 1)]

    # plot graphs
    go.Figure(
        [go.Scatter(x=[*range(1, n_learners + 1)], y=training_error,
                    mode='markers+lines', name="Train Error"),
         go.Scatter(x=[*range(1, n_learners + 1)], y=test_error,
                    mode='markers+lines', name="Test Error")
         ],
        layout=go.Layout(
            title="Training- and Test Errors as a Function of the "
                  "Number of Fitted Learners",
            xaxis={"title": "Number of Fitted Learners"},
            yaxis={"title": "Error"}
        )
    ).show()

    return ABLearner


def Q2(ABLearner: AdaBoost, train_X: np.ndarray,
       test_X: np.ndarray, test_y: np.ndarray) -> None:
    T = [5, 50, 100, 250]
    lims = np.array(
        [np.r_[train_X, test_X].min(axis=0),
         np.r_[train_X, test_X].max(axis=0)]
    ).T + np.array([-.1, .1])

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Number of Iterations: {t} " for t in T],
        horizontal_spacing=0.01, vertical_spacing=.03
    )
    fig.layout.title = "Decision Boundaries as a Function of the Size of " \
                       "the Ensemble"

    for i, t in enumerate(T):
        fig.add_traces(
            [
                decision_surface(
                    lambda X: ABLearner.partial_predict(X, t),
                    lims[0], lims[1], showscale=False
                ),
                go.Scatter(
                    x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                    showlegend=False, marker=dict(
                        color=test_y,
                        colorscale=[custom[0], custom[-1]],
                        line=dict(color="black", width=1)
                    )
                )
            ],
            rows=(i // 2) + 1, cols=(i % 2) + 1
        )
    fig.show()


def Q3(ABLearner: AdaBoost, n_learners: int, train_X: np.ndarray,
       test_X: np.ndarray, test_y: np.ndarray) -> None:
    test_error = [ABLearner.partial_loss(test_X, test_y, T)
                  for T in range(1, n_learners + 1)]
    lowest_test_error_ens_size = np.argmin(test_error) + 1
    lowest_test_error_ens_acc = test_error[lowest_test_error_ens_size - 1]

    lims = np.array(
        [np.r_[train_X, test_X].min(axis=0),
         np.r_[train_X, test_X].max(axis=0)]
    ).T + np.array([-.1, .1])

    go.Figure(
        [
            decision_surface(
                lambda X:
                ABLearner.partial_predict(X, lowest_test_error_ens_size),
                lims[0], lims[1], showscale=False
            ),
            go.Scatter(
                x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                showlegend=False, marker=dict(
                    color=test_y,
                    colorscale=[custom[0], custom[-1]],
                    line=dict(color="black", width=1)
                )
            )
        ],
        layout=go.Layout(
            title=f"Decision Surface of the Ensemble Which Achieved the Lowest"
                  f" Test Error. Ensemble Size: {lowest_test_error_ens_size}. "
                  f"Ensemble Accuracy: {lowest_test_error_ens_acc}."
        )
    ).show()


def Q4(ABLearner: AdaBoost, train_X: np.ndarray, train_y: np.ndarray,
       test_X: np.ndarray) -> None:

    lims = np.array(
        [np.r_[train_X, test_X].min(axis=0),
         np.r_[train_X, test_X].max(axis=0)]
    ).T + np.array([-.1, .1])

    go.Figure(
        [
            decision_surface(
                lambda X: ABLearner.partial_predict(X, 250),
                lims[0], lims[1], showscale=False
            ),
            go.Scatter(
                x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                showlegend=False, marker=dict(
                    color=train_y,
                    colorscale=[custom[0], custom[-1]],
                    line=dict(color="black", width=1),
                    size=ABLearner.D_ / np.max(ABLearner.D_) * 15
                )
            )
        ],
        layout=go.Layout(
            title="Training Set With a Point Size Proportional to it’s Weight."
        )
    ).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
