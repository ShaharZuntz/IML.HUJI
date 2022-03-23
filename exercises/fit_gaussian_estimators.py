import numpy as np
import scipy.stats

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
from utils import *

pio.templates.default = "simple_white"

PLOT_HEIGHT = 500
TITLE_FONT = {"family": "Times New Roman", "size": 20}

TITLE_FIGURE_Q1 = "Absolute Distance Between Estimated and True Value of " \
                  "the Expectation as a Function of the Sample Size"
TITLE_FIGURE_Q3 = "The Empirical PDF Function Under the Fitted Model"
TITLE_FIGURE_Q5 = "Log-likelihood value as a Function of f1, f3"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    real_mu = 10
    real_var = 1
    sample_size = 1000

    samples = np.random.normal(real_mu, real_var, size=sample_size)

    univariate_gaussian = UnivariateGaussian().fit(samples)
    print((univariate_gaussian.mu_, univariate_gaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    min_num_of_samples = 10
    num_of_samples_axis = np.linspace(
        min_num_of_samples, sample_size, 100
    ).astype(np.int64)

    estimated_mean = []
    for num_of_samples in num_of_samples_axis:
        estimated_mean.append(
            abs(real_mu -
                UnivariateGaussian().fit(samples[:num_of_samples]).mu_)
        )

    go.Figure(
        [go.Scatter(x=num_of_samples_axis, y=estimated_mean,
                    mode='markers+lines')],
        layout=go.Layout(
            title={"text": TITLE_FIGURE_Q1, "font": TITLE_FONT},
            xaxis={"title": "Sample Size"},
            yaxis={"title": "r$|\mu-\hat\mu|$"},
            height=PLOT_HEIGHT
        )).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    PDFs_vector = univariate_gaussian.pdf(samples)

    go.Figure(
        [go.Scatter(x=samples, y=PDFs_vector, mode='markers')],
        layout=go.Layout(
            title={"text": TITLE_FIGURE_Q3, "font": TITLE_FONT},
            xaxis={"title": "Sample Values"},
            yaxis={"title": "PDF Values"},
            height=PLOT_HEIGHT
        )).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    real_mu = np.asarray([0, 0, 4, 0])
    real_cov = np.asarray(
        [[1, 0.2, 0, 0.5],
         [0.2, 2, 0, 0],
         [0, 0, 1, 0],
         [0.5, 0, 0, 1]]
    )
    num_of_samples = 1000

    samples = np.random.multivariate_normal(
        real_mu, real_cov, size=num_of_samples
    )

    multivariate_gaussian = MultivariateGaussian().fit(samples)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    axis_size = 200
    max_axis_val = 10
    min_axis_val = -10

    f1_axis = np.linspace(min_axis_val, max_axis_val, axis_size)
    f3_axis = np.linspace(min_axis_val, max_axis_val, axis_size)

    log_likelihood_values = list()
    for f1 in f1_axis:
        for f3 in f3_axis:
            log_likelihood_values.append(MultivariateGaussian.log_likelihood(
                np.asarray([f1, 0, f3, 0]), real_cov, samples
            ))

    log_likelihood_axis = np.asarray(log_likelihood_values).reshape(
        axis_size, axis_size
    )

    go.Figure(
        [go.Heatmap(x=f3_axis, y=f1_axis, z=log_likelihood_axis)],
        layout=go.Layout(
            title={"text": TITLE_FIGURE_Q5, "font": TITLE_FONT},
            xaxis={"title": "f3"},
            yaxis={"title": "f1"},
            height=PLOT_HEIGHT, width=PLOT_HEIGHT
        )).show()

    # Question 6 - Maximum likelihood
    max_val = np.amax(log_likelihood_axis)
    argmax_indices = np.where(log_likelihood_axis == max_val)
    argmax_vals = (
        convert_index_to_value(argmax_indices[0][0], min_axis_val,
                               max_axis_val, axis_size),
        convert_index_to_value(argmax_indices[1][0], min_axis_val,
                               max_axis_val, axis_size)
    )

    print(f"max value: {round(max_val, 3)}"
          f"\nf1,f3: {tuple(round(num, 3) for num in argmax_vals)}")


def convert_index_to_value(index: int, min_axis_val, max_axis_val, axis_size
                           ) -> float:
    """
    Converts an index in the axis array into its real value on the heatmap.
    That is, 0 is converted to min_axis_val, and (axis_size-1) is converted to
    max_axis_val.
    """
    return ((max_axis_val - min_axis_val) * index / (axis_size - 1.0)
            + min_axis_val)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
