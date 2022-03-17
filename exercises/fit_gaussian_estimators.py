import numpy as np
import numpy.random

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian

from utils import *

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    real_mu = 10
    sample_size = 1000
    samples = np.random.normal(real_mu, 1, size=sample_size)

    univariate_gaussian = UnivariateGaussian().fit(samples)
    print(f"({univariate_gaussian.mu_}, {univariate_gaussian.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, sample_size, 100).astype(np.int64)
    estimated_mean = []
    for m in ms:
        estimated_mean.append(
            abs(real_mu -
                UnivariateGaussian.fit(UnivariateGaussian(), samples[:m]).mu_)
        )

    go.Figure(
        [go.Scatter(x=ms, y=estimated_mean, mode='markers+lines')],
        layout=go.Layout(
            title=r"$\text{Distance between Estimation and True Value of "
                  r"Expectation As Function Of Number Of Samples}$",
            xaxis={"title": "r$\\text{number of samples}$"},
            yaxis={"title": "r$|\mu-\hat\mu|$"},
            height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    fig = make_subplots().add_trace(
        go.Scatter(x=samples, y=univariate_gaussian.pdf(samples),
                   mode='markers', opacity=0.75)) \
        .update_layout(
        go.Layout(
            title=r"$\text{The Empirical PDF Function Under the Fitted "
                  r"Model}$",
            xaxis={"title": "r$\\text{Sample Values}$"},
            yaxis={"title": "r$\\text{PDF Values}$"},
            height=500))

    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    real_mu = np.asarray([0, 0, 4, 0])
    real_cov = np.asarray([[1, 0.2, 0, 0.5],
                           [0.2, 2, 0, 0],
                           [0, 0, 1, 0],
                           [0.5, 0, 0, 1]])
    num_of_samples = 1000
    samples = numpy.random.multivariate_normal(
        real_mu, real_cov, num_of_samples
    )
    multivariate_gaussian = MultivariateGaussian().fit(samples)
    print(f"estimated expectation:\n{multivariate_gaussian.mu_}")
    print(f"estimated covariance matrix:\n{multivariate_gaussian.cov_}")

    # Question 5 - Likelihood evaluation
    X_axis = np.linspace(-10, 10, 200)
    Y_axis = np.linspace(-10, 10, 200)

    t = np.meshgrid(X_axis, Y_axis)

    z = []

    for i in X_axis:
        for j in Y_axis:
            z.append(MultivariateGaussian.log_likelihood(
                np.asarray([i, 0, j, 0]), real_cov, samples
            ))
        print("\r[" + int((i+10)*5)*'*' + (100-int((i+10)*5))*' ' + ']',end='')

    print()
    zz = np.asarray(z).reshape(200, 200)
    go.Figure(go.Heatmap(x=X_axis, y=Y_axis, z=zz),
              layout=go.Layout(title="title", height=300, width=300)).show()
    # todo: titles, optimizations, delete printings, use np.meshgrid?
    # Question 6 - Maximum likelihood
    # todo: derivative?
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
