from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    real_mu = 10
    samples = np.random.normal(real_mu, 1, size=1000)

    univariate_gaussian = UnivariateGaussian.fit(UnivariateGaussian(), samples)
    print(f"({univariate_gaussian.mu_}, {univariate_gaussian.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(np.int64)
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
            xaxis_title="r$\\text{number of samples}$",
            yaxis_title="r$|\mu-\hat\mu|$",
            height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    # todo: points along the x axis - empirical, samples
    #  points along the y axis - theoretical, univariate_gaussian.pdf(samples)
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
