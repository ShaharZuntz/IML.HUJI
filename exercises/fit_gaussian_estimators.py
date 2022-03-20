import numpy.random

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian

from utils import *

TITLE_FONT = {"family": "Times New Roman", "size": 20}

TITLE_FIGURE_Q1 = "Absolute Distance Between Estimated and True Value of " \
                  "the Expectation as a Function of the Sample Size"
TITLE_FIGURE_Q3 = "The Empirical PDF Function Under the Fitted Model"

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    real_mu = 10
    real_var = 1
    sample_size = 1000

    samples = np.random.normal(real_mu, real_var, size=sample_size)

    univariate_gaussian = UnivariateGaussian().fit(samples)
    print(f"({univariate_gaussian.mu_}, {univariate_gaussian.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    num_of_samples = np.linspace(10, sample_size, 100).astype(np.int64)

    estimated_mean = []
    for n in num_of_samples:
        estimated_mean.append(
            abs(real_mu - UnivariateGaussian().fit(samples[:n]).mu_)
        )

    go.Figure(
        [go.Scatter(x=num_of_samples, y=estimated_mean, mode='markers+lines')],
        layout=go.Layout(
            title={"text": TITLE_FIGURE_Q1, "font": TITLE_FONT},
            xaxis={"title": "Sample Size"},
            yaxis={"title": "r$|\mu-\hat\mu|$"},
            height=500
        )).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    PDF = univariate_gaussian.pdf(samples)

    go.Figure(
        [go.Scatter(x=samples, y=PDF, mode='markers')],
        layout=go.Layout(
            title={"text": TITLE_FIGURE_Q3, "font": TITLE_FONT},
            xaxis={"title": "Sample Values"},
            yaxis={"title": "PDF Values"},
            height=500
        )).show()


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
    print(f"Estimated expectation:\n{multivariate_gaussian.mu_}")
    print(f"Estimated covariance matrix:\n{multivariate_gaussian.cov_}")

    # Question 5 - Likelihood evaluation
    X_axis = np.linspace(-10, 10, 200)
    Y_axis = np.linspace(-10, 10, 200)

    # t = np.meshgrid(X_axis, Y_axis)

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
    # # todo: titles, optimizations, delete printings, use np.meshgrid?
    # # Question 6 - Maximum likelihood
    # # todo: derivative?
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)

    # todo:
    #  Q1 - write expectation and variance in the pdf
    #  Q2 - copy graph to pdf
    #  Q3 - copy graph to pdf, (verify correctness), answer question (max at
    #       real_mu, get denser as y-value get higher)
    test_univariate_gaussian()

    # todo:
    #  gaussian_estimators.py - fix and finish MultivariateGaussian
    #  Q4 - write expectation and cov matrix in the pdf
    #  Q5 - improve efficiency (using numpy - moodle, WhatsApp),
    #       meaningful titles, copy graph to pdf, answer question
    #  Q6 - answer question, check the way to calculate the max
    test_multivariate_gaussian()
