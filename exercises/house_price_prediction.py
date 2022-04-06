import pandas

from IMLearn.learners import MultivariateGaussian
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)

    X = clean_data(df)

    X = process_features(X)

    y = X["price"]
    X = X.drop(["price", "yr_renovated", "yr_built"], axis=1)

    return X, y


def process_features(X):
    # todo: zipcode,lat-long - drop or make valuable
    #  find correlated features that can be dropped
    #  generate new valuable features out of the existing ones

    X["yr_max"] = X[["yr_built", "yr_renovated"]].max(axis=1)
    X["yr_sum"] = X["yr_built"] + 5 * X["yr_renovated"]
    center = np.mean(X["lat"]), np.mean(X["long"])

    X["dist_from_center"] = ((X["lat"] - center[0]) ** 2 +
                             (X["long"] - center[1]) ** 2) ** 0.5
    X["zipcode"] = X["zipcode"] // 100
    one_hot = pd.get_dummies(X["zipcode"])
    X = X.join(one_hot)
    return X


def clean_data(df):
    X = df.drop(["id", "date"], axis=1)

    # todo: consider using date again
    # X = X.drop(X['T' not in str(X.date)].index)
    # X["date"] = pd.Series(int(x[0][:4]) for x in X["date"].str.split('T'))

    X = X.apply(pandas.to_numeric, errors='coerce')
    X = X.dropna()
    X = X.reset_index(drop=True)
    X = X.drop(X[X.price <= 0].index)
    X = X.drop(X[X.bedrooms <= 0].index)
    X = X.drop(X[X.bathrooms <= 0].index)
    X = X.drop(X[X.floors <= 0].index)
    X = X.drop(X[X.sqft_living <= 0].index)
    X = X.drop(X[X.sqft_lot <= 0].index)
    X = X.drop(X[X.sqft_above <= 0].index)
    X = X.drop(X[X.sqft_basement < 0].index)
    X = X.drop(X[X.sqft_living15 <= 0].index)
    X = X.drop(X[X.sqft_lot15 <= 0].index)
    X = X.drop(X[(X.waterfront < 0) | (X.waterfront > 1)].index)
    X = X.drop(X[(X.view < 0) | (X.view > 4)].index)
    X = X.drop(X[(X.condition < 1) | (X.condition > 5)].index)
    X = X.drop(X[(X.grade < 1) | (X.grade > 13)].index)
    X = X.drop(X[(X.yr_built <= 0) | (X.yr_built > 2015)].index)
    X = X.drop(X[(X.yr_renovated < 0) |
                 ((X.yr_renovated != 0) & (
                         X.yr_renovated < X.yr_built))].index)
    X = X.drop(X[X.zipcode <= 0].index)
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # todo: switch between write_image and show
    sigma_y = np.std(y)
    Xy = np.concatenate(
        (np.asarray(X.transpose()), np.asarray(y).reshape((1, -1))))
    covariance_matrix = MultivariateGaussian().fit(Xy.transpose()).cov_

    for feature_index, feature_name in zip(range(X.shape[0]), X):
        sigma_x = np.std(X[feature_name])
        p_correlation = covariance_matrix[feature_index, -1] / (
                sigma_x * sigma_y)

        go.Figure(
            [go.Scatter(x=X[feature_name], y=y, mode="markers")],
            layout=go.Layout(
                title=f"feature:{feature_name}, correlation:{p_correlation}",
                xaxis={"title": f"{feature_name}"},
                yaxis={"title": "price"},
                height=400)
        ).show()
        # ).write_image(f"{output_path}/{feature_name}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, response = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(data, response, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall
    # training data.
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10
    # times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of
    # size (mean-2*std, mean+2*std)
    test_x_array, test_y_array = np.asarray(test_x), np.asarray(test_y)
    p_values = np.arange(10, 101)
    loss_values = np.ndarray((91,))
    var_loss = np.ndarray((91,))
    for p in p_values:
        train = pd.concat([train_x, train_y], axis=1)
        loss_array = np.ndarray((10,))
        for i in range(10):
            p_train_xy = train.sample(frac=p / 100)
            model = LinearRegression()
            model.fit(p_train_xy.drop(["price"], axis=1), p_train_xy["price"])
            loss_array[i] = model.loss(test_x_array, test_y_array)
        loss_values[p - 10] = np.mean(loss_array, axis=0)
        var_loss[p - 10] = np.std(loss_array, axis=0)

    fig = go.Figure([
        go.Scatter(x=p_values, y=loss_values, name="Real Model",
                   showlegend=True, mode="markers+lines"),
        go.Scatter(x=p_values, y=loss_values + 2 * var_loss, fill='tonexty',
                   mode="lines", line=dict(color="lightgrey"),
                   showlegend=False),
        go.Scatter(x=p_values, y=loss_values - 2 * var_loss, fill=None,
                   mode="lines", line=dict(color="lightgrey"),
                   showlegend=False)
    ],
        layout=go.Layout(
            title="loss as a function of percentage of train data",
            xaxis={"title": "percentage"},
            yaxis={"title": "average loss"},
            height=400)
    )

    fig.show()
