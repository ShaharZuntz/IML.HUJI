import pandas
from scipy.stats import zscore

from IMLearn.learners import MultivariateGaussian
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

FEATURE_RESPONSE_PEARSON_COR_PLOT_TITLE_FORMAT = "feature:{}, correlation:{}"

FILL_TONEXTY = "tonexty"
CLR_LIGHT_GREY = "lightgrey"
MODE_MARKERS_AND_LINES = "markers+lines"
MODE_LINES = "lines"
PLOT_HEIGHT = 400
TITLE_ATTR = "title"

RESPONSE = 1
FEATURES = 0

COL_BEDROOMS = "bedrooms"
COL_PRICE = "price"
COL_YR_BUILT = "yr_built"
COL_YR_RENOVATED = "yr_renovated"
COL_DATE = "date"
COL_ID = "id"

HOUSE_PRICES_CSV_PATH = "../datasets/house_prices.csv"

Q4_LOWER_ERROR_TITLE = "mean - 2*std"
Q4_UPPER_ERROR_TITLE = "mean + 2*std"
Q4_LINE_TITLE = "Average Loss"
Q4_PLOT_YAXIS_TITLE = "average loss"
Q4_PLOT_XAXIS_TITLE = "percentage of sampled train data"
Q4_PLOT_TITLE = ("Average Loss as a Function of percentage of sampled train "
                 "data")


def load_data(filename: str) -> pd.DataFrame:
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
    loaded_data = pd.read_csv(filename)

    loaded_data = clean_data(loaded_data)

    return loaded_data


def clean_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the given data from errors and extreme anomalies
    """

    X = X.drop([COL_ID, COL_DATE, "zipcode"], axis=1)

    X = X.apply(pandas.to_numeric, errors='coerce')
    X = X.dropna()
    X = X.reset_index(drop=True)

    # validations
    X = X.drop(X[X.price <= 0].index)

    X = X.drop(X[X.bedrooms <= 0].index)
    X.loc[X[COL_BEDROOMS] == 33, COL_BEDROOMS] = 3

    X = X.drop(X[X.bathrooms <= 0].index)

    X = X.drop(X[X.floors <= 0].index)

    X = X.drop(X[X.sqft_living <= 0].index)
    X = X.drop(X[X.sqft_lot <= 0].index)
    X = X.drop(X[X.sqft_above <= 0].index)
    X = X.drop(X[X.sqft_basement < 0].index)  # can be 0 if not exist
    X = X.drop(X[X.sqft_living15 <= 0].index)
    X = X.drop(X[X.sqft_lot15 <= 0].index)

    X = X.drop(X[~X.waterfront.isin([0, 1])].index)

    X = X.drop(X[~X.view.isin(range(0, 5))].index)

    X = X.drop(X[~X.condition.isin(range(1, 6))].index)

    X = X.drop(X[~X.grade.isin(range(1, 14))].index)

    X = X.drop(X[~X.yr_built.isin(range(0, 2016))].index)
    X = X.drop(X[~X.yr_renovated.isin(range(0, 2016))].index)

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
    std_y = np.std(y)

    covariance_matrix = MultivariateGaussian().fit(
        np.array(pd.concat([X, y], axis=1))).cov_

    for feature_index, feature_name in zip(range(X.shape[0]), X):
        std_x = np.std(X[feature_name])
        p_correlation = covariance_matrix[feature_index, -1] / (std_x * std_y)

        go.Figure(
            [go.Scatter(x=X[feature_name], y=y, mode="markers")],
            layout=go.Layout(
                title=FEATURE_RESPONSE_PEARSON_COR_PLOT_TITLE_FORMAT.format(
                    feature_name, p_correlation),
                xaxis={TITLE_ATTR: feature_name},
                yaxis={TITLE_ATTR: COL_PRICE},
                height=PLOT_HEIGHT)
        ).write_image(f"{output_path}/{feature_name}.png")


def fit_model_over_data(train_set: Tuple[pd.DataFrame, pd.Series],
                        test_set: Tuple[pd.DataFrame, pd.Series],
                        start_p: int, end_p: int = 100,
                        repetitions: int = 10) -> None:
    """
    For every percentage p in <start_p>%, <start_p+1>%, ..., <end_p>%, repeat
    the following <repetitions> times:
      1) Sample p% of the overall training data
      2) Fit linear model (including intercept) over sampled set
      3) Test fitted model over test set
      4) Store average and variance of loss over test set
    Then plot average loss as function of training size with error ribbon of
    size (mean-2*std, mean+2*std)
    """
    test_features_arr = np.asarray(test_set[FEATURES])
    test_response_arr = np.asarray(test_set[RESPONSE])

    p_values = tuple(range(start_p, end_p + 1))
    loss_values = list()
    var_loss = list()

    for p in p_values:
        train_concat = pd.concat(
            [train_set[FEATURES], train_set[RESPONSE]], axis=1)
        p_loss_values = list()

        for i in range(repetitions):
            train_sample = train_concat.sample(frac=(p / 100))

            model = LinearRegression()
            model.fit(train_sample.drop([COL_PRICE], axis=1),
                      train_sample[COL_PRICE])

            p_loss_values.append(
                model.loss(test_features_arr, test_response_arr))

        loss_values.append(np.mean(p_loss_values, axis=0))
        var_loss.append(np.std(p_loss_values, axis=0))

    go.Figure([
        go.Scatter(x=p_values, y=loss_values, name=Q4_LINE_TITLE,
                   showlegend=True, mode=MODE_MARKERS_AND_LINES),
        go.Scatter(x=p_values,
                   y=np.array(loss_values) + 2 * np.array(var_loss),
                   fill=FILL_TONEXTY, mode=MODE_LINES,
                   line=dict(color=CLR_LIGHT_GREY),
                   showlegend=False, name=Q4_UPPER_ERROR_TITLE),
        go.Scatter(x=p_values,
                   y=np.array(loss_values) - 2 * np.array(var_loss),
                   fill=None, mode=MODE_LINES, line=dict(color=CLR_LIGHT_GREY),
                   showlegend=False, name=Q4_LOWER_ERROR_TITLE)
    ],
        layout=go.Layout(
            title=Q4_PLOT_TITLE,
            xaxis={TITLE_ATTR: Q4_PLOT_XAXIS_TITLE},
            yaxis={TITLE_ATTR: Q4_PLOT_YAXIS_TITLE},
            height=PLOT_HEIGHT)
    ).show()


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    data = load_data(HOUSE_PRICES_CSV_PATH)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data.drop([COL_PRICE], axis=1), data[COL_PRICE])

    # Question 3 - Split samples into training- and testing sets.
    (train_features, train_prices,
     test_features, test_prices) = split_train_test(
        data.drop([COL_PRICE], axis=1), data[COL_PRICE], 0.75)

    # Question 4 - Fit model over increasing percentages of the overall
    # training data.
    fit_model_over_data((train_features, train_prices),
                        (test_features, test_prices), 10)
