from typing import Tuple, Any, Union, List

import scipy.stats
from pandas import Series, DataFrame
from pandas.core.generic import NDFrame
from scipy.stats import zscore

from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio


Q2_SCATTER_PLOT_TITLE = ("Temperature in Israel as a function of Day of "
                         "year, 1995-2007")
Q2_BAR_PLOT_TITLE = ("Standard Deviation of temperature in Israel for each "
                     "month, 1996-2007")
Q4_PRINT_FORMAT = "k={},loss={}"
Q4_BAR_PLOT_TITLE = ("Loss values as a function of the degree (1 <= k <= 10) "
                     "of the polynomial fit")
Q5_BAR_PLOT_TITLE = ("Loss of Polynomial fit of degree {} trained on "
                     "Israel as a function of all other countries")

COUNTRY_ISRAEL = "Israel"
COUNTRY_NETHERLANDS = "The Netherlands"
COUNTRY_SOUTH_AFRICA = "South Africa"
COUNTRY_JORDAN = "Jordan"

COL_STD = "std"
COL_MEAN = "mean"
COL_YEAR = "Year"
COL_TEMP = "Temp"
COL_DAY_OF_YEAR = "DayOfYear"
COL_DATE = "Date"
COL_MONTH = "Month"
COL_COUNTRY = "Country"

CITY_TEMPERATURE_CSV_PATH = "../datasets/City_Temperature.csv"

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess loaded_data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    loaded_data = pd.read_csv(filename, parse_dates=[COL_DATE])

    loaded_data = clean_data(loaded_data)
    loaded_data = process_features(loaded_data)

    loaded_data = loaded_data.drop([COL_DATE], axis=1)

    return loaded_data


def clean_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the given data from extreme anomalies
    """
    X["zscore"] = zscore(X[COL_TEMP])
    X = X.drop(X[(X.zscore <= -3) | (X.zscore >= 3)].index)
    X = X.drop(["zscore"], axis=1)

    X = X.drop(X[X.Year <= 0].index)
    X = X.drop(X[(X.Month <= 0) | (X.Month >= 13)].index)
    X = X.drop(X[(X.Day <= 0) | (X.Day >= 32)].index)

    return X


def process_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a basic processing of the given data
    """
    X[COL_DAY_OF_YEAR] = X[COL_DATE].dt.dayofyear
    X[COL_YEAR] = X[COL_YEAR].astype(str)

    return X


def explore_data_for_specific_country(
        country_name: str, full_data: DataFrame) -> None:
    """
    Explores the data of a given country in the data.
    """
    country_data = get_country_data(country_name, full_data)
    px.scatter(country_data, x=COL_DAY_OF_YEAR, y=COL_TEMP, color=COL_YEAR,
               title=Q2_SCATTER_PLOT_TITLE).show()

    std_per_month = country_data.groupby(COL_MONTH).Temp.agg(np.std)

    px.bar(std_per_month, title=Q2_BAR_PLOT_TITLE).show()


def get_country_data(country_name: str, full_data: pd.DataFrame) -> DataFrame:
    """
    Returns a subset the given data that contains samples only from the given
    country.
    """
    return full_data[full_data[COL_COUNTRY] == country_name]


def explore_differences_between_countries(full_data: pd.DataFrame) -> None:
    """
    Explores the differences between the countries.
    """
    std_and_mean_per_country_and_month = full_data.groupby(
        [COL_COUNTRY, COL_MONTH]).Temp.agg([np.mean, np.std])

    px.line(std_and_mean_per_country_and_month,
            x=std_and_mean_per_country_and_month.index.get_level_values(1),
            y=COL_MEAN,
            color=std_and_mean_per_country_and_month.index.get_level_values(0),
            error_y=COL_STD).show()


def fit_model_for_different_pol_deg(
        min_k: int, max_k: int, country_name: str,
        full_data: DataFrame) -> None:
    """
    Fits a model (Polyfit) to a given country with different polynomial
    degrees.
    """

    country_data = get_country_data(country_name, full_data)

    train_x, train_y, test_x, test_y = split_train_test(
        country_data.drop([COL_TEMP], axis=1),
        country_data[COL_TEMP]
    )

    loss_values = list()
    for k in range(min_k, max_k + 1):
        polyfit = PolynomialFitting(k)
        polyfit.fit(train_x[COL_DAY_OF_YEAR], np.array(train_y))

        loss = polyfit.loss(test_x[COL_DAY_OF_YEAR], np.array(test_y))
        rounded_loss = round(loss, 2)

        print(Q4_PRINT_FORMAT.format(k, rounded_loss))
        loss_values.append(rounded_loss)

    px.bar(x=range(min_k, max_k + 1), y=loss_values,
           title=Q4_BAR_PLOT_TITLE).show()


def evaluate_fitted_model_on_different_countries(
        chosen_k: int, original_country_name: str,
        full_data: DataFrame, other_countries: List) -> None:
    """
    Evaluates a fitted model of the given country on the given other
    countries.
    """

    original_country_data = get_country_data(original_country_name, full_data)

    countries_losses = list()

    polyfit = PolynomialFitting(chosen_k)
    polyfit.fit(original_country_data[COL_DAY_OF_YEAR],
                original_country_data[COL_TEMP])

    for country in other_countries:
        country_data = data[data[COL_COUNTRY] == country]

        country_loss = polyfit.loss(country_data[COL_DAY_OF_YEAR],
                                    country_data[COL_TEMP])
        countries_losses.append(country_loss)

    px.bar(x=other_countries, y=countries_losses,
           title=Q5_BAR_PLOT_TITLE.format(chosen_k)).show()


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data(CITY_TEMPERATURE_CSV_PATH)

    # Question 2 - Exploring data for specific country
    explore_data_for_specific_country(COUNTRY_ISRAEL, data)

    # Question 3 - Exploring differences between countries
    explore_differences_between_countries(data)

    # Question 4 - Fitting model for different values of `k`
    fit_model_for_different_pol_deg(1, 10, COUNTRY_ISRAEL, data)

    # Question 5 - Evaluating fitted model on different countries
    evaluate_fitted_model_on_different_countries(
        5, COUNTRY_ISRAEL, data,
        [COUNTRY_JORDAN, COUNTRY_SOUTH_AFRICA, COUNTRY_NETHERLANDS]
    )
