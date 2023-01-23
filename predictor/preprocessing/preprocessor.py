"""Module containing the preprocessing functionality"""
# pylint: disable=C0103

import numpy as np
import pandas as pd
import xlrd

from predictor.utils.constants import SELECTED_FEATURES


def preprocess_data(
    df_greenhouse: pd.DataFrame, df_weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Preprocesses the greenhouse dataframe and the weather dataframe and joins them.

    Args:
        df_greenhouse (pd.DataFrame): Greenhouse dataframe
        df_weather (pd.DataFrame): Weather dataframe

    Returns:
        pd.DataFrame: The preprocessed dataframe
    """

    df_greenhouse_preprocessed = _set_time(df_greenhouse)
    df_weather_preprocessed = _set_time(df_weather)

    df_joined = df_greenhouse_preprocessed.join(df_weather_preprocessed)

    # For an explanation on the feature selection, look in the exploration notebook or the
    # README.md
    df_feature_selection = df_joined[SELECTED_FEATURES]

    # For an explanation on the feature engineering, look in the exploration notebook or the
    # README.md
    df_feature_engineering = df_feature_selection.assign(
        t=lambda df: np.arange(len(df.index)) - (len(df.index) - 1),
        hour_of_day=lambda df: df.index.hour,
        month=lambda df: df.index.month,
    )

    # Impute the remaining NaN values and resample to hours
    df_output = (
        df_feature_engineering.fillna(df_feature_engineering.mean())
        .resample("1H")
        .mean()
    )

    return df_output


def _set_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the excel timestamp to a pandas timestamp and set as the index.

    Args:
        df (pd.DataFrame): Dataframe with a column "time" in excel format

    Returns:
        pd.DataFrame: Dataframe with new index
    """

    df_index = df.assign(
        time=lambda df: df["time"].apply(
            lambda col: pd.Timestamp(xlrd.xldate_as_datetime(col, 0))
        )
    ).set_index("time")

    return df_index
