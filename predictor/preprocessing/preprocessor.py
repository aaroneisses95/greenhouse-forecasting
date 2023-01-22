"Module containing the Preprocessor class"
import numpy as np
import pandas as pd
import xlrd


class Preprocessor:
    """
    The Preprocessor class contains the methods that preprocess the data sets

    Example usage:
    -------------

    >>> from predictor.preprocessing.preprocessor import Preprocessor
    >>>
    >>> preprocessor = Preprocessor()
    >>> preprocessed_data = preprocessor.preprocess_data()
    """

    def __init__(self) -> None:
        """
        Initialization of the Preprocessor class.
        """

    def preprocess_data(
        self, df_greenhouse: pd.DataFrame, df_weather: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Preprocesses the weather dataset and the greenhouse dataset and joins them.

        Args:
            df_greenhouse (pd.DataFrame): Greenhouse dataset
            df_weather (pd.DataFrame): Greenhouse dataset

        Returns:
            pd.DataFrame: The preprocessed and joined dataset
        """

        df_greenhouse_preprocessed = (
            df_greenhouse.assign(
                time=lambda df: df["time"].apply(
                    lambda col: pd.Timestamp(xlrd.xldate_as_datetime(col, 0))
                )
            )
            .set_index("time")
            .resample("1H")
            .mean()
        )

        df_weather_preprocessed = (
            df_weather.assign(
                time=lambda df: df["time"].apply(
                    lambda col: pd.Timestamp(xlrd.xldate_as_datetime(col, 0))
                )
            )
            .set_index("time")
            .resample("1H")
            .mean()
        )

        df_joined = df_greenhouse_preprocessed.join(df_weather_preprocessed)

        df_enriched = df_joined.assign(
            period_num=lambda df: np.arange(len(df.index)),
            hour_of_day=lambda df: df.index.hour,
            month=lambda df: df.index.month,
        )

        # Remove features with many NaN values
        df_dropped_columns = df_enriched.drop(
            columns=[
                "int_blue_vip",
                "int_farred_vip",
                "int_red_vip",
                "int_white_vip",
                "t_vent_sp",
            ]
        )

        # Impute the remaining NaN values
        df_output = df_dropped_columns.fillna(df_dropped_columns.mean())

        return df_output
