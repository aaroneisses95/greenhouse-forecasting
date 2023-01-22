"Module containing the Preprocessor class"
import numpy as np
import pandas as pd
import xlrd

from predictor.utils.constants import SELECTED_FEATURES


class Preprocessor:
    """
    The Preprocessor class contains the methods that preprocess the data sets

    Example usage:
    -------------

    >>> from predictor.preprocessing.preprocessor import Preprocessor
    >>>
    >>> preprocessor = Preprocessor()
    >>> preprocessed_data = preprocessor.preprocess_data(df_greenhouse, df_weather)
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

        df_greenhouse_preprocessed = self._set_time(df_greenhouse)
        df_weather_preprocessed = self._set_time(df_weather)

        df_joined = df_greenhouse_preprocessed.join(df_weather_preprocessed)

        # For an explanation on the feature selection, look in the exploration notebook
        df_feature_selection = df_joined[SELECTED_FEATURES]

        # For an explanation on the feature engineering, look in the exploration notebook
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

    @staticmethod
    def _set_time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Covert the timestamp to a pandas timestamp and set as the index.

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
