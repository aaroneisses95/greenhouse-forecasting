"""Module for loading the data"""
import os
from typing import Union

import pandas as pd

from predictor.utils.constants import PATH_WEATHER_CSV, TEAMS


def ingest_data(team: str) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    Ingests the greenhouse dataset and the weather dataset of the selected team.

    Args:
        team (str): Name of the team that will be used to train the model.

    Raises:
        ValueError: Raises an error if the given team is not in the TEAMS list. It is case
        sensitive.

    Returns:
        Union[pd.DataFrame, pd.DataFrame]: The greenhouse and weather datasets
    """

    if team not in TEAMS:
        raise ValueError(f"Team {team} does not exist")

    path_greenhouse_csv = os.path.abspath(f"data/{team}/GreenhouseClimate.csv")

    df_greenhouse = pd.read_csv(path_greenhouse_csv)
    df_weather = pd.read_csv(PATH_WEATHER_CSV)

    return df_greenhouse, df_weather
