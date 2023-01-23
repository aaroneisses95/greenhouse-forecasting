"""Module for analysing and visualising the data"""
# pylint: disable=C0103
# pylint: disable=R0913

from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.utils import resample

from predictor.utils.constants import N_SAMPLES


def generate_prediction_intervals(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    best_params: dict,
) -> Tuple[Any, Any]:
    """
    Generates the lower (P10) and upper (P90) prediction intervals

    Args:
        best_params (dict): The best parameters that are returned from the GridSearchCV

    Returns:
        Union[np.ndarray,np.ndarray]: The lower and upper percentiles as numpy arrays
    """

    best_max_depth = best_params["model__max_depth"]

    # Train a new model using the best hyperparameters on the full training dataset
    rf = RandomForestRegressor(max_depth=best_max_depth)
    rf.fit(X_train, y_train)

    # Train the model N_SAMPLES times and get the prediction of the test set for every
    # model. With all these predictions, we can calculate the prediction interval. However, it
    # is important to realise that for more accurate results we should increase N_SAMPLES but
    # this will also mean that it will take longer to run.
    predictions = np.zeros((N_SAMPLES, len(X_test)))

    for i in range(N_SAMPLES):
        print("\n")
        print(f"Sample: {i}")
        print("\n")
        X_resampled, y_resampled = resample(X_train, y_train)
        rf.fit(X_resampled, y_resampled)
        predictions[i] = rf.predict(X_test)

    # Calculate the 10th and 90th percentiles of the predictions
    lower = np.percentile(predictions, 10, axis=0)
    upper = np.percentile(predictions, 90, axis=0)

    return lower, upper


def generate_plot(
    df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    start_date: str,
    end_date: str,
    fig_path: str,
) -> None:
    """
    Generate a plot of the dataset with the y_true, y_pred and the prediction intervals.

    Args:
        df (pd.DataFrame): The entire dataset with the predicted column
        train (pd.DataFrame): Train data
        test (pd.DataFrame): Test data
        start_date (str): Start date of the figure
        end_date (str): End date of the figure
        fig_path (str): The name of file
    """

    fig, ax = plt.subplots(figsize=(18, 6))

    train[["Tair"]].plot(ax=ax, c="green")
    test[["Tair"]].plot(ax=ax, c="blue")
    df[["pred"]].plot(ax=ax, c="red")
    test[["P10"]].plot(ax=ax, c="black", linestyle="dashed")
    test[["P90"]].plot(ax=ax, c="black", linestyle="dotted")

    ax.legend(
        ["Train set", "Test set", "Prediction", "P10", "P90"],
        prop={"size": 15},
    )
    plt.xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    fig.savefig(fig_path)
    plt.close()


def generate_metric_report(
    target_values: pd.DataFrame, predicted_values: pd.DataFrame
) -> dict:
    """
    Generate metrics (MAE, MSE, RMSE, MAPE, R2) of the model

    Args:
        target_values (pd.DataFrame): The target values of the dataset
        predicted_values (pd.DataFrame): The predicted values based on the features of the dataset

    Returns:
        dict: Dictionary with the metrics
    """

    metric_report = {
        "mae": mean_absolute_error(target_values, predicted_values),
        "mse": mean_squared_error(target_values, predicted_values),
        "rmse": np.sqrt(mean_squared_error(target_values, predicted_values)),
        "mape": mean_absolute_percentage_error(target_values, predicted_values),
        "r2": r2_score(target_values, predicted_values),
    }

    return metric_report
