"""Train endpoint for MLflow run"""

import argparse

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklego.preprocessing import RepeatingBasisFunction

from predictor.ingestion.loader import Dataloader
from predictor.preprocessing.preprocessor import Preprocessor


def generate_plot(
    df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, fig_path: str
) -> None:
    """
    Generate a plot of the last month with the y_true and y_pred

    Args:
        df (pd.DataFrame): _description_
        train (pd.DataFrame): _description_
        test (pd.DataFrame): _description_
        fig_path (str): _description_
    """

    fig, ax = plt.subplots(figsize=(18, 6))

    train[["Tair"]].plot(ax=ax, c="blue")
    test[["Tair"]].plot(ax=ax, c="green")
    df[["pred"]].plot(ax=ax, c="red")

    ax.legend(
        ["Train set", "Test set", "Prediction"],
        prop={"size": 15},
    )
    plt.xlim(pd.Timestamp("2020-05-01"), pd.Timestamp("2020-06-01"))
    fig.savefig(fig_path)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="mlflow training entry point")
    parser.add_argument("--team", help="Name of the team", type=str)

    args = parser.parse_args()

    with mlflow.start_run() as run:

        # Ingest the data
        dataloader = Dataloader()
        df_greenhouse, df_weather = dataloader.ingest_data(team=args.team)

        # Preprocess the weather and greenhouse data
        preprocessor = Preprocessor()
        df = preprocessor.preprocess_data(
            df_greenhouse=df_greenhouse, df_weather=df_weather
        )

        # Split the data in training data and testing data. Since we want to test for the coming
        # 24 hours, the test data is just the last day of the data set.
        train = df.loc[:"2020-05-29"]
        test = df.loc["2020-05-29":]

        X_train = train.drop(columns="Tair")
        y_train = train[["Tair"]]
        X_test = test.drop(columns="Tair")
        y_test = test[["Tair"]]

        # A RepeatingBasisFunction is introduced to deal with the cycle of the day
        rbf = RepeatingBasisFunction(
            n_periods=24, remainder="passthrough", column="hour_of_day"
        )

        # We define the hyperparameters
        parameters = {"model__max_depth": [2, 5, 10]}

        # To do cross-validation properly for a time-series dataset, we use TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)

        model = Pipeline(
            [
                ("preprocess", rbf),
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor()),
            ]
        )

        # We train the model
        gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=parameters)
        gsearch.fit(X_train, y_train)

        # Make predictions for the train and test data
        pred_train = gsearch.predict(X_train)
        pred_test = gsearch.predict(X_test)

        metric_report_train = {
            "mae": mean_absolute_error(y_train, pred_train),
            "mse": mean_squared_error(y_train, pred_train),
            "rmse": np.sqrt(mean_squared_error(y_train, pred_train)),
            "mape": mean_absolute_percentage_error(y_train, pred_train),
            "r2": r2_score(y_train, pred_train),
        }
        metric_report_test = {
            "mae": mean_absolute_error(y_test, pred_test),
            "mse": mean_squared_error(y_test, pred_test),
            "rmse": np.sqrt(mean_squared_error(y_test, pred_test)),
            "mape": mean_absolute_percentage_error(y_test, pred_test),
            "r2": r2_score(y_test, pred_test),
        }

        # Write the metrics for the train and test data set to a text file
        open("metrics_report_train.txt", "w").write(str(metric_report_train))
        open("metrics_report_test.txt", "w").write(str(metric_report_test))

        # Plot the predicted values alongside the
        df["pred"] = gsearch.predict(df.drop(columns="Tair"))
        generate_plot(df=df, train=train, test=test, fig_path="prediction_plot.png")

        # Log the different artifacts
        mlflow.log_artifact("metrics_report_train.txt")
        mlflow.log_artifact("metrics_report_test.txt")
        mlflow.log_artifact("prediction_plot.png")

        mlflow.sklearn.log_model(sk_model=gsearch, artifact_path="model")
