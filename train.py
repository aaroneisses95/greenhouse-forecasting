"""Train endpoint for MLflow run"""

import argparse

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklego.preprocessing import RepeatingBasisFunction

from predictor.ingestion.loader import Dataloader
from predictor.preprocessing.preprocessor import Preprocessor

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
        parameters = {"model__max_depth": [5, 10, 15]}

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

        open("metrics_report_train.txt", "w").write(str(metric_report_train))
        open("metrics_report_test.txt", "w").write(str(metric_report_test))

        mlflow.log_artifact("metrics_report_train.txt")
        mlflow.log_artifact("metrics_report_test.txt")

        mlflow.sklearn.log_model(sk_model=gsearch, artifact_path="model")
