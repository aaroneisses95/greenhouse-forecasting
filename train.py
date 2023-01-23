"""Train endpoint for MLflow run"""

import argparse

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklego.preprocessing import RepeatingBasisFunction

from predictor.analysis.analyser import (
    generate_metric_report,
    generate_plot,
    generate_prediction_intervals,
)
from predictor.ingestion.loader import ingest_data
from predictor.preprocessing.preprocessor import preprocess_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="mlflow training entry point")
    parser.add_argument("--team", help="Name of the team", type=str)

    args = parser.parse_args()

    with mlflow.start_run() as run:

        # Ingest the data
        df_greenhouse, df_weather = ingest_data(team=args.team)

        # Preprocess the weather and greenhouse data
        df = preprocess_data(df_greenhouse=df_greenhouse, df_weather=df_weather)

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

        # Get the best parameters
        best_params = gsearch.best_params_

        # Make predictions for the train and test data
        pred_train = gsearch.predict(X_train)
        pred_test = gsearch.predict(X_test)

        # Create the metric reports
        metric_report_train = generate_metric_report(y_train, pred_train)
        metric_report_test = generate_metric_report(y_test, pred_test)

        # Write the metrics for the train and test data set to a text file
        open("metrics_report_train.txt", "w").write(str(metric_report_train))
        open("metrics_report_test.txt", "w").write(str(metric_report_test))

        # Plot the predicted values alongside the train and test data
        df["pred"] = gsearch.predict(df.drop(columns="Tair"))

        # Add P10 and P90 to the test data set
        lower, upper = generate_prediction_intervals(
            X_train=X_train, y_train=y_train, X_test=X_test, best_params=best_params
        )
        test["P10"] = lower
        test["P90"] = upper

        # Generate the plot of last month
        generate_plot(
            df=df,
            train=train,
            test=test,
            start_date="2020-05-01",
            end_date="2020-06-01",
            fig_path="prediction_plot_lastmonth.png",
        )

        # Generate the plot of 5 days
        generate_plot(
            df=df,
            train=train,
            test=test,
            start_date="2020-05-25",
            end_date="2020-06-01",
            fig_path="prediction_plot_last5days.png",
        )

        # Log the different artifacts
        mlflow.log_artifact("metrics_report_train.txt")
        mlflow.log_artifact("metrics_report_test.txt")
        mlflow.log_artifact("prediction_plot_lastmonth.png")
        mlflow.log_artifact("prediction_plot_last5days.png")

        mlflow.sklearn.log_model(sk_model=gsearch, artifact_path="model")
