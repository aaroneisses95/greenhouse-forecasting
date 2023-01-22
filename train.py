"""Train endpoint for MLflow run"""

import argparse

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

        dataloader = Dataloader()
        df_greenhouse, df_weather = dataloader.ingest_data(team=args.team)

        preprocessor = Preprocessor()
        df = preprocessor.preprocess_data(
            df_greenhouse=df_greenhouse, df_weather=df_weather
        )

        train = df.loc[:"2020-05-29"]
        test = df.loc["2020-05-29":]
        X_train = train.drop(columns="Tair")
        y_train = train[["Tair"]]
        X_test = test.drop(columns="Tair")
        y_test = test[["Tair"]]

        rbf = RepeatingBasisFunction(
            n_periods=24, remainder="passthrough", column="hour_of_day"
        )

        model = Pipeline(
            [
                ("preprocess", rbf),
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor()),
            ]
        )

        # parameters = {"model__max_depth": [1, 2, 3, 4, 5, 7, 8, 9, 10]}
        parameters = {"model__max_depth": [5, 10, 12]}

        tscv = TimeSeriesSplit(n_splits=5)

        gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=parameters)

        gsearch.fit(X_train, y_train)

        results = pd.DataFrame(gsearch.cv_results_)

        # lr = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
        # lr.fit(train_x, train_y)

        pred = gsearch.predict(X_test)

        mlflow.log_metrics(
            {
                # "team": args.team,
                "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                "mae": mean_absolute_error(y_test, pred),
                "r2": r2_score(y_test, pred),
            }
        )

        mlflow.sklearn.log_model(gsearch, "model")
