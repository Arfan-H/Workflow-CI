import mlflow
import os

mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("housing-ci")

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mlflow.set_experiment("ci-experiment")

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

data = pd.read_csv("housing_clean_auto.csv")
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("ci-experiment")

with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # ===== LOGGING =====
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_metric("rmse", mse ** 0.5)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

print("Training selesai. MSE:", mse)
