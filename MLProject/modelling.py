import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

# Load Data
data = pd.read_csv("housing_clean_auto.csv")
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

run = mlflow.active_run()

model = RandomForestRegressor(
    n_estimators=args.n_estimators,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = mse ** 0.5

# ===== LOGGING =====
mlflow.log_param("n_estimators", args.n_estimators)
mlflow.log_metric("rmse", rmse)
mlflow.sklearn.log_model(model, artifact_path="model")

# ===== AMBIL RUN_ID =====
run_id = run.info.run_id
print(f"Training selesai. RMSE: {rmse}")
print(f"MLflow Run ID: {run_id}")

# ===== KIRIM KE GITHUB ACTIONS =====
if "GITHUB_ENV" in os.environ:
    with open(os.environ["GITHUB_ENV"], "a") as f:
        f.write(f"RUN_ID={run_id}\n")
