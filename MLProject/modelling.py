import os

# FORCE MLflow ke directory CI (ANTI /C:)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

os.environ["MLFLOW_TRACKING_URI"] = f"file:{MLRUNS_DIR}"
os.environ["MLFLOW_ARTIFACT_URI"] = f"file:{MLRUNS_DIR}"

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

data = pd.read_csv("housing_clean_auto.csv")
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=args.n_estimators,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

mlflow.log_param("n_estimators", args.n_estimators)
mlflow.log_metric("mse", mse)

# IMPORTANT: name bukan artifact_path (MLflow terbaru)
mlflow.sklearn.log_model(model, name="model")

print("Training selesai. MSE:", mse)
