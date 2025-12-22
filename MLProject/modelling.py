import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

# Load data
data = pd.read_csv("housing_clean_auto.csv")
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print("Training selesai. MSE:", mse)
