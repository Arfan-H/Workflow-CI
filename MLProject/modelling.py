import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# =====================
# Arg parser
# =====================
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

# =====================
# Load data
# =====================
df = pd.read_csv("housing_preprocessing.csv")
X = df.drop("median_income", axis=1)
y = df["median_income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# Train & log model
# =====================
mlflow.set_experiment("ci-experiment")

with mlflow.start_run():
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=42
    )
    model.fit(X_train, y_train)

    mlflow.log_param("n_estimators", args.n_estimators)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

print("✅ Model logged to MLflow")
