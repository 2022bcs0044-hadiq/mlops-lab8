import pandas as pd
import json
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
data = pd.read_csv("data/housing.csv")

# Remove missing values
data = data.dropna()

# Convert categorical column
data = pd.get_dummies(data, columns=["ocean_proximity"])

# Target
y = data["median_house_value"]

# Features
X = data.drop("median_house_value", axis=1)

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
pred = model.predict(X)

# Metrics
rmse = np.sqrt(mean_squared_error(y, pred))
r2 = r2_score(y, pred)

# Save model
joblib.dump(model, "model.pkl")

# Save metrics
metrics = {
    "rmse": rmse,
    "r2": r2,
    "dataset_size": len(data)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print(metrics)