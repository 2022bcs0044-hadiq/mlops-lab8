import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

data = pd.read_csv("data/housing.csv")
data = data.dropna()

data = pd.get_dummies(data, columns=["ocean_proximity"])

y = data["median_house_value"]
X = data.drop("median_house_value", axis=1)

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

rmse = mean_squared_error(y, pred, squared=False)
r2 = r2_score(y, pred)

joblib.dump(model, "model.pkl")

metrics = {
    "rmse": rmse,
    "r2": r2,
    "dataset_size": len(data)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print(metrics)