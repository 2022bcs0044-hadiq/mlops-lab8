import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# load dataset
data = pd.read_csv("data/housing.csv")

# remove missing values
data = data.dropna()

# convert categorical column to numeric
data = pd.get_dummies(data, columns=["ocean_proximity"])

# target
y = data["median_house_value"]

# features
X = data.drop("median_house_value", axis=1)

# train model
model = LinearRegression()
model.fit(X, y)

# save model
joblib.dump(model, "model.pkl")

print("Model trained and saved")