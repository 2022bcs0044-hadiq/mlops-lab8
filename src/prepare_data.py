import pandas as pd

# load dataset
df = pd.read_csv("data/housing.csv")

# take first 5000 rows
df = df.head(5000)

# save dataset version 1
df.to_csv("data/housing.csv", index=False)

print("Dataset Version 1 created")