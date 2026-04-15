import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("STEP 1: Loading data...")

df = pd.read_csv("data/housing.csv")

print("STEP 2: Cleaning data...")

df = df.select_dtypes(include=['float64','int64']).dropna()

print("STEP 3: Preparing data...")

X = df.drop(df.columns[-1], axis=1)
y = df[df.columns[-1]]

print("STEP 4: Training model...")

model = LinearRegression()
model.fit(X, y)

print("STEP 5: Predicting...")

pred = model.predict(X)

print("STEP 6: Calculating metrics...")

from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y, pred))

r2 = r2_score(y, pred)

print("\n✅ RESULTS:")
print("RMSE:", rmse)
print("R2:", r2)
print("Dataset size:", len(df))