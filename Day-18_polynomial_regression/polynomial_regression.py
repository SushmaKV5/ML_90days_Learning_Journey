import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("dataset.csv")

# Feature & Target
X = df[["Experience"]]
y = df["Salary"]

# Handle missing values
X = X.fillna(X.mean())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to Polynomial Features (degree = 2)
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predictions
y_pred = model.predict(X_test_poly)

# Visualization
plt.figure()
plt.scatter(X, y, label="Actual Data")

# Sort values for smooth curve
X_sorted = np.sort(X.values, axis=0)
X_sorted_poly = poly.transform(X_sorted)

plt.plot(X_sorted, model.predict(X_sorted_poly), label="Polynomial Curve")

plt.title("Polynomial Regression (Degree 2)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()
