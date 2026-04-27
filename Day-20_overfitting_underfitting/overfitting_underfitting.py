import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("dataset.csv")

X = df[["X"]].values
y = df["Y"].values

# Function to train and plot model
def plot_model(degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)

    y_pred = model.predict(X_range_poly)

    plt.scatter(X, y)
    plt.plot(X_range, y_pred)
    plt.title(f"Polynomial Degree {degree}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Underfitting (too simple)
plot_model(1)

# Good Fit
plot_model(2)

# Overfitting (too complex)
plot_model(10)
