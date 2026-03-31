import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset.csv")

# Use one feature for simplicity
X = df["Experience"].values
y = df["Salary"].values

# Calculate mean
mean_x = np.mean(X)
mean_y = np.mean(y)

# Calculate slope (m)
numerator = np.sum((X - mean_x) * (y - mean_y))
denominator = np.sum((X - mean_x) ** 2)

m = numerator / denominator

# Calculate intercept (b)
b = mean_y - (m * mean_x)

print("Slope (m):", m)
print("Intercept (b):", b)

# Predictions
y_pred = m * X + b

# Plot
plt.figure()
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.title("Linear Regression from Scratch")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
