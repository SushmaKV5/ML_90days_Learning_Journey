import pandas as pd
from sklearn.model_selection import test_train_split
from sklear.linear_model import LinearRegression
import matplot.pyplotlib as plt

#Load dataset
df = pd.csv("dataset.csv")

print("Dataset loaded!")

#Feature and target
X = df['experience']
y = df['Salary']

#Handle the missing values
X = X.fillna(X.mean())

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Model
model = LinearRegression()

#Model train
model.fit(X_train, y_train)

print("Model loaded successfully!")

#Prediction
y_pred = model.predict(X_test)

# Output predictions
print("Predictions:")
print(y_pred[:5])

# Plot results
plt.figure()
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.title("Experience vs Salary (Linear Regression)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
