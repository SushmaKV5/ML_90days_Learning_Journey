import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Features & Target
X = df[["Age", "Salary"]]
y = df["Purchased"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Visualization
metrics = ["Precision", "Recall", "F1 Score"]
values = [precision, recall, f1]

plt.figure()
plt.bar(metrics, values)

plt.title("Model Evaluation Metrics")
plt.xlabel("Metrics")
plt.ylabel("Score")

plt.ylim(0, 1)
plt.show()
