import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("dataset.csv")

# Features & Target
X = df[["Age", "Salary"]]
y = df["Purchased"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Model WITHOUT Scaling
# ===============================
model_no_scale = KNeighborsClassifier(n_neighbors=3)
model_no_scale.fit(X_train, y_train)

y_pred_no_scale = model_no_scale.predict(X_test)
accuracy_no_scale = accuracy_score(y_test, y_pred_no_scale)

print("Accuracy WITHOUT Scaling:", accuracy_no_scale)

# ===============================
# Model WITH Scaling
# ===============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = KNeighborsClassifier(n_neighbors=3)
model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = model_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

print("Accuracy WITH Scaling:", accuracy_scaled)
