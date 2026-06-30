import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset.csv")

print("Dataset:\n")
print(df)

# =========================
# Correlation Matrix
# =========================

corr_matrix = df.corr(numeric_only=True)

print("\nCorrelation Matrix:\n")
print(corr_matrix)

# =========================
# Heatmap Visualization
# =========================

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True)

plt.title("Feature Correlation Heatmap")
plt.show()

# =========================
# Feature Selection
# =========================

target = "PerformanceScore"

correlation_with_target = corr_matrix[target].sort_values(ascending=False)

print("\nCorrelation with Target:\n")
print(correlation_with_target)

# Select important features
selected_features = correlation_with_target[abs(correlation_with_target) > 0.5].index

print("\nSelected Features:\n")
print(selected_features)
