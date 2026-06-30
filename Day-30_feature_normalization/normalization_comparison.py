import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load dataset
df = pd.read_csv("dataset.csv")

print("Original Data:\n")
print(df)

# =========================
# 1. Min-Max Normalization
# =========================

minmax = MinMaxScaler()
df_minmax = pd.DataFrame(minmax.fit_transform(df), columns=df.columns)

print("\nMin-Max Normalized Data:\n")
print(df_minmax)

# =========================
# 2. Standardization
# =========================

scaler = StandardScaler()
df_standard = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("\nStandardized Data:\n")
print(df_standard)
