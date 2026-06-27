import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

print("Original Data:\n")
print(df)

# =========================
# 1. Label Encoding
# =========================

df["Gender"] = df["Gender"].map({
    "Male": 0,
    "Female": 1
})

print("\nAfter Label Encoding (Gender):\n")
print(df)

# =========================
# 2. One-Hot Encoding
# =========================

df = pd.get_dummies(df, columns=["City"])

print("\nAfter One-Hot Encoding (City):\n")
print(df)
