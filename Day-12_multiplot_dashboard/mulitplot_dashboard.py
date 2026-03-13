import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")

# Create dashboard layout (2 rows × 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(10,8))

# 1. Histogram
axes[0,0].hist(df["Age"], bins=10)
axes[0,0].set_title("Age Distribution")

# 2. Scatter Plot
axes[0,1].scatter(df["Experience"], df["Salary"])
axes[0,1].set_title("Experience vs Salary")

# 3. Boxplot
axes[1,0].boxplot(df["Salary"])
axes[1,0].set_title("Salary Outliers")

# 4. Bar Chart (Average Salary by Experience)
avg_salary = df.groupby("Experience")["Salary"].mean()
axes[1,1].bar(avg_salary.index, avg_salary.values)
axes[1,1].set_title("Avg Salary by Experience")

plt.tight_layout()
plt.show()
