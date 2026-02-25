import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")

print("Generating Visualizations...")

#1. Histogram (Distribution of Age)
plt.figure()
plt.hist(df["Age"])
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#2. Scatter Plot (Experience vs Salary)
plt.figure()
plt.scatter(df["Experience"], df["Salary"])
plt.title("Experience vs Salary")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#3. Bar Chart (Average Salary by Experience)
avg_salary = df.groupby("Experience")["Salary"].mean()

plt.figure()
avg_salary.plot(kind="bar")
plt.title("Average Salary by Experience")
plt.xlabel("Experience")
plt.ylabel("Average Salary")
plt.show()