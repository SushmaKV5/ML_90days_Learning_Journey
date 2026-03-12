import pandas as pd
import matplotlib.pyplot as plt

#loading the data
df = pd.read_csv("cleaned_dataset.csv")

print("Generating boxplots for outlier detection...")

#Selecting only numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

#Create boxplot for each columns
for col in numeric_cols:
    plt.figure()
    plt.boxplot(df[col])
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.show()
