import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load dataset
df = pd.read_csv("cleaned_dataset.csv")

print("Generating Seaborn Pairplot...")

#Select only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

#Create pairplot
sns.pairplot(numeric_df)

#Show plot
plt.show()
