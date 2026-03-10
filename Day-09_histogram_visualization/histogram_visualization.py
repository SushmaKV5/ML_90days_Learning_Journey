import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset
df = pd.read_csv("cleaned_dataset.csv")

print("\nCreating histogram for numerical columns")

#Select numerical columns
numeric_cols = df.select_dtypes(include = ['int64', 'float64']).columns

#Create histogram for each columns
for col in numeric_cols:
    plt.figure()
    plt.hist(df[col], bins=10)

    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()