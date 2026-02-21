import pandas as pd

#Load cleaned dataset
file_path = "cleaned_dataset.csv"
df = pd.read_csv(file_path)

print("\n===Exploratory Statistics Report===")

#Basic information
print("Dataset shape:", df.shape)
print("Columns:")
print(df.columns)

#Mean
print("\nMean of each columns:")
print(df.mean(numeric_only=True))

#Median
print("\nMedian of each column:")
print(df.median(numeric_only=True))

#Standard deviation
print("\nStandard deviation")
print(df.std(numeric_only=True))

#Variance
print("\nVariance:")
print(df.var(numeric_only=True))

#Maximum and minimum value
print("\nMaximum value:")
print(df.max(numeric_only=True))

print("\nMinimum value:")
print(df.mean(numeric_only=True))

#Full statistical summary
print("\nFull statistical summary:")
print(df.describe())