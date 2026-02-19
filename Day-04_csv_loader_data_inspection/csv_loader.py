import pandas as pd

#load the csv file
file_path = "generated_data.csv"
df = pd.read_csv(file_path)

#display the basic information
print("First 5 rows of the datasets")
print(df.head())

print("\nDataset shape (rows, columns)")
print(df.shape)

print("\nColumn name")
print(df.columns)

print("\nData type")
print(df.dtypes)

print("\nStatistical summary")
print(df.describe)

print("\nMissing values check")
print(df.isnull().sum())