import pandas as pd

#loading the dataset
file_path = "generated_data.csv"
df = pd.read_csv(file_path)

print("Original dataset shape", df.shape)

#check for missing values
print("\nMissing value before cleaning:")
print(df.isnull().sum())

#Fill missing value with mean(numerical columns)
df_filled = df.fillna(df.mean(numeric_only=True))

#remove duplicate rows
df_cleaned = df_filled.drop_duplicates()

print("\nMissing value after cleaning:")
print(df_cleaned.isnull().sum())

print("\nCleaned dataset shape", df_cleaned.shape)

#Save cleaned dataset
df_cleaned.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned dataset saved as cleaned_dataset.csv")