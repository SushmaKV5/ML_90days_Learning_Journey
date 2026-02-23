import pandas as pd

#Ask user for file name
file_path = input("Enter the csv file name as .csv:")

try:
    #Load the dataset
    df = pd.read_csv(file_path)

    print("\n===DATA SUMMARY REPORT===")


    #Shape of the dataset
    print("Dataset shape(rows, columns):", df.shape)

    #Columns name
    print("\nColumn name:")
    print(list(df.columns))

    #Datatype
    print("\nData types:")
    print(df.dtypes)

    #Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    #Statistical summary
    print("\nStatistical summary:")
    print(df.describe())

    #Unique value count
    print("\nUnique value count:")
    print(df.nunique())
    
except FileNotFoundError:
    print("Error: File not found error. Please check the file name and try again!")