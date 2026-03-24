import pandas as pd
from sklearn.model_selection import train_test_split

#Load the dataset
df = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")

#select the features and target
X = df[['Pclass', 'Age', 'Fare']]   #features
y = df['Survived']                  #target

#Handle missing values
X = X.fillna(X.mean)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Output
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nFirst 5 rows of X_train:")
print(X_train.head())

print("\nFirst 5 rows y_train:")
print(y_train.head())