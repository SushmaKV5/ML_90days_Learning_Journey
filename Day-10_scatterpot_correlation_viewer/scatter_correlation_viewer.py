import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_dataset.csv")

print("Generating Scatter Plot Correlation Viewer....")

#Select numeric columns
numeric_cols = df.select_dtypes(include=['int64','float64']).columns

#Generate scatter plots betweern pairs of columns
for i in range(1,len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        x = numeric_cols[i]
        y = numeric_cols[j]

        plt.figure()
        plt.scatter(df[x], df[y])

        plt.title(f"{x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

        #Correlation value
        corr = df[x].corr(df[y])
        print(f"Correlation between {x} and {y}: {corr:.2f}")
