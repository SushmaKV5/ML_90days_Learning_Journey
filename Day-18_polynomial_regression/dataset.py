import pandas as pd
import numpy as np

np.random.seed(42)

experience = np.arange(1, 21)
salary = 20000 + (experience * 5000) + np.random.randint(-5000, 5000, size=20)

df = pd.DataFrame({
    "Experience": experience,
    "Salary": salary
})

df.to_csv("dataset.csv", index=False)

print("Dataset created!")
