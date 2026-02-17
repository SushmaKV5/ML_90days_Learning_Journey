import numpy as np
import pandas as pd

#Generate a random data
data = {
    "age": np.random.randint(18,60,100),
    "salary":np.random.randint(20000, 100000, 100),
    "experience":np.random.randint(1,20,100)
}

df = pd.DataFrame(data)

df.to_csv("generated_data.csv")

print("Random data is generated and saved to generated_data.csv")