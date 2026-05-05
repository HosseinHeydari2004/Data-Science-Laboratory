import pandas as pd
from Core.preprocessor import EDA

data = pd.read_csv("diabetes.csv")

print(data.memory_usage(deep=True))
