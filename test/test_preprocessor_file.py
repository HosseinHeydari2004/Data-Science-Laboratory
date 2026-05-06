import pandas as pd
from Core.preprocessor import EDA

data = pd.read_csv("diabetes.csv")
print(data.isna().sum())
print(EDA.remove_missing_values(data=data, axis=1))
