import pandas as pd
from Core.preprocessor import EDA

data = pd.read_csv("diabetes.csv")

print(EDA.find_high_col_missing_values(data=data).keys())

