import pandas as pd
from Core.preprocessor import MissingValue

data = pd.read_csv("diabetes.csv")

pre = MissingValue()
print(pre.list_columns(data))
