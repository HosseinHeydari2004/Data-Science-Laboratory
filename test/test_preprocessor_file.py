import pandas as pd
from Core.preprocessor import EDA

data = pd.read_csv("diabetes.csv")

EDA.information_data(data=data)
