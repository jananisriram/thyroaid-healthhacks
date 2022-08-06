# Dosage Docent
# (c) 2022

# old: https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/

# new: xgboost: https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390

import pandas as pd

data = pd.read_csv("hypothyroid.csv")

# clean data
drop_cols = []
data.drop(drop_cols, axis=1, inplace=True)

