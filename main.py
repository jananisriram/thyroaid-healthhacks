# Dosage Docent
# (c) 2022

# old: https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/

# new: xgboost: https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390

from ast import Interactive
from pipes import Template
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

import random

data = pd.read_csv("hypothyroid.csv")

# clean data
cleaned_data = data


res_index = 20



cleaned_data = cleaned_data.query("TSH_measured == 'y'")
cleaned_data = cleaned_data.query("T3_measured == 'y'")
cleaned_data = cleaned_data.query("T4U_measured == 'y'")
cleaned_data = cleaned_data.query("FTI_measured == 'y'")
cleaned_data = cleaned_data.query("Age != '?'")
cleaned_data = cleaned_data.query("Sex != '?'")
cleaned_data = cleaned_data.query("thyroid_surgery != '?'")
cleaned_data = cleaned_data.query("pregnant != '?'")
cleaned_data = cleaned_data.query("sick != '?'")

cleaner_data_lr = cleaned_data.copy()

drop_cols = ["on_antithyroid_medication", "on_thyroxine", "query_on_thyroxine", "query_hypothyroid", "query_hyperthyroid", "TT4", "TT4_measured", "TBG", "TBG_measured"]
cleaned_data.drop(drop_cols, axis=1, inplace=True)

#print(cleaned_data)



drop_cols_2 = ["TSH_measured", "T3_measured", "T4U_measured", "FTI_measured", "TSH", "T3", "T4U", "FTI"]
cleaned_data.drop(drop_cols_2, axis=1, inplace=True)

cleaned_data = cleaned_data.rename(columns={"Unnamed: 0":"disease"})

#print(type(cleaned_data['disease']))



def helper(cleaned_data, datatype):
    # create x and y
    X = cleaned_data.drop("disease", axis=1)
    y = cleaned_data.disease

    #print(type(y[2]))

    for i in range(y.size):
        if y.iloc[i] == "hypothyroid":
            y.iloc[i] = '1'
        else:
            y.iloc[i] = '0'

    y_lbl = pd.DataFrame(y)
    y_lbl["disease"] = y_lbl["disease"].astype(int)
    y = y_lbl["disease"]


    #print(X)

    #print(X.iloc[:, 1])

    for i in range(X.shape[1]):
        curr = X.iloc[:, i]
        #print(curr)
        for j in range(curr.size):
            if curr.iloc[j] == 't' or curr.iloc[j] == 'F':
                curr.iloc[j] = '1'
            elif curr.iloc[j] == 'f' or curr.iloc[j] == 'M':
                curr.iloc[j] = '0'

        X.iloc[:, i] = X.iloc[:, i].astype(datatype)

    return X, y



X, y = helper(cleaned_data, int)


categorical_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

numeric_pipeline = Pipeline(
    steps=[("impute", SimpleImputer(strategy="mean")), 
           ("scale", StandardScaler())]
)

cat_cols = X.select_dtypes(exclude="number").columns
num_cols = X.select_dtypes(include="number").columns

full_processor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, num_cols),
        ("categorical", categorical_pipeline, cat_cols),
    ]
)

# begin xgboost

#xgb_cl = xgb.XGBClassifier()

# preprocessing

#print(X)

X_processed = full_processor.fit_transform(X)
y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
    y.values.reshape(-1, 1)
)

random_int = random.randint(0, 1000000)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed, stratify=y_processed, random_state=random_int
)

_, X_test_copy, _, _ = train_test_split(
    X, y, random_state=random_int
)

person_info = X_test_copy.iloc[res_index]


# initialize classifier
#xgb_cl = xgb.XGBClassifier()
dt_cl = DecisionTreeRegressor(min_samples_leaf=10)

# for i in range(len(y_train)):
#     if y_train[i] == "hypothyroid":
#         y_train[i] = 1
#     else:
#         y_train[i] = 0

# for i in range(len(y_test)):
#     if y_test[i] == "hypothyroid":
#         y_test[i] = 1
#     else:
#         y_test[i] = 0

# fit
#xgb_cl.fit(X_train, y_train)
dt_cl.fit(X_train, y_train)


# predict
#prediction = xgb_cl.predict(X_test)
prediction = dt_cl.predict(X_test)

# tuning

param_grid = {
    "max_depth": [3, 4, 5, 7],
    "learning_rate": [0.1, 0.01, 0.05],
    "gamma": [0, 0.25, 1],
    "reg_lambda": [0, 1, 10],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.5],
}


percents = prediction * 100

person_pred = percents[res_index]

# print("Decision Tree Regression:")

# print(person_info)
# print(person_pred)









###############

print()




# https://www.activestate.com/resources/quick-reads/how-to-run-linear-regressions-in-python-scikit-learn/



predictors = ["Unnamed: 0", "TSH", "T3", "T4U", "FTI"]

#print(cleaner_data_lr)

clean_data_lr = cleaner_data_lr[predictors]

clean_data_lr = clean_data_lr.rename(columns={"Unnamed: 0":"disease"})

X_lr, y_lr = helper(clean_data_lr, float)

model_lr = LinearRegression().fit(X_lr, y_lr)

r_sq_lr = model_lr.score(X_lr, y_lr)

y_pred_lr = model_lr.predict(X_lr)

percent_pred_lr = y_pred_lr * 100

person_data_lr = X_lr.iloc[res_index]
person_pred_lr = percent_pred_lr[res_index]

# print("Linear Regression:")

# print(person_data_lr)
# print(person_pred_lr)



