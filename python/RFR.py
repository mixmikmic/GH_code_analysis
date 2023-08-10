# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.ensemble import RandomForestRegressor as RFR

x_raw = pd.read_csv("data/cleaned_input_data.csv", encoding = "ISO-8859-1")
y_raw = pd.read_csv("data/cleaned_output_data.csv", encoding = "ISO-8859-1")

x_raw.head()

y_raw.head()

y_raw.set_index("Full Name", inplace = True)
x_raw.set_index("Full Name", inplace = True)

y = y_raw[['Valuation Increase']].copy()

y.head()

y.dropna(inplace=True)

y = y.convert_objects(convert_numeric=True)

y.dropna(inplace=True)

y.info()

x_raw.head()

x = x_raw.copy()
x.drop(['Primary Company', 'Crunchbase', 'LinkedIn'], inplace=True, axis=1)

x.head()

from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()
for col in ['Standardized University','Standardized Major', 'Degree Type','Standardized Graduate Institution',"Standardized Graduate Studies",'Graduate Diploma']:
    x[col] = encoder.fit_transform(x[col].fillna('-1'))

x.fillna(-1, inplace=True)

x.astype(int, inplace=True)

df = pd.merge(x, y, left_index=True, right_index=True)

df.info()

modl = RFR(n_estimators = 1000, n_jobs=-1)

modl.fit(df.drop('Valuation Increase', axis=1), df['Valuation Increase'])

print(modl.score(df.drop('Valuation Increase', axis=1), df['Valuation Increase']))

# Look at importance of features for random forest

modl.feature_importances_



