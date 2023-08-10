get_ipython().magic('pylab --no-import-all inline')

from os import path
import sys

import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict

# Load the "autoreload" extension
get_ipython().magic('load_ext autoreload')

# always reload modules marked with "%aimport"
get_ipython().magic('autoreload 1')

# add the 'src' directory as one where we can import modules
src_dir = path.join("..", 'src')
sys.path.append(src_dir)

# import my method from the source code
get_ipython().magic('aimport features.build_features')
get_ipython().magic('aimport models.fit_predict')
get_ipython().magic('aimport visualization.visualize')
from features.build_features import previous_value
from models.fit_predict import cv_predict
from visualization.visualize import modified_bland_altman_plot
from sklearn import metrics
from sklearn.cluster import KMeans

#import utilities

file = path.join("..", "data", "interim", "df.csv")
df = pd.read_csv(file, index_col=0)

df.groupby(['ID'])['AGE'].value_counts().sum()

df['CYCLE_ID'].count().sum()
#UPDATE: Arya's code caluclates an age for the beginning of each cycle
#because there's a different start date for each cycle UGH 

non_null_num = len(df[~df.AGE.isnull()])
print(non_null_num/len(df))

non_null_num_users = len(df[~df.AGE.isnull()].ID.unique())
print(non_null_num)#/len(df.ID.unique()))

df[~df.AGE.isnull()].ID.unique().size

df.ID.unique().size

NUMBER_OF_DAYS = 10
df = df[df.L_PREOVULATION > NUMBER_OF_DAYS]  # No use predicting backward in time.
temp_measurements = ["TEMP" + str(i + 1) for i in range(NUMBER_OF_DAYS)]
features = [*temp_measurements,"past_L_CYCLE", "past_L_PREOVULATION"]

df['past_L_PREOVULATION'] = previous_value('L_PREOVULATION', df)
df['past_L_CYCLE'] = previous_value('L_CYCLE', df)

df.dropna(subset=[
    'past_L_PREOVULATION', 
    'past_L_CYCLE'
], inplace=True)

features += ['past_L_PREOVULATION', 'past_L_CYCLE']

NUMBER_OF_DAYS = 10
df = df[df.L_PREOVULATION > NUMBER_OF_DAYS]  # No use predicting backward in time.
temp_measurements = ["TEMP" + str(i + 1) for i in range(NUMBER_OF_DAYS)]
features += temp_measurements

X = df[features]
y = df.L_PREOVULATION
grouping = df.ID

y_pred = cv_predict(X, y, grouping)

mean_squared_error(y_pred=y_pred, y_true=y)

mean_absolute_error(y_pred=y_pred, y_true=y)

modified_bland_altman_plot(y_pred, y);

