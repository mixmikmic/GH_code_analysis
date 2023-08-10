get_ipython().magic('pylab --no-import-all inline')

from os import path
import sys

import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error

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

file = path.join("..", "data", "interim", "df.csv")
df = pd.read_csv(file, index_col=0)

features = []

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

features

WINDOW_SIZE = 10

TEMP_COLUMNS = ["TEMP" + chr(i + ord("A")) for i in range(WINDOW_SIZE)]
DISTANCE_COLUMN = ["DISTANCE"]
OTHER_COLUMNS = ["COUNTDOWN", "past_L_PREOVULATION", "past_L_CYCLE", "ID"]
STARTING_COLUMNS = TEMP_COLUMNS + DISTANCE_COLUMN + OTHER_COLUMNS
df2 = pd.DataFrame(columns=STARTING_COLUMNS)

for i in range(99 - WINDOW_SIZE):
    df['COUNTDOWN'] = df.L_PREOVULATION - (i + WINDOW_SIZE + 1)
    df['DISTANCE'] = i
    columns = ["TEMP" + str(i + j + 1) for j in range(WINDOW_SIZE)]
    df2.columns = columns + DISTANCE_COLUMN + OTHER_COLUMNS
    df2 = df2.append(df[df.COUNTDOWN > 0][columns + DISTANCE_COLUMN + OTHER_COLUMNS], ignore_index=True)
df2.columns = STARTING_COLUMNS
df2.dropna(subset=TEMP_COLUMNS, thresh=WINDOW_SIZE - 1, inplace=True)

X = df2.drop(labels=['COUNTDOWN', 'ID'], axis=1)
y = df2.COUNTDOWN
grouping = df2.ID

X.columns

from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.ensemble import RandomForestRegressor

mlpr = MLPRegressor(random_state=1337, hidden_layer_sizes=(50, 20))
imp = Imputer(strategy='mean')
scl = StandardScaler()
pipeline = Pipeline([('imp', imp), ('scl', scl), ('mlp', mlpr)])

cv = GroupKFold(n_splits=10)

y_pred = cross_val_predict(pipeline, X, y,
                           cv=cv, groups=grouping,
                           verbose=3, n_jobs=-1)

mean_squared_error(y_pred=y_pred, y_true=y)

mean_absolute_error(y_pred=y_pred, y_true=y)

modified_bland_altman_plot(y_pred, y);



df.L_PERIOD.median()



