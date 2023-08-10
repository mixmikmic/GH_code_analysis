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

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

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

X = df[features]
y = df.L_PREOVULATION
grouping = df.ID

from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

reg = GradientBoostingRegressor(random_state=1337, n_estimators=200)
imp = Imputer(strategy='mean')
scl = StandardScaler()
pipeline = Pipeline([('imp', imp), ('scl', scl), ('reg', reg)])

cv = GroupKFold(n_splits=10)

y_pred = cross_val_predict(pipeline, X, y,
                           cv=cv, groups=grouping,
                           verbose=True, n_jobs=-1)

mean_squared_error(y_pred=y_pred, y_true=y)

mean_absolute_error(y_pred=y_pred, y_true=y)

modified_bland_altman_plot(y_pred, y);

df.L_PERIOD.median()



