get_ipython().magic('pylab --no-import-all inline')

from os import path
import sys

import pandas as pd
import seaborn as sns

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
from visualization.visualize import modified_bland_altman_plot, residual_plot

import keras; print(keras.__version__)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPool1D
from keras.wrappers.scikit_learn import KerasRegressor

file = path.join("..", "data", "interim", "df.csv")
df = pd.read_csv(file, index_col=0)

features = []

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
from sklearn.model_selection import GroupKFold

def mlp_model():
    model = Sequential()
    model.add(Dense(20, input_dim=10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

reg = KerasRegressor(build_fn=mlp_model, 
                     epochs=20, batch_size=5, verbose=1)
imp = Imputer(strategy='mean')
scl = StandardScaler()
pipeline = Pipeline([('imp', imp), ('scl', scl), ('reg', reg)])

cv = GroupKFold(n_splits=5)

y_pred = cross_val_predict(pipeline, X, y,
                           cv=cv, groups=grouping,
                           verbose=True, n_jobs=-1)

mean_squared_error(y_pred=y_pred, y_true=y)

mean_absolute_error(y_pred=y_pred, y_true=y)

modified_bland_altman_plot(y_pred, y);

residual_plot(y_pred, y);

y_pred

df.L_PERIOD.median()

