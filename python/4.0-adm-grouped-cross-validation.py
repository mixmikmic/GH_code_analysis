get_ipython().magic('pylab --no-import-all inline')

from os import path
import sys

import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the "autoreload" extension
get_ipython().magic('load_ext autoreload')

# always reload modules marked with "%aimport"
get_ipython().magic('autoreload 1')

# add the 'src' directory as one where we can import modules
src_dir = path.join("..", 'src')
sys.path.append(src_dir)

# import my method from the source code
get_ipython().magic('aimport models.fit_predict')
get_ipython().magic('aimport visualization.visualize')
from models.fit_predict import cv_predict
from visualization.visualize import modified_bland_altman_plot

file = path.join("..", "data", "interim", "df.csv")
df_orig = pd.read_csv(file, index_col=0)

features = []

NUMBER_OF_DAYS = 10
df = df_orig[df_orig.L_PREOVULATION > NUMBER_OF_DAYS]  # No use predicting backward in time.
temp_measurements = ["TEMP" + str(i + 1) for i in range(NUMBER_OF_DAYS)]
features += temp_measurements

X = df[features]
y = df.L_PREOVULATION
grouping = df.ID

y_pred = cv_predict(X, y, grouping)

mean_squared_error(y_pred=y_pred, y_true=y)

mean_absolute_error(y_pred=y_pred, y_true=y)

modified_bland_altman_plot(y_pred, y);

df.L_PERIOD.median()



