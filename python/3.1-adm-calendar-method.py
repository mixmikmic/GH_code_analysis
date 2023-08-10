get_ipython().magic('pylab --no-import-all inline')

from os import path
import sys
import pandas as pd
import seaborn as sns

# Load the "autoreload" extension
get_ipython().magic('load_ext autoreload')

# always reload modules marked with "%aimport"
get_ipython().magic('autoreload 1')

# add the 'src' directory as one where we can import modules
src_dir = path.join("..", 'src')
sys.path.append(src_dir)

# import my method from the source code
get_ipython().magic('aimport features.build_features')
get_ipython().magic('aimport visualization.visualize')
from features.build_features import previous_value
from visualization.visualize import modified_bland_altman_plot

file = path.join("..", "data", "processed", "df.csv")
df = pd.read_csv(file, index_col=0)

calendar_guess = previous_value('L_PREOVULATION', df)

sns.residplot(x=df.L_PREOVULATION, y=calendar_guess, order=0);

# Load the "autoreload" extension
get_ipython().magic('load_ext autoreload')

# always reload modules marked with "%aimport"
get_ipython().magic('autoreload 1')

import os
import sys

# add the 'src' directory as one where we can import modules
src_dir = os.path.join("..", 'src')
sys.path.append(src_dir)

# import my method from the source code
get_ipython().magic('aimport visualization.visualize')
from visualization.visualize import modified_bland_altman_plot

modified_bland_altman_plot(calendar_guess, df.L_PREOVULATION);

from sklearn.metrics import mean_squared_error, mean_absolute_error
data = df
data['GUESS'] = calendar_guess
data.dropna(subset=['GUESS'], inplace=True)
mean_squared_error(data.L_PREOVULATION, data.GUESS), mean_absolute_error(data.L_PREOVULATION, data.GUESS)

