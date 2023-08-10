from sml import execute

query = 'READ "../data/auto-mpg.csv" (separator = "\s+", header = None) AND REPLACE ("?", "mode")'

execute(query, verbose=True)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve, validation_curve

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize']=(12,12)
sns.set()

#Names of all of the columns
names = [
       'mpg'
    ,  'cylinders'
    ,  'displacement'
    ,  'horsepower'
    ,  'weight'
    ,  'acceleration'
    ,  'model_year'
    ,  'origin'
    ,  'car_name'
]

#Import dataset
data = pd.read_csv('../data/auto-mpg.csv', sep = '\s+', header = None, names = names)

data.head()

# Remove NaNs
data_clean=data.applymap(lambda x: np.nan if x == '?' else x).dropna()



