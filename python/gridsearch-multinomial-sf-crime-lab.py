import numpy as np
import pandas as pd
import patsy

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

crime_csv = './datasets/sf_crime_train.csv'

# A:

# A:

# A:

# A:

# A:

# Example:
# fit model with five folds and lasso regularization
# use Cs=15 to test a grid of 15 distinct parameters
# remember: Cs describes the inverse of regularization strength

# logreg_cv = LogisticRegressionCV(solver='liblinear', 
#                                  Cs=[1,5,10], 
#                                  cv=5, penalty='l1')

# A:

# A:

# A:

# A:

# A:

# A:

# A:

