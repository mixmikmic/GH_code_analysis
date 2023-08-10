# Load relevant libraries.

get_ipython().magic('pylab inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.api import abline_plot
import patsy
import seaborn as sns
sns.set(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, rc=None)
import sklearn as skl

# Spam database.

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
              "spambase/spambase.data")

spam = pd.read_csv("spam.csv")
print(spam.info())
print(spam['v57'].describe())

spam['index'] = range(len(spam))

# In Python, the % indicates modulus.
spam['index'] = spam['index'] % 3
spam['test'] = pd.get_dummies(spam['index'] == 1)[1]

# We don't need the index, so drop it.
del spam['index']

# Now we can create new train and test dataframes.
# Note the format of these lines code.
# It basically resolves as create spamtest as a subset of spam when test is 1.
# Otherwise, it is train.
spamtrain = spam[spam['test'] == 0]
spamtest = spam[spam['test'] == 1]

# Confirm data has been split properly.
print(spamtrain['v57'].count())
print(spamtest['v57'].count())
print(spam['v57'].count())

seed(12345)
spam['index'] = np.random.uniform(low = 0, high = 1, size = len(spam))
spam['test'] = pd.get_dummies(spam['index'] <= 0.3333)[1]

# We don't need the index, so drop it.
del spam['index']

# Now we can create new train and test dataframes.
# Note the format of these command lines.
# It basically resolves as create spamtest as a subset of spam when test is 1.
# Otherwise, it is train.
spamtrain = spam[spam['test'] == 0]
spamtest = spam[spam['test'] == 1]

# Confirm data has been split properly.
print(spamtrain['v57'].count())
print(spamtest['v57'].count())
print(spam['v57'].count())

spamtrain = spam.sample(frac = 0.67, random_state = 1066)
spamtest = spam.drop(spamtrain.index)

# Confirm data has been split properly.
print(spamtrain['v57'].count())
print(spamtest['v57'].count())
print(spam['v57'].count())

