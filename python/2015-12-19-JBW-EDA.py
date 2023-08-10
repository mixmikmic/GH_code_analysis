get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.datasets import load_boston

import statsmodels.api as sm
from statsmodels.formula.api import ols

sns.set()

# Load the example Titanic dataset
titanic = sns.load_dataset("titanic")

g = sns.factorplot(x="class", y="survived", data=titanic, 
                   hue="sex", 
                   size=6, kind="bar", palette="muted")

g.set_ylabels("survival probability")



