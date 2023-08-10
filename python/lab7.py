import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().magic('matplotlib inline')

# Load data
proj = pd.read_csv('data/projects.csv')
outcomes = pd.read_csv('data/outcomes.csv')





from sklearn.decomposition import PCA



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.feature_selection import RFE

