import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import patsy

plt.style.use('fivethirtyeight')

from ipywidgets import *
from IPython.display import display

from sklearn.svm import SVC

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

# A:
file_path = './datasets/breast_cancer_wisconsin/breast_cancer.csv'
df = pd.read_csv(file_path)

df.head()

f = 'Class ~ ' + ' + '.join([col for col in df.columns if col != 'Class'])
y, X = patsy.dmatrices(f, data=df, return_type='dataframe')

y = y.applymap(lambda x: 0 if x == 2 else 1)

1 - y.mean()

from sklearn.metrics import classification_report, confusion_matrix

# A:
baseline_acc = 1 - y.mean()

# A:

# A:

# A:

