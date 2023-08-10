# some imports
from IPython.display import Image
from pprint import pprint
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # just for the styling
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
get_ipython().magic('matplotlib inline')

# we have a few categorical variables, so we specify the dtype
categoricals = ['ChestPain', 'Thal', 'AHD']
df = pd.read_csv('data/heart.csv.gz', index_col=0,
                 dtype={name: 'category' for name in categoricals})

target_name = 'AHD'  # this is our target variable
feature_names = [name for name in df if name != target_name]

# let's see what we have
df.head(3)

# and the dtypes
df.dtypes

# do we have missing values?
df.isnull().head() # .sum(0)

# let's get rid off them
df = df.dropna()

# construct a mapping that assign each categorical value it's code
categorical_codes = {}
for categorical in categoricals:
    categories = df[categorical].cat.categories
    categorical_codes[categorical] = {'names': categories, 'codes': list(range(len(categories)))}

pprint(categorical_codes)

# for later use we save the class names
class_names = categorical_codes[target_name]['names']
class_names

# the classifier needs numbers, so we replace the categorical values by their code
# copy all non-categorical columns
df_new = df[[name for name in df if name not in categoricals]]
# from the categorical ones we keep the code-representation
for categorical in categoricals:
    df_new[categorical] = df[categorical].cat.codes

# now it looks like this
df_new.head(3)

df_new.dtypes

# our predictor variables / feature matrix / ...
X = df_new[feature_names].values
# our target variable
y = df_new[target_name].values

# our classifier - that's it! ;)
clf = DecisionTreeClassifier()  # max_depth=2
clf = clf.fit(X, y)

clf

# since it is a tree, it would be nice to plot it
# forturnately, we can export a dot file which we can then render as png using graphviz like so

dot_file_name = 'heart.dot'
png_file_name = 'heart.png'

with open(dot_file_name, 'w') as dot_data_file:
    dot_data_file = export_graphviz(clf,
                                    out_file=dot_data_file,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True)   
subprocess.check_call(['dot', '-Tpng', dot_file_name, '-o', png_file_name])

# let's see...
Image(filename=png_file_name)

# do not trust all these splits! we are overfitting...



