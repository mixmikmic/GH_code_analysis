# Allow rendering of Matplotlib plots directly on Jupyter Notebooks.
get_ipython().run_line_magic('matplotlib', 'inline')

# Import all dependencies required for the problem.
from __future__ import print_function
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

# Beautify the traditional looking plots.
sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Set a Seed for random number generation for reproducible results
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load the titanic dataset using Pandas library 
df = pd.read_excel('../data/titanic_dataset.xlsx').dropna(subset=['Age'])

# Split the dataset into dependent features (passenger details used for prediction)
# and target features (prediction if the passenger survived)
x = df.loc[:,:'Embarked']
y = df['Survived']

# Preview the Titanic Dataset
x.head()

# Our Fancy Smart Classifier tries to predict if a passenger survived or not based on his / her details. 
def smart_classifier(row):
    """The function predicts if the passenger survived titanic (1 -> survived / 0 -> Died)"""
    return 1

from sklearn.metrics import accuracy_score
def print_accuracy():
    predictions = []
    for index, record in x.iterrows():
        predictions.append(smart_classifier(record))
    print("Accuracy: {:.2f}".format(accuracy_score(y, predictions) * 100.0))
print_accuracy()

