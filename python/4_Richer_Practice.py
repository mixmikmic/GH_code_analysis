from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn import datasets
# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target



