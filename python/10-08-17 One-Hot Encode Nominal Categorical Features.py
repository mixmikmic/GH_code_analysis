# Load Libraries
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd

# Create NumPy array
X = np.array([['Texas'], 
              ['California'], 
              ['Texas'], 
              ['Delaware'], 
              ['Texas']])

# Create LabelBinzarizer object
one_hot = LabelBinarizer()

# One-hot encode data
one_hot.fit_transform(X)

# View classes
one_hot.classes_

# Dummy feature
pd.get_dummies(x[:,0])

