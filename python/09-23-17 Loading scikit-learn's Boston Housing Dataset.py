# Load libraries
from sklearn import datasets
import matplotlib.pyplot as plt

# load the dataset

boston = datasets.load_boston()

# Create feature matrix
X = boston.data

# Create target vector
y = boston.target

# View the first observation's feature values
X[0]

# Display each feature value of the first observatisn of floats
['{:f}'.format(x) for x in X[0]]

