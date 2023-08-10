from sklearn import datasets
import pandas as pd
import numpy as np

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target



