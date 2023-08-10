get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv('data_stocks.csv')
# Drop date variable
data = data.drop(['DATE'], 1)
# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]
print("n: ", n)
print("p: ", p)
# Make data a numpy array
data = data.values

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

print(data_train)



