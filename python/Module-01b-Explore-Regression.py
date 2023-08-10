#Load the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Default Variables
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (16,9)
plt.rcParams['font.size'] = 18
plt.style.use('fivethirtyeight')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#Load the dataset
df = pd.read_csv("data/loan_data_clean.csv")

df.head()

# create the target "amount * default'

# Create histogram for the target variable

# Explore other variables



# Create a crosstab of grade and ownership







# Transform the income variable

# Plot the transformed variable













