

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

# Create a crosstab of default and grade

# Create a crosstab of default and grade - percentage by default type

# Create a crosstab of default and grade - percentage by all type

# Create a crosstab of default and grade - percentage by default type











# Create the transformed income variable



#Plot age, years and default









