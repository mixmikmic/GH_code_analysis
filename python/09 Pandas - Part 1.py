# Series
import numpy as np
import pandas as pd
myArray = np.array([2,3,4])
row_names = ['p','q','r']
mySeries = pd.Series(myArray,index=row_names)
print mySeries
print mySeries[0]
print mySeries['p']

# Dataframes
myArray = np.array([[2,3,4],[5,6,7]])
row_names = ['p','q']
col_names = ['One','Two','Three']
myDataFrame = pd.DataFrame(myArray,index = row_names,columns = col_names)
print myDataFrame
print 'Method 1 :'
print 'One column = \n%s'%myDataFrame['One']
print 'Method 2 :'
print 'One column = \n%s'%myDataFrame.One

# Let's load data from a csv
df = pd.read_csv("../data/diabetes.csv")
df.info()

# Examine data
df.head()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
# Histogram
bins=range(0,100,10)

plt.hist(df["Age"].values, bins, alpha=0.5, label='age')
plt.show()

plt.hist(df["BMI"].values, bins, alpha=0.5, label='BMI')
plt.show()

plt.hist(df["Age"].values, bins, alpha=0.5, label='age')
plt.hist(df["BMI"].values, bins, alpha=0.5, label='BMI')
plt.show()

from numpy.random import normal
gaussian_numbers = normal(size=5000)
plt.hist(gaussian_numbers, bins=np.linspace(-5.0, 5.0, num=20)) # Set bin bounds
plt.show()

# Let's start with an example on the AGE feature
# I create a new array for easier manipulation
arr_age = df["Age"].values
arr_age[:10]

mean_age = np.mean(arr_age)
std_age = np.std(arr_age)
print 'Age Mean: {} Std:{}'.format(mean_age, std_age)

# So to compute the standardized array, I write :
arr_age_new = (arr_age - mean_age)/std_age
arr_age_new[:10]

# I can now apply the same idea to a pandas dataframe
# using some built in pandas functions :
df_new = (df - df.mean()) / df.std()
df_new.head()

df.head()

# Histogram
bins=np.linspace(-5.0, 5.0, num=20)

plt.hist(df_new["Age"].values, bins, alpha=0.5, label='age')
plt.hist(df_new["BMI"].values, bins, alpha=0.5, label='BMI')
plt.show()



