# Plot our graphs in ipython, not in a new window.
get_ipython().magic('matplotlib inline')

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cols = ['Lead_(ppb)', 'Land_Value', 'Year_Built', 'SL_Type', 'SL_Type2', 'HomeSEV', 'Land_Value', 'Building_Storeys']

train_df = pd.read_csv('./data/flint_train.csv', usecols=cols)

train_corr = train_df.corr()
plt.imshow(train_corr - np.eye(7), interpolation='none', cmap='gray')

counts = plt.hist(train_df[train_df['Lead_(ppb)'] > 0]['Lead_(ppb)'], bins=20, log=True)

plt.figure()
plt.scatter(train_df[train_df['Lead_(ppb)'] > 15]['Lead_(ppb)'].values, 
            train_df[train_df['Lead_(ppb)'] > 15]['Land_Value'].values)
plt.axis([0, 100, 0, 200000])

plt.scatter(train_df[np.logical_and(train_df['Year_Built'] > 1900, train_df['Lead_(ppb)'] > 15)]['Year_Built'], 
            train_df[np.logical_and(train_df['Year_Built'] > 1900, train_df['Lead_(ppb)'] > 15)]['Lead_(ppb)'])



