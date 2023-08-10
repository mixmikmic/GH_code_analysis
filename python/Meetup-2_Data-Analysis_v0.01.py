# General syntax to import specific functions in a library: 
##from (library) import (specific library function)
from pandas import DataFrame, read_csv

import numpy as np
import pandas as pd
import matplotlib as plt
import sys
import datetime as dt

# Enable inline plotting
get_ipython().magic('matplotlib inline')

# Style the plots
plt.pyplot.style.use('ggplot')

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + plt.__version__)

# Loading the CSV file and transforming into a DataFrame
file_location = r'../feedback_survey.csv'
# Read the CSV file and add custom header names
raw_data = pd.read_csv(file_location, header=0, names=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']);

raw_data

raw_data.describe()

plot = raw_data.plot(kind='area', figsize=(20,10), stacked=True, xticks=raw_data.index)
plot.set_title('Distribution')

referral_channels = raw_data['Q5'].groupby(raw_data['Q5']).count()
rf_plot = referral_channels.plot(kind='bar', figsize=(15, 8), rot=0)
rf_plot.set_title('Referral Channels')
rf_plot.set_xlabel('How did you first hear about our event?')



