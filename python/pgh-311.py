# use the %ls magic to list the files in the current directory.
get_ipython().magic('ls')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sms
get_ipython().magic('matplotlib inline')

three11s = pd.read_csv("data/pgh-311.csv", parse_dates=['CREATED_ON'])

three11s.dtypes

three11s.head()

three11s.loc[0]

# Plot the number of 311 requests per month

month_counts = three11s.groupby(three11s.CREATED_ON.dt.month)

y = month_counts.size()
x = month_counts.CREATED_ON.first()

axes = pd.Series(y.values, index=x).plot(figsize=(15,5))

plt.ylim(0)
plt.xlabel('Month')
plt.ylabel('Complaint')

grouped_by_type = three11s.groupby(three11s.REQUEST_TYPE)

size = grouped_by_type.size()
size
#len(size)
#size[size > 200]

codebook = pd.read_csv('data/codebook.csv')
codebook.head()

merged_data = pd.merge(three11s, 
                       codebook[['Category', 'Issue']], 
                       how='left',
                       left_on="REQUEST_TYPE", 
                       right_on="Issue")

merged_data.head()

grouped_by_type = merged_data.groupby(merged_data.Category)
size = grouped_by_type.size()
size

size.plot(kind='barh', figsize=(8,6))

merged_data.groupby(merged_data.NEIGHBORHOOD).size().sort_values(inplace=False,
                                                         ascending=False)

merged_data.groupby(merged_data.NEIGHBORHOOD).size().sort_values(inplace=False,
                                                         ascending=True).plot(kind="barh", figsize=(5,20))

# create a function that generates a chart of requests per neighborhood
def issues_by_neighborhood(neighborhood):
    """Generates a plot of issue categories by neighborhood"""
    grouped_by_type = merged_data[merged_data['NEIGHBORHOOD'] == neighborhood].groupby(merged_data.Category)
    size = grouped_by_type.size()
    size.plot(kind='barh', figsize=(8,6))

issues_by_neighborhood('Greenfield')

issues_by_neighborhood('Brookline')

issues_by_neighborhood('Garfield')

from ipywidgets import interact

@interact(hood=sorted(list(pd.Series(three11s.NEIGHBORHOOD.unique()).dropna())))
def issues_by_neighborhood(hood):
    """Generates a plot of issue categories by neighborhood"""
    grouped_by_type = merged_data[merged_data['NEIGHBORHOOD'] == hood].groupby(merged_data.Category)
    size = grouped_by_type.size()
    size.plot(kind='barh',figsize=(8,6))



