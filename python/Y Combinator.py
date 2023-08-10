import numpy as np
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.feature_selection import RFE
from sklearn.svm import SVC

from subprocess import check_output

companies = pd.read_csv('Y Combinator.csv')

companies.head()

print ("Total number of companies funded by Y Combinator since 2005:", companies.shape[0])

companies.info()

companies.isnull()

sns.heatmap(companies.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.figure(figsize=(15,10))
sns.countplot(data=companies,x='year',palette='prism')
plt.title('# of Companies Funded Per Year')
plt.ylabel('Number')

plt.figure(figsize=(15,10))
sns.countplot(data=companies,x='batch',palette='hsv')
plt.tight_layout()

print ("Number of areas that Y Combinator invests in:", len(companies.vertical.unique()))

companies.vertical.unique()

plt.figure(figsize=(15,10))
sns.countplot(data=companies,x='vertical',palette='flag')
plt.title('Type of companies funded')
plt.ylabel('# of companies')
plt.tight_layout()

print ("B2B companies form" ,
       round((companies['vertical']=='B2B').value_counts()[1]/float(len(companies))*100),"% of YC portfolio")

plt.figure(figsize=(15,10))
sns.countplot(data=companies,x='vertical',hue='year')
plt.tight_layout()









