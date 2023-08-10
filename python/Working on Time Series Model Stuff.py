# Load required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bays
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier #stochastic gradient descent
from sklearn.tree import DecisionTreeClassifier

# Garbage Collector
import gc

from pandas import Series
from matplotlib import pyplot
data = pd.read_csv('DataSets/final.csv')
for i in data:
    data_series = data[i]

for i in data:
    print(i)

y = data['Average.retail.price.of.electricity.monthly.California...all.sectors.cents.per.kilowatthour']
x = data.drop(['Average.retail.price.of.electricity.monthly.California...all.sectors.cents.per.kilowatthour'], axis=1)

gdp_Series = data['GDP']
gdp_Series.plot()

corrs = data.corr()
corrs = corrs['Average.retail.price.of.electricity.monthly.California...all.sectors.cents.per.kilowatthour']
corrs.sort_values()

data.columns

new = data[['GenCalifornia...other.renewables.thousand.megawatthours', 'Solar.Consump.TrillionBTU.']]

new['Solar megawatthour'] = new['Solar.Consump.TrillionBTU.'] * 2.93071e-7 * 1000000000000
new

new = new.reindex(index=new.index[::-1])

new.plot(subplots=True)

corrs = data.corr()
corrs = corrs['']
corrs.sort_values()

