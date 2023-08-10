import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sbrn
import numpy as np
import re
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from patsy import dmatrices

get_ipython().magic('matplotlib inline')

dat = pd.read_csv('water_training.csv', header=0)

labels = pd.read_csv('water_training_labels.csv', header=0)
#join labels to dat on "id" (left outer)

dat=dat.merge(labels, how='left', left_on='id', right_on='id',copy=False)

dat['functional'] = [1 if x=='functional' else 0 for x in dat['status_group']]





df_drop = dat.loc[dat['construction_year']!=0]

y, X = dmatrices('functional ~  construction_year',
                  df_drop, return_type="dataframe")
print X.columns
y = np.ravel(y)
logit = sm.Logit(y, X)
result = logit.fit()

print result.summary()

dat.basin.isnull().sum().sum()

y, X = dmatrices('functional ~  C(basin)',
                  dat, return_type="dataframe")
print X.columns
y = np.ravel(y)
logit = sm.Logit(y, X)
result = logit.fit()

dat.basin.unique()

print result.summary()

dat.quantity.isnull().sum().sum()
#no missing values!

y, X = dmatrices('functional ~  C(quantity)',
                  dat, return_type="dataframe")
print X.columns
y = np.ravel(y)
logit = sm.Logit(y, X)
result = logit.fit()

dat.quantity.unique()

print result.summary()

df_drop = dat.loc[dat['gps_height']!=0]

y, X = dmatrices('functional ~  gps_height',
                  df_drop, return_type="dataframe")
print X.columns
y = np.ravel(y)
logit = sm.Logit(y, X)
result = logit.fit()

print result.summary()

df_drop = dat.loc[dat['population']!=0]

y, X = dmatrices('functional ~  population',
                  df_drop, return_type="dataframe")
print X.columns
y = np.ravel(y)
logit = sm.Logit(y, X)
result = logit.fit()

print result.summary()

dat.district_code.isnull().sum().sum()

df_drop = dat.loc[dat['district_code']!=0]

y, X = dmatrices('functional ~  C(district_code)',
                  dat, return_type="dataframe")
print X.columns
y = np.ravel(y)
logit = sm.Logit(y, X)
result = logit.fit()

df_drop.district_code.unique()

#check this... what's the reference group?
print result.summary()

dat.scheme_management.isnull().sum().sum()

df_drop = dat[pd.notnull(dat['scheme_management'])]

df_drop.scheme_management.isnull().sum().sum()

df_drop['scheme_management'] = df_drop['scheme_management'].str.replace('SWC','a_SWC')

y, X = dmatrices('functional ~  C(scheme_management)',
                  df_drop, return_type="dataframe")
print X.columns
y = np.ravel(y)
logit = sm.Logit(y, X)
result = logit.fit()

df_drop.scheme_management.unique()

print result.summary()

#dat.scheme_management.isnull().sum().sum()
dat.extraction_type_class.isnull().sum().sum()

dat['extraction_type_class'] = dat['extraction_type_class'].str.replace('other','a_other')

y, X = dmatrices('functional ~  C(extraction_type_class)',
                  dat, return_type="dataframe")
print X.columns
y = np.ravel(y)
logit = sm.Logit(y, X)
result = logit.fit()

dat.extraction_type_class.unique()

print result.summary()

dat.water_quality.isnull().sum().sum()

dat['water_quality'] = dat['water_quality'].str.replace('unknown','a_unknown')

y, X = dmatrices('functional ~  C(water_quality)',
                  dat, return_type="dataframe")
print X.columns
y = np.ravel(y)
logit = sm.Logit(y, X)
result = logit.fit()

dat.water_quality.unique()

print result.summary()

dat.quantity.isnull().sum().sum()

y, X = dmatrices('functional ~  C(quantity)',
                  dat, return_type="dataframe")
print X.columns
y = np.ravel(y)
logit = sm.Logit(y, X)
result = logit.fit()

dat.quantity.unique()

print result.summary()

dat.source.isnull().sum().sum()

dat['source'] = dat['source'].str.replace('lake','a_lake')

y, X = dmatrices('functional ~  C(source)',
                  dat, return_type="dataframe")
print X.columns
y = np.ravel(y)
logit = sm.Logit(y, X)
result = logit.fit()

dat.source.unique()

print result.summary()

dat.waterpoint_type.isnull().sum().sum()

dat['waterpoint_type'] = dat['waterpoint_type'].str.replace('other','a_other')

y, X = dmatrices('functional ~  C(waterpoint_type)',
                  dat, return_type="dataframe")
print X.columns
y = np.ravel(y)
logit = sm.Logit(y, X)
result = logit.fit()

dat.waterpoint_type.unique()

print result.summary()

dat.subvillage.describe()



