import pandas as pd
import os
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

df = pd.read_excel("C:/Users/jangn/CODE/Sundog_DataScience/DataScience/DataScience-Python3/data_sets/KingsCountyHouseData/kc_house_data.xls")

df.head()

X = df[['bedrooms','bathrooms', 'sqft_living','sqft_lot','floors','waterfront','views','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']]
y = df['price']

X1 = sm.add_constant(X)
est = sm.OLS(y, X1).fit()

est.summary()

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

X = df.loc[:,('bedrooms','bathrooms', 'sqft_living','sqft_lot','floors','waterfront','views','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15')]
y = df['price']

X[['bedrooms','bathrooms', 'sqft_living','sqft_lot','floors','waterfront','views','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']] = scale.fit_transform(X[['bedrooms','bathrooms', 'sqft_living','sqft_lot','floors','waterfront','views','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].as_matrix())

est = sm.OLS(y, X).fit()

est.summary()

df_sqft = df[['sqft_living', 'sqft_above']]
df_sqft.corr() 

X = df[[ 'bedrooms','sqft_living','waterfront', 'views', 'grade','yr_built','lat']]
y = df['price']

X1 = sm.add_constant(X)
est = sm.OLS(y, X1).fit()
est.summary()

df_sub = df[['price','bedrooms','sqft_living','waterfront', 'views', 'grade','yr_built','lat']]
df_sub.corr() 

corrmat = df_sub.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=1, square=True);
plt.show()

fig_size = plt.rcParams["figure.figsize"] 
fig_size[0]=16.0
fig_size[1]=8.0
df_sub.hist(bins=100)
plt.show()


for index, row in df.iterrows():

    Constant = (est.params.const)
    Bedrooms = row[3]*(est.params.bedrooms)
    Sqft_living = row[5]*(est.params.sqft_living)
    Waterfront = row[8]*(est.params.waterfront)
    Views = row[9]*(est.params.views)
    Grade = row[11]*(est.params.grade)
    Year_built = row[14]*(est.params.yr_built)
    Latitude = row[17]*(est.params.lat)
    
    price_estimates = ([index, row[1], Constant+Bedrooms+Sqft_living+Waterfront+Views+Grade+Year_built+Latitude][2])
     


df_estimates = pd.read_excel("C:/Users/jangn/CODE/Sundog_DataScience/DataScience/DataScience-Python3/data_sets/KingsCountyHouseData/kc_house_data_estimates.xls")

f, ax = plt.subplots(figsize=(17, 6))
sns.regplot(x="price", y="price_estimates", data=df_estimates, ax=ax)
plt.show()

fig_size[0]=16.0
fig_size[1]=2.0
df_estimates.plot(bins=2000, kind='hist', alpha=0.7)

plt.title('Actual house prices vs. price estimates from the model - all prices')
axes=plt.axes()
axes.set_xlim([0,7700000])
#axes.set_ylim([0,20])
plt.show()

fig_size[0]=16.0
fig_size[1]=8.0
df_estimates.plot(bins=2000, kind='hist', alpha=0.7)

plt.title('Actual house prices vs. price estimates from the model - with prices only up to $1.5mio ')
axes=plt.axes()
axes.set_xlim([0,1500000])
plt.show()

