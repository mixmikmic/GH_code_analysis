from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

housing_owners = pd.read_csv('fema_data/chi_housing_assistance_owners.csv')
housing_owners['zipCode'] = housing_owners['zipCode'].fillna(0).astype(int).astype(str)
housing_renters = pd.read_csv('fema_data/chi_housing_assistance_renters.csv')
housing_renters['zipCode'] = housing_renters['zipCode'].fillna(0).astype(int).astype(str)
housing_renters.head()

housing_owners.shape

housing_renters.shape

housing_owners.dtypes

housing_renters.dtypes

owner_approved_zip = housing_owners.groupby(['zipCode'])['approvedForFemaAssistance'].sum()
owner_approved_df = pd.DataFrame(owner_approved_zip).reset_index()
owner_approved_df = owner_approved_df.dropna()
owner_approved_df = owner_approved_df.sort_values(by='approvedForFemaAssistance',ascending=False)
owner_approved_df.head()

renter_approved_zip = housing_renters.groupby(['zipCode'])['approvedForFemaAssistance'].sum()
renter_approved_df = pd.DataFrame(renter_approved_zip).reset_index()
renter_approved_df = renter_approved_df.dropna()
renter_approved_df = renter_approved_df.sort_values(by='approvedForFemaAssistance',ascending=False)
renter_approved_df.head()

both_approved_df = owner_approved_df.merge(renter_approved_df, on='zipCode')
both_approved_df['approvedForFemaAssistance'] = both_approved_df['approvedForFemaAssistance_x'] + both_approved_df['approvedForFemaAssistance_y']
both_approved_df = both_approved_df[['zipCode', 'approvedForFemaAssistance']]
both_approved_df = both_approved_df.sort_values(by='approvedForFemaAssistance', ascending=False)
both_approved_df.head()

flood_zip_df = pd.read_csv('311_data/wib_calls_311_zip.csv')
flood_zip_df = flood_zip_df[flood_zip_df.columns.values[1:]]
flood_zip_stack = pd.DataFrame(flood_zip_df.stack()).reset_index()
flood_zip_stack = flood_zip_stack.rename(columns={'level_0':'Created Date','level_1':'Zip Code',0:'Count Calls'})
flood_zip_sum = flood_zip_stack.groupby(['Zip Code'])['Count Calls'].sum()
flood_zip_sum = flood_zip_sum.reset_index()
flood_zip_sum = flood_zip_sum.sort_values(by='Count Calls',ascending=False)
flood_zip_sum.head()

flood_zips = flood_zip_sum['Zip Code'].unique()
fema_approved_zip_df = both_approved_df.loc[both_approved_df['zipCode'].isin(flood_zips)].copy()
fema_approved_zip_df.shape

fig, axs = plt.subplots(1,2)
plt.rcParams["figure.figsize"] = [15, 5]
fema_approved_zip_df[:20].plot(title='FEMA Data', ax=axs[0], kind='bar',x='zipCode',y='approvedForFemaAssistance')
flood_zip_sum[:20].plot(title='FOIA Data', ax=axs[1], kind='bar',x='Zip Code',y='Count Calls')

fema_flood_zip = pd.DataFrame()
fema_flood_zip[['Zip Code', 'FEMA Approved']] = fema_approved_zip_df.copy()
fema_flood_zip = fema_flood_zip.merge(flood_zip_sum, on='Zip Code')
fema_flood_zip.plot(title='FEMA and FOIA Data', x='Zip Code', y=['FEMA Approved', 'Count Calls'], kind='bar')

wib_07_df = pd.read_csv('311_data/wib_calls_311_zip.csv')
wib_07_df['Created Date'] = pd.to_datetime(wib_07_df['Created Date'])
wib_07_df = wib_07_df.set_index(wib_07_df['Created Date'])
wib_07_df = wib_07_df['2007-01-01':][wib_07_df.columns.values[2:]]
wib_07_df.head()

wib_07_stack = pd.DataFrame(wib_07_df.stack()).reset_index()
wib_07_stack = wib_07_stack.rename(columns={'level_0':'Created Date','level_1':'Zip Code',0:'Count Calls'})
wib_07_sum = wib_07_stack.groupby(['Zip Code'])['Count Calls'].sum()
wib_07_sum = wib_07_sum.reset_index()
wib_07_sum = wib_07_sum.sort_values(by='Count Calls',ascending=False)
wib_07_sum.head()

wib_07_zips = wib_07_sum['Zip Code'].unique()
fema_approved_wib_df = both_approved_df.loc[both_approved_df['zipCode'].isin(wib_07_zips)].copy()

fig, axs = plt.subplots(1,2)
plt.rcParams["figure.figsize"] = [15, 5]
fema_approved_wib_df[:20].plot(title='FEMA Data', ax=axs[0], kind='bar',x='zipCode',y='approvedForFemaAssistance')
wib_07_sum[:20].plot(title='FOIA Data', ax=axs[1], kind='bar',x='Zip Code',y='Count Calls')

fema_wib_zip = pd.DataFrame()
fema_wib_zip[['Zip Code', 'FEMA Approved']] = fema_approved_wib_df.copy()
fema_wib_zip = fema_wib_zip.merge(wib_07_sum, on='Zip Code')
fema_wib_zip.plot(title='FEMA and FOIA Data since 2007', x='Zip Code', y=['FEMA Approved', 'Count Calls'], kind='bar')

fema_wib_zip.plot(title='FEMA and FOIA Data Scatter Plot', x='Count Calls', y=['FEMA Approved'], kind='scatter')

fema_flood_zip.plot(title='FEMA and FOIA Data Scatter Plot', x='Count Calls', y=['FEMA Approved'], kind='scatter')

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
fema_lr = fema_flood_zip[["FEMA Approved", "Count Calls"]]

x_data = [[x] for x in fema_lr["Count Calls"].values]
y_data = [[y] for y in fema_lr["FEMA Approved"].values]
lm.fit(x_data, y_data)
y_lr = lm.predict(x_data)

plt.plot(x_data, y_lr, 'g')
plt.scatter(fema_lr['Count Calls'].values, fema_lr['FEMA Approved'].values)
plt.title("FEMA Approved and FOIA Calls Linear Regression")
plt.show()

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lm.score(x_data, y_data))



