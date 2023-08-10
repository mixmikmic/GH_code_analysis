# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the data into DataFrames
df_train = pd.read_csv('train_users_2.csv')
df_test = pd.read_csv('test_users.csv')
sessions = pd.read_csv('sessions.csv')
usergrp = pd.read_csv('age_gender_bkts.csv')
countries = pd.read_csv('countries.csv')

usergrp = pd.read_csv('age_gender_bkts.csv')

#Convert 100+ into a bin.
usergrp['age_bucket'] = usergrp['age_bucket'].apply(lambda x: '100-104' if x == '100+' else x)
#Define mean_age feature
usergrp['mean_age'] = usergrp['age_bucket'].apply(lambda x: (int(x.split('-')[0]) + int(x.split('-')[1]))/2)
usergrp = usergrp.drop('age_bucket', axis=1)

countries = pd.read_csv('countries.csv')

sessions = pd.read_csv('sessions.csv')

print("We have", df_train.shape[0], "users in the training set and", 
      df_test.shape[0], "in the test set.")
print("In total we have", df_train.shape[0] + df_test.shape[0], "users.")

users = pd.concat((df_train, df_test), axis=0, ignore_index=True)

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
destination_percentage = df_train.country_destination.value_counts() / df_train.shape[0] * 100
destination_percentage.plot(kind='bar',color='#3498DB')
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
sns.despine()

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
destination_percentage = df_train.language.value_counts() / df_train.shape[0] * 100
destination_percentage.plot(kind='bar',color='#3498DB')
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
sns.despine()

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
gender_percentage = df_train.gender.value_counts() / df_train.shape[0] * 100
gender_percentage.plot(kind='bar',color='#D35400')
plt.xlabel('Gender of users')
plt.ylabel('Percentage')
sns.despine()

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
device_percentage = df_train.first_device_type.value_counts() / df_train.shape[0] * 100
device_percentage.plot(kind='bar',color='#196F3D')
plt.xlabel('Device used by user')
plt.ylabel('Percentage')
sns.despine()

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.distplot(users.age.dropna())
plt.xlabel('PDF of Age')
sns.despine()

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
users['age']=users['age'].apply(lambda x : 36 if x>100 else x)
sns.distplot(users.age.dropna())
plt.xlabel('PDF of Age')
sns.despine()

df_train['date_account_created_new'] = pd.to_datetime(df_train['date_account_created'])
df_train['date_first_active_new'] = pd.to_datetime((df_train.timestamp_first_active // 1000000), format='%Y%m%d')
df_train['date_account_created_day'] = df_train.date_account_created_new.dt.weekday_name
df_train['date_account_created_month'] = df_train.date_account_created_new.dt.month
df_train['date_account_created_year'] = df_train.date_account_created_new.dt.year
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
data_without_NDF = df_train[df_train['country_destination']!='US']
data_without_NDF1= data_without_NDF[data_without_NDF['country_destination']!='NDF']
sns.countplot(x='date_account_created_day',data=df_train)
plt.xlabel('Days in a week')
plt.ylabel('Number of users')
sns.despine()

# order from Monday - Friday, Saturday - Sunday

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
data_without_NDF = df_train[df_train['country_destination']!='US']
data_without_NDF1= data_without_NDF[data_without_NDF['country_destination']!='NDF']
df_train['booked'] = df_train.country_destination.apply(lambda x:1 if x!='NDF' else 0 )
destination_percentage = df_train.groupby(['date_account_created_year','date_account_created_month']).booked.sum() / df_train.shape[0] * 100
destination_percentage.plot(kind='bar',color="#F4D03F")
plt.xlabel('Year wise - each month Travel count')
plt.ylabel('Percentage')
sns.despine()

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(18.7, 12.27)
df_train.date_account_created_new.value_counts().plot(kind='line', linewidth=1.2)
plt.xlabel('Date account created - line plot ')
sns.despine()

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
data_without_NDF = df_train[df_train['country_destination']!='US']
data_without_NDF1= data_without_NDF[data_without_NDF['country_destination']!='NDF']
sns.countplot(x='country_destination', hue='signup_app',data=data_without_NDF1)
plt.xlabel('Destination Country based on signup app')
plt.ylabel('Number of users')
sns.despine()

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
data_without_NDF = df_train[df_train['country_destination']!='US']
data_without_NDF1= data_without_NDF[data_without_NDF['country_destination']!='NDF']
sns.countplot(x='country_destination', hue='signup_method',data=data_without_NDF1)
plt.xlabel('Destination Country based on signup method ( removed NDF,US )')
plt.ylabel('Number of Users')
sns.despine()

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
affiliate_provider_percentage = df_train.affiliate_provider.value_counts() / df_train.shape[0] * 100
affiliate_provider_percentage.plot(kind='bar',color='#CB4335')
plt.xlabel('Percentage of users based on affiliate providers ')
plt.ylabel('Percentage')
sns.despine()



