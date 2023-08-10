import pandas as pd

# read the pva97nk dataset
df = pd.read_csv('pva97nk.csv')

# show all columns information
df.info()

df['DemAge'].describe()

df['DemAge'].unique()

df['DemAge'].value_counts()

df.groupby(['TargetB'])['DemAge'].mean()

df.groupby(['TargetB'])['DemGender'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns

# dropna is used because 'DemAge' has missing values
dg = sns.distplot(df['DemAge'].dropna())
plt.show()

dg = sns.countplot(data=df, x='DemGender')
plt.show()

ax = sns.boxplot(x="TargetB", y="DemMedHomeValue", data=df)
plt.show()

# change DemCluster from interval/integer to nominal/str
df['DemCluster'] = df['DemCluster'].astype(str)

df['DemHomeOwner'].value_counts()

# change DemHomeOwner into binary 0/1 variable
dem_home_owner_map = {'U':0, 'H': 1}
df['DemHomeOwner'] = df['DemHomeOwner'].map(dem_home_owner_map)

df['DemMedIncome'].value_counts()

# denote errorneous values in DemMidIncome
mask = df['DemMedIncome'] < 1

import numpy as np
df.loc[mask, 'DemMedIncome'] = np.nan

# impute missing values in DemAge with its mean
df['DemAge'].fillna(df['DemAge'].mean(), inplace=True)

# impute med income using mean
df['DemMedIncome'].fillna(df['DemMedIncome'].mean(), inplace=True)

# impute gift avg card 36 using mean
df['GiftAvgCard36'].fillna(df['GiftAvgCard36'].mean(), inplace=True)

# drop ID and the unused target variable
df.drop(['ID', 'TargetD'], axis=1, inplace=True)

# for gender, before one hot encoding. .head() is used to display first 5 records.
df['DemGender'].head(5)

# after one hot encoding
demo_df = pd.get_dummies(df['DemGender'])
demo_df.head(5)

# one hot encoding all categorical variables
# all numerical variables are automatically excluded
# number of columns after the conversion should explode
print("Before:", len(df.columns))

# one hot encoding
df = pd.get_dummies(df)

print("After:", len(df.columns))

# dm_tools.py
import numpy as np
import pandas as pd

def data_prep():
    # read the pva97nk dataset
    df = pd.read_csv('pva97nk.csv')
    
    # change DemCluster from interval/integer to nominal/str
    df['DemCluster'] = df['DemCluster'].astype(str)
    
    # change DemHomeOwner into binary 0/1 variable
    dem_home_owner_map = {'U':0, 'H': 1}
    df['DemHomeOwner'] = df['DemHomeOwner'].map(dem_home_owner_map)
    
    # denote errorneous values in DemMidIncome
    mask = df['DemMedIncome'] < 1
    df.loc[mask, 'DemMedIncome'] = np.nan
    
    # impute missing values in DemAge with its mean
    df['DemAge'].fillna(df['DemAge'].mean(), inplace=True)

    # impute med income using mean
    df['DemMedIncome'].fillna(df['DemMedIncome'].mean(), inplace=True)

    # impute gift avg card 36 using mean
    df['GiftAvgCard36'].fillna(df['GiftAvgCard36'].mean(), inplace=True)
    
    # drop ID and the unused target variable
    df.drop(['ID', 'TargetD'], axis=1, inplace=True)
    
    df = pd.get_dummies(df)
    
    return df

# can just import it like this
# from dm_tools import data_prep

