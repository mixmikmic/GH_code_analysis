#import library 
import pandas as pd
import numpy as np

#load the data 
df1 = pd.read_csv('titanic passenger list.csv') 

df1.info()

df1.describe()

df1.describe(include=['O']) # 'O' for Objects

df1.isnull().sum()

np.mean(df1.age)

# age - we know there are some missing, let's dig deeper
df1['age'].value_counts()

# mostly younger, 20s to 30s
df1['age'].unique()

df1[df1['age'].isnull()].head() # look at some

df1[df1['age'] < 1]

get_ipython().run_line_magic('matplotlib', 'inline')
df1['age'].hist()

sum(df1['age'].isnull())
# 263 passengers have no age recorded.

# Look into titles, e.g. 'Mrs' implies married (implies not child)
def name_extract(word):
     return word.split(',')[1].split('.')[0].strip()
    
# because names are in this format:
# Allison, Master. Hudson Trevor
# we can split on ','
# then '.'

temp = pd.DataFrame({'Title':df1['name'].apply(name_extract)}) # testing, apply the method to the data
# check unique values
temp['Title'].unique()

# a couple of strange ones but most of the standard titles are there
temp['Title'].value_counts()

# did we miss any?
sum(temp['Title'].value_counts())

df2 = df1 # copy then insert new column
df2['Title'] = df1['name'].apply(name_extract)
df2.head() # title at far right

# just check (again) we got most of them
df2[df2['Title'].isnull()]

df2[df2.age.isnull()].Title.value_counts()

df2[df2['Title'] == "Dr"]

df2[df2['Title'] == "Dr"].mean()

df2[(df2['Title'] == "Dr") & (df2['sex'] == "male")].mean()

df2[df2['Title'] == "Master"] # how many?

#there are a lot, 61, use describe()
df2[df2['Title'] == "Master"].describe() # min age is 0.33 (4 months?), max is 14.5, mean is 5.5

df2[df2['Title'] == "Master"].mean()

# this seems too easy, is it right? 
df2["age"].fillna(df2.groupby("Title")["age"].transform("mean"), inplace=True)
df2.age.describe()

df2.groupby("Title")["age"].transform("mean")

