import pandas as pd
# Initializes an empty series by default
s = pd.Series()
print(s)

import numpy as np
# Load some data into a series
data = np.array(['a','b','c','d'])
pd.Series(data)

# We can also load some data from a dictionary too
data = {'a' : 0., 'b' : 1., 'c' : 2.}
# Any values we don't have data for become NaN
pd.Series(data,index=['b','c','d','a'])

s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

#retrieve the first element
s[0]

s[:3]

s['a']

s[:'c']

s['f']

import pandas as pd
# Create an Empty DF
empty_df = pd.DataFrame()
print(empty_df)

# Create Dataframe from lists
data = [['Alex',10],['Bob',12],['Clarke',13]]
# Note the dtype=float casts the age to floats
name_df = pd.DataFrame(data,columns=['Name','Age'], dtype=float)
print("List DF")
print(name_df)

import numpy as np
# Most commonly you'll be making dataframes from numpy arrays
data = np.random.uniform(size=(100, 3))
random_df = pd.DataFrame(data, columns=['a', 'b', 'c'])
# Since this is a big dataframe, we don't print the whole thing out
# Lets just print the last 5 values
random_df.tail()

random_df['d'] = np.ones((100,))
random_df.tail()

random_df['e'] = random_df['c'] + random_df['d']
random_df.tail()

random_df.pop('e')
random_df.tail()

# Add a new column that is the average columns 'a', 'b', and 'c'.
# YOUR CODE HERE

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
     'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
df

df.loc['b']

df.iloc[1]

df['c':]

df[2:]

df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])

df = df.append(df2)
df

df.loc[0]

df.drop(0)

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])}

#Create a DataFrame
df = pd.DataFrame(d)
df.count()

# Find the Standard Deviation of the ages in d
# YOUR CODE HERE

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
df.groupby(['Team','Year']).groups

left = pd.DataFrame({
         'id':[1,2,3,4,5],
         'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
         'subject_id':['sub1','sub2','sub4','sub6','sub5']})
right = pd.DataFrame(
         {'id':[1,2,3,4,5],
         'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
         'subject_id':['sub2','sub4','sub3','sub6','sub5']})
pd.merge(left,right,on='id')

# Merge left and right by subject but do not throw
# out any subjects that are not shared between the two.
# HINT: Think about the 'how' of the merge

# YOUR CODE HERE

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
# We'll write to a csv file
# Index says we will not write the index column
df.to_csv("temp.csv", index=False)

# Now we can read the csv file
pd.read_csv("temp.csv")

