import numpy as np
import pandas as pd

data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data)
print(type(data))

print(data.values)
print(type(data.values))

print(data.index)
print(type(data.index))  # the row names are known as the index

print(data[1])
print(type(data[1]))  # when you select only one value, it simplifies the object

print(data[1:3])
print(type(data[1:3]))  # selecting multiple values returns a series

print(data[np.array([1,0,2])])  # fancy indexing using a numpy array

# specifying the index values
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
print(data)

data[1]  # subset with index position

data["a"]  # subset with index names

data["a":"c"] # using names includes the last value

data[0:2]  # slicing behavior is unchanged

# creating a series from a python dictionary
# remember, dictionary construction uses curly braces {}
samp_dict = {'Archie': 71,
             'Betty': 66,
             'Veronica': 62,
             'Jughead': 72,
             'Cheryl': 66}
samp_series = pd.Series(samp_dict)
samp_series # the series gets alphabetized by its index

print(samp_series.index)

print(type(samp_dict))
print(type(samp_series))

actor_dict = {'Archie': "KJ",
              'Jughead': "Cole",
              'Betty': "Lili",
              'Veronica': "Camila",
              'Cheryl': "Madelaine"}  # note that the dictionary order is not same here
actor = pd.Series(actor_dict)  # still get alphabetized by index
print(actor)

# we create a dataframe by providing a dictionary of series objects
riverdale = pd.DataFrame({'height': samp_series,
                       'actor': actor})  

print(riverdale)

print(type(riverdale))  # this is a DataFrame object

data = [{'a': 0, 'b': 0}, {'a': 1, 'b': 2}, {'a': 2, 'b': 5}]  # data is a list of dictionaries
data

print(pd.DataFrame(data, index = [1,2,3]))

data = [{'a': 0, 'b': 0}, {'a': 1, 'b': 2}, {'a': 2, 'c': 5}]  # data is a list of dictionaries
data

print(pd.DataFrame(data))

data = [{'a': 0, 'b': 0}, {'a': 1, 'b': 2}, {'a': 2, 'c': "5"}]  # data is a list of dictionaries
data

print(pd.DataFrame(data))

data = np.random.randint(10, size = 10).reshape([5,2])
print(data)

print(pd.DataFrame(data, columns = ["x","y"], index = ['a','b','c','d','e']))

print(riverdale)

print(riverdale.keys())

riverdale['actor']  # extracting the column

riverdale.actor

riverdale.actor[1]

riverdale.actor['Jughead']

print(riverdale.T)  # prints a copy of the transpose

print(riverdale.loc['Jughead']) # subset based on location to get a row
print(type(riverdale.loc['Jughead']))
print(type(riverdale.loc['Jughead'].values))  # the values are of mixed type but is still a numpy array. 
# this is possible because it is a structured numpy array. (covered in "Python for Data Science" chapter 2)

print(riverdale.loc[:,'height']) # subset based on location to get a column
print(type(riverdale.loc[:,'height']))  #the object is a pandas series
print(type(riverdale.loc[:,'height'].values))

riverdale.loc['Archie','height']  # you can provide a pair of 'coordinates' to get a particular value

riverdale.iloc[3,] # subset based on index location

riverdale.iloc[0, 1] # pair of coordinates

