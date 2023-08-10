import pandas as pd

url = 'http://bit.ly/kaggletrain'
train = pd.read_csv(url)
train.head(3)

# create new column
train['Sex_num'] = train.Sex.map({'female':0, 'male':1})

# let's compared Sex and Sex_num columns
# here we can see we map male to 1 and female to 0
train.loc[0:4, ['Sex', 'Sex_num']]

# say we want to calculate length of string in each string in "Name" column

# create new column
# we are applying Python's len function
train['Name_length'] = train.Name.apply(len)

# the apply() method applies the function to each element
train.loc[0:4, ['Name', 'Name_length']]

import numpy as np

# say we look at the "Fare" column and we want to round it up
# we will use numpy's ceil function to round up the numbers
train['Fare_ceil'] = train.Fare.apply(np.ceil)

train.loc[0:4, ['Fare', 'Fare_ceil']]

# let's extract last name of each person

# we will use a str method
# now the series is a list of strings
# each cell has 2 strings in a list as you can see below
train.Name.str.split(',').head()

# we just want the first string from the list
# we create a function to retrieve
def get_element(my_list, position):
    return my_list[position]

# use our created get_element function
# we pass position=0
train.Name.str.split(',').apply(get_element, position=0).head()

# instead of above, we can use a lambda function
# input x (the list in this case)
# output x[0] (the first string of the list in this case)
train.Name.str.split(',').apply(lambda x: x[0]).head()

# getting the second string
train.Name.str.split(',').apply(lambda x: x[1]).head()

url = 'http://bit.ly/drinksbycountry'
drinks = pd.read_csv(url)
drinks.head()

drinks.loc[:, 'beer_servings':'wine_servings'].head()

# you want apply() method to travel axis=0 (downwards, column) 
# apply Python's max() function
drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=0)

# you want apply() method to travel axis=1 (right, row) 
# apply Python's max() function
drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=1)

# finding which column is the maximum's category name
drinks.loc[:, 'beer_servings':'wine_servings'].apply(np.argmax, axis=1)

drinks.loc[:, 'beer_servings': 'wine_servings'].applymap(float).head()

# overwrite existing table

drinks.loc[:, 'beer_servings': 'wine_servings'] = drinks.loc[:, 'beer_servings': 'wine_servings'].applymap(float)
drinks.head()

