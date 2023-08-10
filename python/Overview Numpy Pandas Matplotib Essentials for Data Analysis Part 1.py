# How to import these

import numpy

import pandas

import matplotlib

# very simple
# Now How to use them
# Where I Get help to use them

# What background i should have to learn or master these
# Em, hold on hold on

# We answer these questions but be patience 

# Lets Get Overview of these three in this class

# I am sure its gonna be exciting

#load the library and check its version, just to make sure we aren't using an older version
import numpy as np
np.__version__

d = np.array( [ [ 1., 0., 0.] ,
                [ 0.,1., 2.]]   )

print(d)
print(d.shape)


a = np.zeros( (2,2) )   # Create an array of all zeros
# Read about zeros funtion by just click shift + tab
print("a of zeros: \n",a)


b = np.ones( (2,2) )
print("b of ones: \n",b)


c = np.full((2,2), 9.)  # Create a constant array with each element as 9
print("c of full: \n",c)


d = np.eye(3)         # Create a 2x2 identity matrix
print("d of identity \n",d)              # Prints "[[ 1.  0.]

e = np.random.random((2,2))  # Create an array filled with random values
print("e of type: ",type(e))
print("e of random elements: \n",e)         

f = np.arange(15)
print("f dtype: ",f.dtype)
print("f: \n",f)

g = f.reshape(3,5) # Shape it in 3 rows and 5 cols: Rank 3
print("Reshaping f to g: \n",g)

get_ipython().run_line_magic('pinfo', 'numpy.ndarray')

get_ipython().run_line_magic('pinfo', 'np')

# Full demo of it

# Case 1


import numpy as np 

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print("a: \n", a)

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[0:2, 1:3]
print("b: \n",b )

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
# print(a[0, 1])   # Prints "77"

# Case 2

import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"


import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])     # Prints "[3 4 5 6]"

import numpy as np

# Please Imagine doing those operations using loops or so :D 

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print("Addition: \n", np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print("Substraction: \n" ,x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print("Multiplication: \n",x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print("Division: \n", x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print("Square Root: \n",np.sqrt(x))

a= np.array([2,2])  
b= np.array([3,5])
print("Elementwise: \n", a*b )

print("Dot product: ", np.dot(a,b))
print('___________________________')

# Case 3:


import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
print("x shape: ", x.shape)

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))

# lets summarize
# other methods
import numpy as np

x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"

# Lets play more on numpy
# Transpose

x = np.array([[1,23,3],[3,4,5]])

print(x)
print ( x.shape ) 
print ( x.T ) # Transpose
print ( x.T.shape ) 

# Not a better way of doing it

import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)

# What is better way of doing it
# Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v. Consider this version, using broadcasting:

import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"

#load library - pd is just an alias. I used pd because it's short and literally abbreviates pandas.
#You can use any name as an alias. 
import pandas as pd

# lets learn from quick docs
get_ipython().run_line_magic('pinfo', 'pandas')

# First clear the idea as we are dealing with tabular form many times so

#create a data frame 
# We already know dictionaries yup so
# dictionary is used here where keys get converted to column names and values to row values.


# pd.DataFrame?

# 1
data = pd.DataFrame( 
                    {
                    'Country': ['Pakistan','India','Australia','South Africa'],
                    'Rank':[1,2,3,4]
                    }
                )


data


# 2
# temp = pd.DataFrame(np.array([1,2]), columns=['ABC'])
# print(temp)



# We can do a quick analysis of any data set using:
data.describe()

# enter shift+tab on describe to get quick overview
# Generate various summary statistics, excluding NaN values.

# Concise summary of a DataFrame.
data.info()

# Now lets create a dataframe

#Let's create another data frame about some heights and weights.
data = pd.DataFrame(
                        {
                        'names':['Adam', 'Kabir', 'James', 'Hammad','uttoman', 'Saleem', 'Durga'],
                         'heights':[5  , 6      ,  7     ,  6      , 8       ,  5      ,  4],
                         'weights':[55 , 60     ,  78    ,  69     , 88      ,  89     ,  38],
                        }
                    )
data

# 1 
# Let's sort the data frame by ounces - inplace = True will make changes to the data
data.sort_values(by=['heights'],ascending=True)

# 2

data.sort_values(by=['heights','weights'],ascending=[True,True])

#create another data with duplicated rows

data = pd.DataFrame(
                        {
                            'name': ['Adam']*3 + ['James']*4 ,
                            'rollno'   : [3,2,1,3,3,4,4]
                        }
                    )

data

#sort values 
data.sort_values(by='rollno')

# See above whoo
# functions as duplicate rows removed

data.drop_duplicates()

# Done haha
# See docs as mentioned and tell how to remove based on particular column

# Here it is
data.drop_duplicates(subset='rollno')

# voila ! 
# isn't :D 

data = pd.DataFrame(
                    {
                        'food':   ['beef','camel','chicken', 'Bridie' ],
                        'ounces': [ 4    , 3     , 12    , 6          ]
                    }
                )


data

# Now, we want to create a new variable which indicates the type of animal which acts as the source of the food. 
# To do that, first we'll create a dictionary to map the food to the animals. Then, we'll use map function to map the dictionary's values to the keys. Let's see how is it done.

# Let create a dic
meat_to_animal = {
'bacon': 'pig',
'pulled pork': 'pig',
'pastrami': 'cow',
'beef': 'cow',
'honey ham': 'pig',
'nova lox': 'salmon',
'camel':'camel',
'chicken':'chicken',
'Bridie': 'goat'
}




#create a new variable
data['animal'] = data['food'].map(meat_to_animal)
data

data['animal_test'] = data['food'].map(meat_to_animal)
data

#### How to remove a column from dataframe
data.drop('animal_test',axis='columns')
data

#Series function from pandas are used to create arrays
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data

# Replace values given in 'to_replace' with 'value'
data.replace(-999, np.nan)

#We can also replace multiple values at once.
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data.replace([-999,-1000],np.nan)
data

data = pd.DataFrame(np.arange(12).reshape((3, 4)),index=['Ohio', 'Colorado', 'New York'],columns=['one', 'two', 'three', 'four'])
data

# Got it?
# you should it is pretty simple

# But this when we created own values
# what about already created variable

#Using rename function
data.rename(index = {'Ohio':'SanF'}, columns={'one':'one_p','two':'two_p'})
data

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]

# We'll divide the ages into bins such as 18-25, 26-35,36-60 and 60 and above.

bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats

# Understand the output - '(' means the value is included in the bin, '[' means the value is excluded

#pandas library intrinsically assigns an encoding to categorical variables.
print(type(cats))
print(cats.codes) # We already seen a feature before recall it

#Let's check how many observations fall under each bin
pd.value_counts(cats)

# Not only this 
# we can pass a unique name to each label.

bin_names = ['Youth', 'YoungAdult', 'MiddleAge', 'Senior']
new_cats = pd.cut(ages, bins,labels=bin_names)

pd.value_counts(new_cats)

#we can also calculate their cumulative sum
pd.value_counts(new_cats).cumsum()

# Now can you create a simple dataframe

df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : [2,2,2,1,1],
                   'data2' : [2,1,1,1,1]})
df

#calculate the mean of data1 column by key1
grouped = df['data1'].groupby(df['key1'])
print( grouped.mean() )

print(grouped)

dates = pd.date_range('20130101',periods=6)

# np.random.randn(6,4) Hope you know what it is going to do
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['A','B','C','D'] )
df

# 1 get first n rows from the data frame
df[:3]

#slice based on date range
df['20130101':'20130104']

#slicing based on column names
df.loc[:,['A','B']]      # All rows and  'A' and 'B' cols

#slicing based on both row index labels and column names
df.loc['20130102':'20130103',['A','B']]

print(df)

#slicing based on index of columns
df.iloc[:3,2:] #returns 4th row (index is 3rd)

#Similarly, we can do Boolean indexing based on column values as well. This helps in filtering a data set based on a pre-defined condition.

df[df.A > 1]

#we can copy the data set
df2 = df.copy()
df2['E']=['one', 'one','two','three','four','three']
df2

#select rows based on column values
df2[df2['E'].isin(['two','four'])]

#list all columns where A is greater than C
df.loc[df['A'] > df['C']]

df.query('A < B | C > A')

#create a data frame
data = pd.DataFrame({'group': ['a', 'a', 'a', 'b','b', 'b', 'c', 'c','c'],
                 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data

# See how simple it is 
# data.pivot_table?

#calculate means of each group
data.pivot_table(values='ounces',index='group',aggfunc=np.mean)

# look above the frame 

data.pivot_table(values='ounces',index='group',aggfunc='count')

