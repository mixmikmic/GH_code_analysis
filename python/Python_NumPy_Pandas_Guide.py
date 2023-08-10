a = 3
print('a:', a)

b = 2
print('b:', b)

c = a + b
print('c:', c)

print('Multiplication for a and b:', a * b)

print('Normal (float) division for a and b:', a / b)

print('Floor (integer) division for a and b:', a // b)

print('a to the power of b:', a ** b)

a = 'I am a string'
print('a:', a)

b = 'I am a different string'
print('b:', b)

c = a + b
print('String concatenation:', c)

print('Looping through characters in a string:')
for i in a:
    print(i)

d = c[1:4]
print('Making sub-strings by slicing a string from index 1 to index 4:', d)

a = ['a', 'b', 'c']
print('A whole list could be printed out all at once:', a)

b = [1, 2, 3]
print('Looping through elements in a list:')
for i in b:
    print(i)

print('Accessing an element at a specific index:', a[0], b[1])

c = a + b
print('List concatenation', c)

d = c[1:3]
print('Slicing a list from index 1 to index 3:', d)

a = {'a': 1, 'b': 2}
print('A whole dictionary could be printed out all at once:', a)

a['c'] = 3
print('Assign a value to a new key to add items to a dictionary:', a)

print('Looping through keys in a dictionary and getting the value at each key:')
for key in a:
    print('key:', key)
    print('value:', a[key])

a = 1
b = 2
c = 4

if a == 1:
    print('a is equal to 1.')
else:
    print('a is not equal to 1.')

if a == 2:
    print('a is equal to 2.')
elif c == b * 2:
    print('c is twice b')
else:
    print('No condition is true.')

a = [1, 'a', 'A String']

for item in a:
    print(item)

def sum_of(a, b):
    return a + b

x = 1
y = 2
z = sum_of(x, y)

print('The returned value could be directly printed out:', sum_of(x, y))
print('Or it could be stored in a varible:', z)

import numpy as np

my_list = [1, 2, 3]
print(my_list)

my_array = np.array(my_list)
print(my_array)

array1 = np.arange(6)
print('NumPy array from 0 up to 6:', array1)

array2 = np.zeros((3,))
print('NumPy array of zeros with the specified shape:', array2)

array3 = np.ones((5,))
print('NumPy array of ones with the specified shape:', array3)

my_array = np.array([1, 3, 5, 2])

print('Original array:', my_array)
print('Elements in array multiplied by 3:', my_array * 3)
print('Elements in array squared:', my_array ** 2)

new_array = np.array([1, 2, 3, 4])
print('Another array:', new_array)
print('Adding two arrays:', my_array + new_array)
print('Dividing one array by another:', my_array / new_array)

my_array = np.array([[1, 3, 5], [2, 4, 6]])
print(my_array)
print('Number of rows and number of columns:', my_array.shape)

my_array = my_array.reshape((3,2))
print(my_array)
print('Number of rows and number of columns:', my_array.shape)

my_array = np.array([1, 2, 3, 4, 5])
print('Element at index 1:', my_array[1])
print('Element at index 3:', my_array[3])

sub_array = my_array[1:4]
print('Sub array from index 1 to index 4:', sub_array)

my_array = np.array([[1, 2, 3, 4, 5], [2, 3, 6, 7, 3], [-1, 3, -4, 5, 3]])
print('Original array:')
print(my_array)

print('Row at index 1 (second row):', my_array[1,:])
print('Column at index 0 (first column):', my_array[:,0])

print('Elements between rows with index 0 and index 2, columns with index 1 and index 3:')
print(my_array[0:2, 1:3])

import pandas as pd

my_df = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 3, 4], 'z': [9, 8, 7]})
print('The whole dataset:')
print(my_df)

print('Accessing specific columns:')
print(my_df['x'])
print(my_df['z'])

print('Getting row at index 1:')
print(my_df.loc[1])

print('Getting the rows from index 0 to index 1 (inclusively):')
print(my_df.loc[0:1])

my_df = my_df.set_index(['x'])
print(my_df)

# looping through the columns
for column in my_df.columns:
    print(column)
    print(my_df[column])

# looping through the indices (i.e. through the rows)
for index in my_df.index:
    print(my_df.loc[index])

print('Original DataFrame:')
print(my_df)

# adding a column with the name 't'
my_df['t'] = [20, 3, 19]
print('DataFrame after adding t:')
print(my_df)

my_df = pd.DataFrame({'x': [None, 1, 3, 3], 'y': [4, 2, 9, 8], 'z': [3, 4, None, 5]})
print(my_df)

# dropping the rows with NaN values
filled_df = my_df.dropna(axis=0)
print(filled_df)

# dropping the columns with NaN values
filled_df = my_df.dropna(axis=1)
print(filled_df)

# filling the missing values with the mean, mode, and median of corresponding columns
filled_df = my_df.fillna(my_df.mean())
print(filled_df)
filled_df = my_df.fillna(my_df.mode().loc[0])
print(filled_df)
filled_df = my_df.fillna(my_df.median())
print(filled_df)

filled_df = my_df.interpolate()
print(filled_df)



