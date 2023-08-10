from IPython.core.display import HTML
def css_styling():
    styles = open("styles/custom.css", "r").read()
    return HTML(styles)
css_styling()

# Import modules you might use
import numpy as np

# Some data, in a list
my_data = [12, 5, 17, 8, 9, 11, 21]

# Function for calulating the mean of some data
def mean(data):

    # Initialize sum to zero
    sum_x = 0.0

    # Loop over data
    for x in data:

        # Add to sum
        sum_x += x 
    
    # Divide by number of elements in list, and return
    return sum_x / len(data)

get_ipython().magic('timeit mean(np.arange(1000000))')

nmean = np.mean
get_ipython().magic('timeit nmean(np.arange(1000000))')

sum_x = 0

# Loop over data
for x in my_data:
    
    # Add to sum
    sum_x += x 

print(sum_x)

mean(my_data)

# Function for calulating the mean of some data
def mean(data):

    # Call sum, then divide by the numner of elements
    return sum(data)/len(data)

# Function for calculating variance of data
def var(data):

    # Get mean of data from function above
    x_bar = mean(data)

    # Do sum of squares in one line
    sum_squares = sum([(x - x_bar)**2 for x in data])

    # Divide by n-1 and return
    return sum_squares/(len(data)-1)

x = (45, 95, 100, 47, 92, 43)
y = (65, 73, 10, 82, 6, 23)
z = (56, 33, 110, 56, 86, 88) 
datasets = (x,y,z)

datasets

means = []
for d in datasets:
    means.append(mean(d))

means

list(map(mean, datasets))

np.mean(datasets, axis=1)

42              # Integer
0.002243        # Floating-point
5.0J            # Imaginary
'foo'
"bar"           # Several string types
s = """Multi-line
string"""

type(True)

not False

x = None
print(x)

15/4

(14 - 5) * 4

(34,90,56) # Tuple with three elements

(15,) # Tuple with one element

(12, 'foobar') # Mixed tuple

foo = (5,7,2,8,2,-1,0,4)
foo[0]

foo[2:5]

foo[:-2]

foo[1::2]

a = (1,2,3)
a[0] = 6

tuple('foobar')

# List with five elements
[90, 43.7, 56, 1, -4]

# Tuple with one element
[100]    

# Empty list
[]       

bar = [5,8,4,2,7,9,4,1]
bar[3] = -5
bar

bar * 3

[0]*10

(3,)*10

bar.extend(foo) # Adds foo to the end of bar (in-place)
bar

bar.append(5) # Appends 5 to the end of bar
bar

bar.insert(0, 4) # Inserts 4 at index 0
bar

bar.remove(7) # Removes the first occurrence of 7
bar

bar.remove(100) # Oops! Doesn’t exist

bar.pop(4) # Removes and returns indexed item

bar.reverse() # Reverses bar in place
bar

bar.sort() # Sorts bar in place
bar

bar.count(7) # Counts occurrences of 7 in bar

bar.index(7) # Returns index of first 7 in bar

my_dict = {'a':16, 'b':(4,5), 'foo':'''(noun) a term used as a universal substitute 
           for something real, especially when discussing technological ideas and 
           problems'''}
my_dict

my_dict['b']

len(my_dict)

# Checks to see if ‘a’ is in my_dict
'a' in my_dict

# Returns a copy of the dictionary
my_dict.copy() 

# Returns key/value pairs as list
my_dict.items() 

# Returns list of keys
my_dict.keys() 

# Returns list of values
my_dict.values() 

my_dict['c']

my_dict.get('c')

my_dict.get('c', -1)

my_dict.popitem() 

# Empties dictionary
my_dict.clear()
my_dict

my_set = {4, 5, 5, 7, 8}
my_set

empty_set = set()
empty_set

empty_set.add(-5)
another_set = empty_set
another_set

my_set | another_set

my_set & another_set

my_set - {4}

bar

set(bar)

