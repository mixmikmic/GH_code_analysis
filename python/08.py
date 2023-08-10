record = ('Marco', 'UK', True, 123)  # tuple
colours = ['blue', 'red', 'green']  # list

for item in record:
    print(item)

for item in colours:
    print(item)

['blue', 'blue', 'red'] == ['red', 'blue', 'blue']  # order matters!

record[0]  # Python is zero-indexed

colours[2]  # get the Nth item

colours[-1]  # get the last item

record[0] = 'Jane'  # tuples are immutable!

colours[0] = 'yellow'  # lists are mutable
colours

colours.append('orange')
colours

colours.extend(['black', 'white'])
colours

colours.extend('purple')
colours

colours = ['blue', 'red', 'green', 'black', 'white']

colours[0:2]

colours[3:4]

colours[:3]

colours[:-1]

# This is equivalent of the built-in range() in Python 3
def my_range(n):
    num = 0
    while num < n:
        yield num
        num += 1

x = my_range(5)
x

for item in x:
    print(item)

for item in x:
    print(item)

import sys

sys.getsizeof(my_range(5))

sys.getsizeof(list(my_range(5)))

sys.getsizeof(my_range(1000))

sys.getsizeof(list(my_range(1000)))



