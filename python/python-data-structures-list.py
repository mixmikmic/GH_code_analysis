# Define a list of integers
number_list = [3, 2, 1, 3, 5, 9, 6, 3, 9]
number_list

# List can contain strings
word_list = ['Jan', 'Feb', 'Mar', 'Apr']
word_list

# List can contain mixed data types
mixed_list = [1, 'Jan', 2, 'Feb', 3, 'Mar']
mixed_list

# Lists can be n-dimensional or list of lists of...
matrix_list = [[1, 'Jan'], [2, 'Feb'], [3, 'Mar']]
matrix_list

list_3d = [[1, 'Jan', ['Mon', 'Tue']], 
           [2, 'Feb', ['Wed', 'Thu']], 
           [3, 'Mar', ['Fri', 'Sat']]]
list_3d

# First item on the list starts at 0 index
number_list[0]

# Last item on the list starts at -1 index
number_list[-1]

# Refer list of lists items using index
list_3d[2][2][0]

# String is a list of characters, can refer using index
print(word_list[1], word_list[1][1])

# Slicing first 6 items in the list
number_list[:6]

# Last 4 items in the list
number_list[-4:]

# 3 items (5-2) starting from 2 index
number_list[2:5]

# Find first index of an item
number_list.index(9)

# Length of list
len(number_list)

# Count of items in the list
number_list.count(9)

# Max item in list
max(number_list)

# Min item in list
min(number_list)

# Append an item to the list
print(number_list)
number_list.append(10)
number_list

# Extend list with another list
number_list.extend([13, 15, 15, 12])
number_list

# Insert at index i, an item n to the list (i, n)
number_list.insert(5, 19)
number_list

# Remove first occurance of an item from the list
number_list.remove(9)
number_list

# Remove at index, an item from the list, return removed item
last_item = number_list.pop(5)
print(number_list)
last_item

# Delete a slice from the list
del number_list[1:3]
number_list

# Replace an item in the list
number_list[0] = 7
number_list

# Replace a slice in the list
number_list[:3] = [31, 42, 91]
number_list

# Sort list
number_list.sort()
number_list

# Reverse list
number_list.reverse()
number_list

# Filter list to apply functional checks on all list items
def odds(x): return (x % 2) != 0
odds_list = filter(odds, number_list)
print(number_list)
odds_list

# Map list to apply operations on all list items
def squares(x): return (x ** 2)
squares_list = map(squares, number_list)
squares_list

# Map multiple lists to combine into one
def divide(x, y): return (x / y)
original_list = map(divide, squares_list, number_list)
original_list

# Reduce list to one item
def total(x, y): return (x + y)
totalled = reduce(total, number_list)
totalled

stack = number_list

# Push two items at the end or top of the stack
stack.append(21)
stack.append(23)
stack

# Pop one item from the end or top of the stack
popped_item = stack.pop()
print(stack)
popped_item

# Create new list of squares for first n integers
n = 10
squares_n = [x**2 for x in range(n)]
squares_n

# Create new list of squares for first n odd integers
n = 10
squares_odd_n = [x**2 for x in range(n) if (x % 2) != 0]
squares_odd_n

# Create new list by applying a method on items from another
smaller_words = [x.lower() for x in word_list]
smaller_words

# Create new list by applying a function on items from another
absolute_list = [abs(x) for x in [-3, -5, 1, 4, 7]]
absolute_list

# Return a list of lists with original and new item
capitalize_words = [[x, x.upper()] for x in word_list]
capitalize_words

# Return list applying operation on each item combination from two lists
combine_list = [x * y for x in [1, 2, 3] for y in [3, 1, 4]]
combine_list

# Return list of lists for matching criteria between two lists
match_list = [[x, y] for x in [1, 2, 3] for y in [3, 1, 4] if x != y]
match_list

# Pack and then unpack two lists
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
open_times = ['9:30', '8:30', '8:30', '10:30', '11:00']
packed = zip(days, open_times)
packed

unpacked_days, unpacked_times = zip(*packed)
print(unpacked_days)
unpacked_times

# Transpose a matrix of lists
print(matrix_list)
zip(*matrix_list)

