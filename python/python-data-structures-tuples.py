# Define a tuple as comma separate values, with(out) enclosing round brackets
number_tuple = 4, 9, 11, 2, 1, 9, 11
number_tuple

# Define a mixed data type tuple
mixed_tuple = (1, 'One', 2, 'Two', 3, 'Three')
mixed_tuple

# Define nested tuple from multiple tuples
nested_tuple = number_tuple, mixed_tuple
nested_tuple

# Define tuple of other sequence types
number_list = [1, 2, 3]
simple_string = 'abcdefg'
sequence_tuple = number_list, simple_string, 1, 3
sequence_tuple

# Define tuple of one item with trailing comma
single_item_tuple = 'One',
single_item_tuple

# Define empty tuple
empty_tuple = ()
empty_tuple

packed_tuple = "Jhon", "Doe", 32, "Male"
first, last, age, sex = packed_tuple
print(first, last, age, sex)

