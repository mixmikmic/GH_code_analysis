my_tuple = ('a', 'b', [1, 2])
type(my_tuple)

# First we'll create a dictionary and insert a (key, value) pair
my_dict = {}
my_dict['test'] = 'foo'
my_dict

# Let's make sure that we cannot use a list element in the dictionary
my_dict[[1, 2]] = 'foo'

# Dictionary has not changed
my_dict

# Let's try inserting a tuple containing a list, as a key
my_dict[my_tuple] = 'bar'

from collections import namedtuple

TestTuple = namedtuple('TestTuple', ['field1', 'field2', 'field3'])
my_item = TestTuple('a', 'b', [1, 2])
type(my_item)

isinstance(my_item, tuple)

my_dict[my_item] = 'bar'

