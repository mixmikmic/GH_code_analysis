my_dict = {}
my_dict[('Aly', 'Sivji')] = True
my_dict

from collections import namedtuple

Person = namedtuple('Person', ['first_name', 'last_name'])
me = Person(first_name='Bob', last_name='Smith')
my_dict[me] = True
my_dict

import this

