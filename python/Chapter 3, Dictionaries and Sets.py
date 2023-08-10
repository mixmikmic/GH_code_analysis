from collections import Mapping
my_dict = {}
isinstance(my_dict, Mapping)  # useful for checking if some object ultimately inherits from the basic Python dict

# what is hashable?

tt = (1, 2, (30, 40))
hash(tt)

tl = (1, 2, [30, 40])
hash(tl)

tf = (1, 2, frozenset([30, 40]))
hash(tf)

# ways to make a dict

a = dict(one=1, two=2, three=3)
b = {'one': 1, 'two': 2, 'three': 3}
c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
d = dict([('two', 2), ('one', 1), ('three', 3)])
e = dict({'three': 3, 'one': 1, 'two': 2})
a == b == c == d == e

DIAL_CODES = [
    (86, 'China'),
    (91, 'India'),
    (1, 'United States'),
    (62, 'Indonesia'),
    (55, 'Brazil'),
    (92, 'Pakistan'),
    (880, 'Bangladesh'),
    (234, 'Nigeria'),
    (7, 'Russia'),
    (81, 'Japan'),
]
country_code = {country: code for code, country in DIAL_CODES}
country_code

{code: country.upper() for country, code in country_code.items() if code < 66}

# setdefault() instead of .get() in a loop.

# so NOT
index = {}
_list = ['one', 'one', 'two', 'four', 'five', 'one']
for count, word in enumerate(_list):
    occurrences = index.get(word, [])
    occurrences.append(count)
    index[word] = occurrences
print(index)

index = {}
for count, word in enumerate(_list):
    index.setdefault(word, []).append(count)
print(index)

#defaultdict

from collections import defaultdict
dd = defaultdict(list)
for count, word in enumerate(_list):
    dd[word].append(count)
print(dd)

# __missing__
# UserDict for user-defined dict; don't inherit straight from dict

from collections import UserDict

class StrKeyDict(UserDict):
    """F"""

    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]
    
    def __contains__(self, key):
        return str(key) in self.data

    def __setitem__(self, key, item):
        self.data[str(key)] = item



