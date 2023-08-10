[6, 28]

[1e3, -2, "I am in a list."]

[[1.0, 0.0], [0.0, 1.0]]

[1, 1] + [2, 3, 5] + [8]

fib = [1, 1, 2, 3, 5, 8]

fib.append(13)     # Try two arguments (13,3)
fib

fib.insert(3,100)
fib

fib.extend([21, 34, 55])
fib

fib += [89, 144]
fib

fib[::2]

fib[3] = "whoops"     # Replace
fib

del fib[:6]
fib

fib[1::2] = [-1, -1, -1] 
fib 

[1, 2, 3] * 6

list("F = dp/dt")   # Including spaces, all spaces

x = []
x.append(x)
x

x[0]

x[0][0]

x = 42   # Python starts by first creating the number 42 in memory. 
         # It sets the name x to refer to the point in memory where 42 lives.    
y = x    # It sees that y should point to the same place that x is pointing to
del x    # x is deleted, but so it keeps both y and 42 around for later use.

x = [3, 2, 1, "blast off!"]
y = x
y[1] = "TWO"  # When y’s second element is changed to the string 'TWO', this change is reflected back onto x. This is because there is only one list in memory, even though there are two names for it (x and y).
print(x)
del x
print(y)

a = 1, 2, 5, 3  # length-4 tuple
b = (42,)       # length-1 tuple, defined by comma
c = (42)        # not a tuple, just the number 42
d = ()          # length-0 tuple- no commas means no elements
type(d)

(1, 2) + (3, 4)

1, 2 + 3, 4    # it carries out 2 + 3 = 5, then makes a tuple.

tuple(["e", 2.718])

x = 1.0, [2, 4], 16
x[1].append(8)
x

# a literal set formed with elements of various types
{1.0, 10, "one hundred", (1, 0, 0,0)}

# a literal set of special values
{True, False, None, "", 0.0, 0}

# conversion from a list to a set
set([2.0, 4, "eight", (16,)])

# Repetition is ignored
{1.0, 1.0, "one hundred", (1, 0, 0,0)}

# 1 and 1.0 are considered repetition
{1.0, 1, "one hundred", (1, 0, 0,0)}

set("Marie Curie")

set(["Marie Curie"])

s={1,2,3,6}
t={3,4,5}
s | t        # Union

s & t        # Intersection

s - t        # Difference - elements in s but not in t

s ^ t        # Symmetric difference - elements in s or t but not both

s < t        # Strict subset - test if every element in s is in t but not every element in t is in s 

s <= t        # Subset - test if every element in s is in t.

hash([3])

# A dictionary on one line that stores info about Einstein
al = {"first": "Albert", "last": "Einstein", "birthday": [1879, 3, 14]}

# You can split up dicts onto many lines
constants = {
    'pi': 3.14159,
    "e": 2.718,
    "h": 6.62606957e-34,
    True: 1.0,
    }

# A dict being formed from a list of (key, value) tuples
axes = dict([(1, "x"), (2, "y"), (3, "z")])
print(axes)

# You pull a value out of a dictionary by indexing with the associated key.
constants['e']

axes[3]

al['birthday']

constants[False] = 0.0
print(constants)
del axes[3]
print(axes)
al['first'] = "You can call me Al"
print(al)

d = {}
d['d'] = d
d['e'] = d
d

{}     # define empty dict
set()  # define empty set

# Tests for containment with the in operator function only on dictionary keys, not values:
"N_A" in constants

axes.update({1: 'r', 2: 'phi', 3: 'theta'})
axes

from IPython.core.display import HTML
def css_styling():
    styles = open("styles/custom.css", "r").read()
    return HTML(styles)
css_styling()



