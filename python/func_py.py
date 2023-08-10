# A pure function
def my_min(x, y):
    if x < y:
        return x
    else:
        return y


# An impure function
# 1) Depends on global variable, 2) Changes its input
exponent = 2

def my_powers(L):
    for i in range(len(L)):
        L[i] = L[i]**exponent
    return L

def min_def(x, y):
    return x if x < y else y

min_lambda = lambda x, y: x if x < y else y

class MinClass:
    def __call__(self, x, y):
        return x if x < y else y

min_class = MinClass()
print(min_def(2,3) == min_lambda(2, 3) == min_class(2,3))

def append_to(element, to=[]):
    to.append(element)
    return to

my_list = append_to(12)
print("my_list:", my_list)
my_other_list = append_to(42)
print("my_other_list:", my_other_list)

def append_to2(element, to=None):
    if to is None:
        to = []
    to.append(element)
    return to

my_list2 = append_to2(12)
print("my_list2:", my_list2)
my_other_list2 = append_to2(42)
print("my_other_list2:", my_other_list2)

def create_multipliers():
    multipliers = []

    for i in range(5):
        def multiplier(x):
            return i * x
        multipliers.append(multiplier)

    return multipliers

for multiplier in create_multipliers():
    print(multiplier(2))

# Higher order functions

def makebold(fn):
    def wrapped():
        return "<b>" + fn() + "</b>"
    return wrapped

def hello():
    return "hello world"

print(hello())
hello = makebold(hello)
print(hello())

# Decorated function with *args and **kewargs

def makebold(fn):
    def wrapped(*args, **kwargs):
        return "<b>" + fn(*args, **kwargs) + "</b>"
    return wrapped

@makebold  # hello = makebold(hello)
def hello(*args, **kwargs):
    return "Hello. args: {}, kwargs: {}".format(args, kwargs)

print(hello('world', 'pythess', where='soho'))

# Decorators can be combined

def makeitalic(fn):
    def wrapped(*args, **kwargs):
        return "<i>" + fn(*args, **kwargs) + "</i>"
    return wrapped

def makebold(fn):
    def wrapped(*args, **kwargs):
        return "<b>" + fn(*args, **kwargs) + "</b>"
    return wrapped

@makeitalic
@makebold  # hello = makeitalic(makebold(hello))
def hello(*args, **kwargs):
    return "Hello. args: {}, kwargs: {}".format(args, kwargs)

print(hello('world', 'pythess', where='soho'))

# Decorators can be instances of callable classes

class BoldMaker:
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, *args, **kwargs):
        return "<b>" + self.fn(*args, **kwargs) + "</b>"

@BoldMaker  # hello = Bookmaker(hello)
def hello(*args, **kwargs):
    return "Hello. args: {}, kwargs: {}".format(args, kwargs)

# hello.__call__(*args, **kwargs)
print(hello('world', 'pythess', where='soho'))

# Decorators can take arguments

def enclose_in_tags(opening_tag, closing_tag):  # returns a decorator
    def make_with_tags(fn):  # returns a decorated function
        def wrapped():  # the function to be decorated (modified)
            return opening_tag + fn() + closing_tag
        return wrapped
    return make_with_tags

# decorator function make_with_tags with the arguments in closure
heading_decorator = enclose_in_tags('<h1>', '</h1>')
paragraph_decorator = enclose_in_tags('<p>', '</p>')

def hello():
    return "hello world"

h1_hello = heading_decorator(hello)
p_hello = paragraph_decorator(hello)
h1_p_hello = heading_decorator(paragraph_decorator(hello))
print(h1_hello())
print(p_hello())
print(h1_p_hello())

# Decorators with arguments combined

def enclose_in_tags(opening_tag, closing_tag):
    def make_with_tags(fn):
        def wrapped():
            return opening_tag + fn() + closing_tag
        return wrapped
    return make_with_tags

# hello = enclose_in_tags('<h1>', '</h1>')(hello)
@enclose_in_tags('<h1>', '</h1>') 
def hello():
    return "hello world"

print(hello())

# hello = enclose_in_tags('<p>', '</p>')(hello)
@enclose_in_tags('<p>', '</p>')
def hello():
    return "hello world"

print(hello())

# hello = enclose_in_tags('<h1>', '</h1>')(enclose_in_tags('<p>', '</p>')(hello))
@enclose_in_tags('<h1>', '</h1>')
@enclose_in_tags('<p>', '</p>')
def hello():
    return "hello world"

print(hello())

# Decorators with arguments as instances of callable classes

class TagEncloser:
    def __init__(self, opening_tag, closing_tag):
        self.opening_tag = opening_tag
        self.closing_tag = closing_tag
    def __call__(self, fn):
        def wrapped():
            return self.opening_tag + fn() + self.closing_tag
        return wrapped

tag_h1 = TagEncloser('<h1>', '</h1>')
tag_p = TagEncloser('<p>', '</p>')

@tag_h1
@tag_p
def hello():  # hello = tag_h1(tag_p(hello))
    return "hello world"

print(hello())

x = [1, 2, 3]  # this is an iterable
y = iter(x)  # an iterator of this iterable
z = iter(x)  # another iterator of this iterable
print(next(y))
print(next(y))
print(next(z))
print(type(x))
print(type(y))

s = 'cat'  # s is an ITERABLE
           # s is a str object that is immutable
           # s has no state
           # s has a __getitem__() method 

t = iter(s)    # t is an ITERATOR
               # t has state (it starts by pointing at the "c")
               # t has a __next__() method and an __iter__() method
try:
    print(next(t))        # the next() function returns the next value and advances the state
    print(next(t))        # the next() function returns the next value and advances
    print(next(t))        # the next() function returns the next value and advances
    print(next(t))        # next() raises StopIteration to signal that iteration is complete
except StopIteration as e:
    print("StopIteration raised")

class FibonacciIterator:
    """
    Produces an arbitrary number of the Fibonacci numbers.
    Is an both an iterable and an iterator.
    """
    def __init__(self):
        self.prev = 0
        self.curr = 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.prev, self.curr = self.curr, self.prev + self.curr
        return self.prev

class Countdown:
    """A simple iterable, NOT an iterator"""
    def __iter__(self):
        return iter([5, 4, 3, 2, 1, 'launch'])
    
f = FibonacciIterator()
print([next(f) for _i in range(10)])
c = Countdown()
print([i for i in c])

# generator expression
g = (x ** 2 for x in range(10))
print(next(g))
print(next(g))
print([i for i in g])

# generator function
def _gen(exp):
    for x in exp:
        yield x ** 2
g = _gen(range(10))
print(next(g))
print(next(g))
print([i for i in g])

# generator
def it_gen(text):
    for ch in text:
        yield ch.upper()

# generator expression
def it_genexp(text):
    return (ch.upper() for ch in text)

# iterator protocol (__iter__)
class ItIter():
    def __init__(self, text):
        self.text = text
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        try:
            result = self.text[self.index].upper()
        except IndexError:
            raise StopIteration
        self.index += 1
        return result

# iterator protocol (__getitem__)
class ItGetItem():
    def __init__(self, text):
        self.text = text
    def __getitem__(self, index):
        result = self.text[index].upper()
        return result

# an iterable of iterables (see what i did there? :P)
for iterator in (it_gen, it_genexp, ItIter, ItGetItem):
    for ch in iterator('abcde'):
        print(ch, end='')
    print('\n')

import collections

def flatten(container):
    """Flatten an iterable of arbitrary depth."""
    for item in container:
        if isinstance(item, collections.Iterable) and not isinstance(item, (str, bytes)):
            for element in flatten(item):
                yield element
        else:
            yield item

d3 = [[[i for i in range(2)] for j in range(3)] for k in range(4)]
print(d3)
print([k for i in d3 for j in i for k in j])
print([i for i in flatten(d3)])

