from toolz import curry, memoize

@curry
class Person(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __repr__(self):
        return "Person(name={!r}, age={!r})".format(self.name, self.age)
        
p = Person(name='alec')
p(age=26)

class PersonWithHobby(Person):
    def __init__(self, name, age, hobby):
        super(PersonWithHobby, self).__init__(name, age)
        self.hobby = hobby

class PersonWithHobby(Person.func):
    def __init__(self, name, age, hobby):
        super(PersonWithHobby, self).__init__(name, age)
        self.hobby = hobby

from functools import wraps

class Curryable(type):
    # one level up from classes
    # cls here is the actual class we've created already
    def __call__(cls, *args, **kwargs):
        # we'd like to preserve metadata but not migrate
        # the underlying dictionary
        @wraps(cls, updated=[])
        # distinguish from what was passed to __call__
        # and what as passed to currier
        def currier(*a, **k):
            return super(Curryable, cls).__call__(*a, **k)
        # there's sometimes odd behavior if this isn't done
        return curry(currier)(*args, **kwargs)

class Person(metaclass=Curryable):
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __repr__(self):
        return "Person(name={!r}, age={!r})".format(self.name, self.age)

p = Person(name='alec')

p(age=26)

class PersonWithHobby(Person):
    # as an example only; it's still best practice to declare required parameters
    def __init__(self, hobby, **kwargs):
        super(PersonWithHobby, self).__init__(**kwargs)
        self.hobby = hobby

    def __repr__(self):
        return "Person(name={!r}, age={!r}, hobby={!r})".format(self.name, self.age, self.hobby)
        
p = PersonWithHobby(hobby='coding')

p(name='alec', age=26)

def default_cache_key(args, kwargs):
    return (args or None, frozenset(kwargs.items()) or None)

class HybridValueStore(object):
    def __init__(self, valuestore):
        self.valuestore = valuestore
        
            #   |+------------------> The Descriptor Instance
            #   |     |+------------> The Memoized Class
            #   |     |     |+------> The Metaclass
    def __get__(self, inst, cls):
        if inst is None:
            return self.valuestore
        else:
            return self.valuestore[inst]
    
    def __set__(self, inst, value):
        self.valuestore[inst] = value
    
    def __delete__(self, inst):
        self.valuestore.pop(inst, None)

from toolz import memoize

class Memoized(type):
    cache = HybridValueStore({})
    cache_key = HybridValueStore({})
    
    def __new__(mcls, name, bases, attrs, **kwargs):
        return super(Memoized, mcls).__new__(mcls, name, bases, attrs)
   
    def __init__(cls, name, bases, attrs, key=default_cache_key, cache=None):
        if cache is None:
            cache = {}
        cls.cache = cache
        cls.cache_key = key
        super(Memoized, cls).__init__(name, bases, attrs)
    
    def __call__(cls, *args, **kwargs):
        @memoize(cache=cls.cache, key=cls.cache_key)
        def memoizer(*a, **k):
            return super(Memoized, cls).__call__(*a, **k)
        return memoizer(*args, **kwargs)

class Frob(metaclass=Memoized):
    def __init__(self, frob):
        self.frob = frob
    
    def __repr__(self):
        return "Frob({})".format(self.frob)

# simply here to show HybridValueStore's fine grained access
class Dummy(metaclass=Memoized):
    def __init__(self, *args, **kwargs):
        pass
    
    def __repr__(self):
        return "Dummy"
    
f = Frob(1)
d = Dummy()
assert f is Frob(1), "guess it didn't work"

print("Master Cache: ", Memoized.cache)
print("Frob   Cache: ", Frob.cache)
print("Dummy  Cache: ", Dummy.cache)

Frob.cache = {}
print("Master Cache: ", Memoized.cache)
print("Frob   Cache: ", Frob.cache)
print("Dummy  Cache: ", Dummy.cache)

from collections import OrderedDict

def make_string_key(args, kwargs):
    return str(args) + str(kwargs)

class KeywordTest(metaclass=Memoized, key=make_string_key, cache=OrderedDict()):
    def __init__(self, *args, **kwargs):
        pass

kwt1 = KeywordTest(1, 2, 3)
kwt2 = KeywordTest(4, 5, 6)

print(KeywordTest.cache)

f.cache

class CurriedMemoized(Curryable, Memoized):
    pass

class CMTester(metaclass=CurriedMemoized):
    def __init__(self, *args, **kwargs):
        pass

CMTester(1, 2, 3)
print(CMTester.cache)

class CMKeywordTest(metaclass=CurriedMemoized, key=make_string_key, cache=OrderedDict()):
    def __init__(self, *args, **kwargs):
        pass
    
CMKeywordTest(1, 2, 3)
CMKeywordTest(4, 5, 6)
print(CMKeywordTest.cache)

class MemoizedCurry(Memoized, Curryable):
    pass

class MCTest(metaclass=MemoizedCurry):
    def __init__(self, name, frob):
        pass
    
m = MCTest(name='default frob')
m(frob=1)
print(MCTest.cache)

