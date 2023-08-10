from flexx import react

@react.input
def first_name(n='John'):
    assert isinstance(n, str)  # validation
    return n.capitalize()  # normalization

@react.input
def last_name(n='Doe'):
    assert isinstance(n, str)
    return n.capitalize()

first_name()  # get signal value

first_name('jane')  # set signal value (for input signals)
first_name()

@react.connect('first_name', 'last_name')
def name(first, last):
    return '%s %s' % (first, last)

@react.connect('name')
def greet(n):
    print('hello %s!' % n)

first_name('Guido')

last_name('van Rossum')

class Item(react.HasSignals):
    
    @react.input
    def name(n):
        return str(n)

class Collection(react.HasSignals):

    @react.input
    def items(items):
        assert all([isinstance(i, Item) for i in items])
        return tuple(list(items))
    
    @react.input
    def ref(i):
        assert isinstance(i, Item)
        return i

itemA, itemB, itemC, itemD = Item(name='A'), Item(name='B'), Item(name='C'), Item(name='D')
C1 = Collection(items=(itemA, itemB))
C2 = Collection(items=(itemC, itemD))

itemB.name()

C1.items()

class Collection2(Collection):
    
    @react.connect('ref.name')
    def show_ref_name(name):
        print('The ref is %s' % name)
    
    @react.connect('items.*.name')
    def show_index(*names):
        print('index: '+ ', '.join(names))

itemA, itemB, itemC, itemD = Item(name='A'), Item(name='B'), Item(name='C'), Item(name='D')
C1 = Collection2(items=(itemA, itemB))
C2 = Collection2(items=(itemC, ))

C1.ref(itemA)

C1.ref(itemD)

itemD.name('D-renamed')

C2.items([itemC, itemD])

itemC.name('C-renamed')

@react.input
def foo(v):
    return str(v)

@react.lazy('foo')
def bar(v):
    print('update bar')
    return v * 10  # imagine that this is an expensive operation

foo('hello')  # Does not trigger bar
foo('heya')
foo('hi')
bar()  # this is where bar gets updated

bar()  # foo has not changed; cached value is returned

@react.input
def some_value(v=0):
    return float(v)
some_value(0)  # init

@react.connect('some_value')
def show_diff(s):
    print('diff: ', s - some_value.last_value)  # note: we might rename this to previous_value

some_value(10)

some_value(12)

