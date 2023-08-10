class Cup:
    pass

coffee_mug = Cup()
type(coffee_mug)

class Cup:
    def __init__(self, name, capacity, current_volume):
        self.name = name
        self.capacity = capacity
        self.current_volume = current_volume

coffee_mug = Cup('coffee mug', 200, 50)
coffee_mug.current_volume

coffee_mug = Cup('coffee mug', 200, 50)
water_glass = Cup('large water glass', 400, 300)

# It time for another coffee?
print("{0} is {1} percent full".format(coffee_mug.name, coffee_mug.current_volume / coffee_mug.capacity * 100))

class Cup:
    def __init__(self, name, capacity, current_volume):
        self.name = name
        self.capacity = capacity
        self.current_volume = current_volume
    
    def status(self):
        return "{0} is {1:.0f}% full".format(
            self.name,
            self.current_volume / self.capacity * 100)

coffee_mug = Cup('coffee mug', 200, 50)
water_glass = Cup('large water glass', 400, 300)
coffee_mug.status()

print(coffee_mug)

class Cup:
    def __init__(self, name, capacity, current_volume):
        self.name = name
        self.capacity = capacity
        self.current_volume = current_volume
        
    def __str__(self):
        return "{0} is {1:.0f}% full".format(
            self.name,
            self.current_volume / self.capacity * 100)
    
    def status(self):
        # Keep the status method, but redirect to the __str__ method
        return self.__str__()

coffee_mug = Cup('coffee mug', 200, 20)
print(coffee_mug)

class Cup:
    def __init__(self, name, capacity, current_volume):
        self.name = name
        self.capacity = capacity
        self.current_volume = current_volume
        
    def __str__(self):
        if self.current_volume > 0:
            return "{0} is {1:.0f}% full".format(
                self.name,
                self.current_volume / self.capacity * 100)
        else:
            return "{0} is empty".format(self.name)
    
    def status(self):
        # Keep the status method, but redirect to the __str__ method
        return self.__str__()
    
    def sip(self, amount=15):
        # We guard against reducing the volume below zero, but don't guard against negative amounts.
        self.current_volume = max(0, self.current_volume - amount)
        print('sip')
        # return a reference to self so we can chain method calls
        return self
        
    def refill(self):
        self.current_volume = self.capacity
        # return a reference to self so we can chain method calls
        return self

coffee_mug = Cup('coffee mug', 200, 200)
print(coffee_mug)
print(coffee_mug.sip().status())
print(coffee_mug.sip().sip().status())

coffee_mug.refill()
print(coffee_mug)

coffee_mug.sip(200)
print(coffee_mug)

class Cup:
    def __init__(self, name, capacity, current_volume):
        self.name = name
        self.capacity = capacity
        self.current_volume = current_volume
        
    def __str__(self):
        if self.current_volume > 0:
            return "{0} can hold {1} ml and currently holds {2} ml ({3:.0f}% full)".format(
                self.name,
                self.capacity,
                self.current_volume,
                self.current_volume / self.capacity * 100)
        else:
            return "{0} is empty".format(self.name)

magic_cup = Cup('magic cup', 100, 50)
print(magic_cup)

# Make it smaller
magic_cup.capacity = 10
print(magic_cup)

class Cup:
    def __init__(self, name, capacity, current_volume):
        self._name = name
        self._capacity = capacity
        self._current_volume = current_volume
        
    def __str__(self):
        if self._current_volume > 0:
            return "{0} can hold {1} ml and currently holds {2} ml ({3:.0f}% full)".format(
                self._name,
                self._capacity,
                self._current_volume,
                self._current_volume / self._capacity * 100)
        else:
            return "{0} is empty".format(self._name)

magic_cup = Cup('magic cup', 100, 50)
magic_cup._capacity = 25
print(magic_cup)

class Cup:
    def __init__(self, name, capacity, current_volume):
        self.__name = name
        self.__capacity = capacity
        self.__current_volume = current_volume
        
    def __str__(self):
        if self.__current_volume > 0:
            return "{0} can hold {1} ml and currently holds {2} ml ({3:.0f}% full)".format(
                self.__name,
                self.__capacity,
                self.__current_volume,
                self.__current_volume / self.__capacity * 100)
        else:
            return "{0} is empty".format(self.__name)

magic_cup = Cup('magic cup', 100, 50)
dir(magic_cup)

magic_cup = Cup('magic cup', 100, 50)
print('before: ', magic_cup)
magic_cup.__capacity = 10
print('after:  ', magic_cup)

magic_cup._Cup__capacity = 10
print(magic_cup)

class Cup:
    def __init__(self, name, capacity, current_volume):
        # There is no reason for name to be private or a property. It's just a name.
        self.name = name
        self._current_volume = current_volume
        
        # When using properties, the property should be called from other class methods.
        # It is a good practice to only access the private backing field from the property 
        # accessors themselves.
        self.capacity = capacity
        
    # First we define a normal method for returning the property value.
    # The name of the method is the name we want for the public property.
    @property
    def capacity(self):
        # In this case we are returning the private value,
        # but you can also define properties that return derived values.
        return self.__capacity
    
    # Now we define a second method with the same name. 
    # This method contains the logic for setting the property value.
    @capacity.setter
    def capacity(self, capacity):
        # Store the new capacity
        # This might not be the best approach, but for simplicity treat
        # negative values as 0
        self.__capacity = max(0, capacity)
            
        # Spill any excess liquid
        self._current_volume = min(self._current_volume, self.capacity)
        
    def refill(self):
        self._current_volume = self.capacity
        
    def __str__(self):
        if self._current_volume > 0:
            return "{0} can hold {1} ml and currently holds {2} ml ({3:.0f}% full)".format(
                self.name,
                self.capacity,
                self._current_volume,
                self._current_volume / self.capacity * 100)
        else:
            return "{0} is empty".format(self.name)

coffee = Cup("Espresso!", 80, 40)
print(coffee)

# I don't want an espresso. Make it a long black
coffee.name = "long black"
coffee.capacity = 200
coffee.refill()
print(coffee)

# Actually, can I have an espresso after all?
coffee.name = "make up your mind!"
coffee.capacity = 80
print(coffee)

# Uh oh, I dropped and smashed the cup
coffee.capacity = -1
print(coffee)

# First, define the function that will become the new class method
@property
def percent_full(self):
    if self.capacity > 0:
        return self._current_volume / self.capacity * 100
    else:
        return 0

# now monkey patch the new property into the existing Cup class
Cup.percent_full = percent_full

# Let's try it out
patched_coffee = Cup("patched", 200, 150)
print(patched_coffee)
print(patched_coffee.percent_full)

try:
    patched_coffee.percent_full = 40
except:
    print("Yep, can't set the property")

dir(Cup)

dir(object)

class Parent(object):
    def implicit(self):
        print("{0} calling Parent implicit()".format(type(self)))
        
class Child(Parent):
    pass

dad = Parent()
son = Child()

dad.implicit()
son.implicit()

class Parent(object):
    def override(self):
        print("{0} calling Parent override()".format(type(self)))
                
class Child(Parent):
     def override(self):
        print("{0} calling Child override()".format(type(self)))

dad = Parent()
son = Child()

dad.override()
son.override()

class Parent(object):
    def altered(self):
        print("{0} calling Parent altered()".format(type(self)))
        
class Child(Parent):

    def altered(self):
        print("Child, doing some work before calling Parent altered()")
        super().altered()
        print("Child, doing some work after calling Parent altered()")

dad = Parent()
son = Child()

dad.altered()
print()
son.altered()

class Grandparent(object):
    def do_something(self):
        print("{0}: Grandparent.do_something()".format(type(self)))
        
class Parent(Grandparent):
    def do_something(self):
        # override Grandparent.do_something()
        print("{0}: Parent.do_something()".format(type(self)))
        
class Child(Parent):
    def do_something(self):
        # Do something ourselves
        print("{0}: Child.do_something()".format(type(self)))
        # Then call Grandparent.do_something(), bypassing the Parent override
        # This works because a class is also the name of an object, so read this line as
        # "call the do_something method on the class object called Grandparent, and pass self as the data reference"
        Grandparent.do_something(self)
        
    def do_something_with_super(self):
        # Do something ourselves
        print("{0}: Child.do_something_with_super()".format(type(self)))
        super().do_something()
        
child = Child()
child.do_something()
print()
child.do_something_with_super()

class Container(object):
    def __init__(self, name='container', current_volume=0, capacity=1000):
        self.name = name
        self.current_volume = current_volume
        self.capacity = capacity
        
    def __str__(self):
        return '{0} holds {1}/{2}'.format(self.name, self.current_volume, self.capacity)
    
class Cup(Container):
    # Override the superclass __init__ so we can define different default values
    def __init__(self, name='cup', current_volume=0, capacity=200):
        # Now call __init__ on the superclass, passing in the required data
        super().__init__(name, current_volume, capacity)
        
    def drink(self, amount=10):
        print('Drinking ', amount)
        self.current_volume = max(self.current_volume - amount, 0)
        
container = Container()
print(container)

cup = Cup(current_volume=150)
cup.drink()

print(cup)

# you can drink from a cup, but not a container?
container.drink()

if isinstance(container, Cup):
    container.drink()
else:
    print("Can't drink")

try:
    container.drink()
except:
    print("Can't drink")

repr(cup)

class Container(object):
    def __init__(self, name='container', current_volume=0, capacity=1000):
        self.name = name
        self.current_volume = current_volume
        self.capacity = capacity
        
    def __str__(self):
        return '{0} holds {1}/{2}'.format(self.name, self.current_volume, self.capacity)
    
    def __repr__(self):
        return 'Container(name={0}, current_volume={1}, capacity={2})'.format(
            self.name,
            self.current_volume,
            self.capacity)
    
c = Container()
print(str(c))
print(repr(c))

# Monkey patch into the previous Container definition
def _repr_html_(self):
    return '<span style="color:green"><h1>{0}</h1></span><span style="color:blue">{1}/{2}</span>'.format(
        self.name, self.current_volume, self.capacity)

Container._repr_html_ = _repr_html_

Container()

class FunctionWrapper(object):

    def __init__(self, arg1, arg2=5):
        self.arg1 = arg1
        self.arg2 = arg2
        
    def __str__(self):
        return 'arg1={0}, arg2={1}'.format(self.arg1, self.arg2)

    def __call__(self, f):
        def new_f():
            print("Entering", f.__name__)
            print(self)
            f()
            print("Leaving", f.__name__)
        return new_f


@FunctionWrapper("Eric the half a bee")
def hello():
    print("Hello")
    
hello()

class Container(object):
    def __init__(self, name='container', current_volume=0, capacity=1000):
        self.name = name
        self.current_volume = current_volume
        self.capacity = capacity
        
    def __str__(self):
        return '{0} holds {1}/{2}'.format(self.name, self.current_volume, self.capacity)
    
    def __bool__(self):
        "Returns True if the container contains something, or False if it is empty."
        return self.current_volume > 0
    
class Cup(Container):
    # Override the superclass __init__ so we can define different default values
    def __init__(self, name='cup', current_volume=0, capacity=200):
        # Now call __init__ on the superclass, passing in the required data
        super().__init__(name, current_volume, capacity)
        
    def drink(self, amount=10):
        self.current_volume = max(self.current_volume - amount, 0)

        
coffee = Cup("coffee", current_volume=50)

# note how we are relying on the __bool__ behaviour inherited from the superclass
while coffee:
    print('drink some coffee')
    coffee.drink()

print('all gone')

class NotVeryUsefulEmail(object):   
    # To call this you need an object
    # And yes, I know this is a pointless and terrible implementation ...
    def is_valid(self, email):
        parts = email.split('@')
        if len(parts) == 2:
            return len(parts[0]) > 0 and len(parts[1]) > 0
        return False
    
my_email = "eric@monty.python"

# To call the not very useful validation method, we need an object
validator = NotVeryUsefulEmail()
print(validator.is_valid(my_email))

# Or we need to jump through strange hoops
print(NotVeryUsefulEmail.is_valid(None, my_email))

class Email(object):   
    # The staticmethod decorator indicates that this method doesn't require a self reference
    @staticmethod
    def is_valid(email):
        parts = email.split('@')
        if len(parts) == 2:
            return len(parts[0]) > 0 and len(parts[1]) > 0
        return False
    
print(Email.is_valid('john@python.com'))
print(Email.is_valid('eric@'))

