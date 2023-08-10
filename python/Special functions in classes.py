a = 2
b = 3

a + b

dir(a)

a.__add__(b)

a = '2'
b = '3'

a + b

a.__add__(b)

class TestClass:
    def __init__(self, value):
        self.value = value

a = TestClass(2)
b = TestClass(3)

a

class TestClass:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return "TestClass({})".format(self.value)

a = TestClass(2)
b = TestClass(3)

a

a + b

class TestClass:
    def __init__(self, value):
        self.value = value
        
    def __repr__(self):
        return "TestClass({})".format(self.value)
    
    def __add__(self, other):
        return TestClass(self.value + other.value)

a = TestClass(2)
b = TestClass(3)

a + b

