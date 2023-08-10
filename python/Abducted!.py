def UFO(abductee):
    abductee.special_mark = True
    return abductee

def subject_A():
    pass

@UFO
def subject_B():
    pass

print(subject_B.special_mark)
        

def UFO(attr, value):
    """returns Abduct, poised to proceed"""
    def Abduct(abductee):  # incoming callable
        """set whatever attribute to the chosen value"""
        abductee.__setattr__(attr, value)
        return abductee # a callable, remember
    return Abduct

@UFO("arm", "strange symbol")  # ">> ☺ <<"
def subject_A():
    """just minding my own busines..."""
    pass

print("What's that on Subject A's arm?", subject_A.arm)

class Composer:
    """allow function objects to chain together"""
    
    def __init__(self, func):
        self.func = func  # swallow a function
        
    def __matmul__(self, other):
        return Composer(lambda x: self(other(x)))
    
    def __rmatmul__(self, other):
        return Composer(lambda x: other(self(x)))
        
    def __call__(self, x):
        return self.func(x)

def addA(s):
    return s + "A"

def addB(s):
    return s + "B"

result = addA(addA(addA("K")))  # ordinary composition
print(result)

result = addB(addA(addB("K")))
print(result)

@Composer
def addA(s):
    return s + "A"

def addB(s):
    return s + "B"

Chained = addB @ addA @ addB @ addA @ addB  # an example of operator overloading
print(Chained("Y"))

import unittest

class TestComposer(unittest.TestCase):
    
    def test_composing(self):
        
        def Plus2(x):
            return x + 2
        
        @Composer
        def Times2(x):
            return x * 2
        
        H = Times2 @ Plus2
        self.assertEqual(H(10), 24)

    def test_composing2(self):
        
        def Plus2(x):
            return x + 2
        
        @Composer
        def Times2(x):
            return x * 2
        
        H = Plus2 @ Times2
        self.assertEqual(H(10), 22)
        
    def test_composing3(self):
        
        def Plus2(x):
            return x + 2
        
        @Composer
        def Times2(x):
            return x * 2
        
        H = Plus2 @ Times2
        self.assertEqual(H(10), 22)
        
a = TestComposer()  # the test suite
suite = unittest.TestLoader().loadTestsFromModule(a) # fancy boilerplate
unittest.TextTestRunner().run(suite)  # run the test suite

from random import choice

def add_tricks(cls):
    tricks = ["play dead", "roll over", "sit up"]
    def do_trick(self):
        return choice(tricks)
    cls.do_trick = do_trick
    return cls
    
@add_tricks
class Animal:
    
    def __init__(self, nm):
        self.name = nm

class Mammal(Animal):
    pass

obj = Animal("Rover")
print(obj.name, "does this trick:", obj.do_trick())

new_obj = Mammal("Trixy")
print(new_obj.name, "does this trick:", obj.do_trick())

