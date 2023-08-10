import types  # <-- to get FunctionType

class Composable:
    """
    Composable swallows a function, which may still be called, by
    calling the instance instead.  Used as a decorator, the Composable
    class enables composition of functions by means of multiplying
    and powering their corresponding Composable instances.
    """
    
    def __init__(self, func):
        self.func = func     # eat a callable
        
    def __call__(self, x):
        return self.func(x)  # still a callable
        
    def __mul__(self, other):
        """
        multiply two Composables i.e. (f * g)(x) == f(g(x))
        g might might a function. OK if f is Composable.
        """
        if isinstance(other, types.FunctionType): # OK if target is a function
            other = Composable(other)
        if not isinstance(other, Composable): # by this point, other must be one
            raise TypeError
            
        return Composable(lambda x: self.func(other.func(x)))  # compose 'em
    
    def __rmul__(self, other): # in case other is on the left
        """
        multiply two Composers i.e. (f * g)(x) == f(g(x))
        f might might a function. OK if g is Composer.
        """
        if isinstance(other, types.FunctionType): # OK if target is a function
            other = Composable(other)
        if not isinstance(other, Composable): # by this point, other must be a Composer
            raise TypeError
        return Composable(lambda x: other.func(self.func(x)))  # compose 'em
        
    def __pow__(self, exp):
        """
        A function may compose with itself why not?
        """
        # type checking:  we want a non-negative integer
        if not isinstance(exp, int):
            raise TypeError
        if not exp > -1:
            raise ValueError
        me = self
        if exp == 0: # corner case
            return Composable(lambda x: x) # identify function
        elif exp == 1:
            return me                # (f**1) == f
        for _ in range(exp-1):       # e.g. once around loop if exp==2
            me = me * self
        return me
        
    def __repr__(self):
        return "Composable({})".format(self.func.__name__)

@Composable           
def f(x):
    "second powering"
    return x ** 2

@Composable
def g(x):
    "adding 2"
    return x + 2

print("(f * g)(7):", (f * g)(7))  # add 2 then 2nd power
print("(g * g)(7):", (g * f)(7))  # 2nd power then add 2

import unittest
import sys

class TestComposer(unittest.TestCase):

    def test_simple(self):
        x = 5
        self.assertEqual((f*g*g*f*g*f)(x), f(g(g(f(g(f(x)))))), "Not same!")
    
    def test_function(self):
        def addA(s): # not decorated
            return s + "A"
        @Composable
        def addM(s): 
            return s + "M"  
            
        addAM = addM * addA  # Composable times regular function, OK?
        self.assertEqual(addAM("I "), "I AM", "appends A then M")
        addMA = addA * addM  # regular function, times Composable OK?
        self.assertEqual(addMA("HI "), "HI MA", "appends M then A")
        
    def test_inputs(self):
        @Composable           
        def f(x):
            "second powering"
            return x ** 2
        
        self.assertRaises(TypeError, f.__pow__, 2.0)  # float not OK!
        self.assertRaises(TypeError, f.__pow__, g)    # another function? No!
        self.assertRaises(ValueError, f.__pow__, -1)  # negative number? No!
        
    def test_powering(self):
        @Composable           
        def f(x):
            "second powering"
            return x ** 2
        @Composable
        def g(x):
            "adding 2"
            return x + 2
        
        self.assertEqual((f*f)(10), 10000, "2nd power of 2nd power")
        self.assertEqual(pow(f, 3)(4), f(f(f(4))), "Powering broken")        
        h = (f**3) * (g**2)
        self.assertEqual(h(-11), f(f(f(g(g(-11))))), "Powering broken")
        self.assertEqual((f**0)(100), 100, "Identity function")
        
the_tests = TestComposer()        
suite = unittest.TestLoader().loadTestsFromModule(the_tests)
output = unittest.TextTestRunner(stream=sys.stdout).run(suite)
if output.wasSuccessful():
    print("All tests passed!")

