from random import choice

class Trick(Exception):
    def __init__(self):
        self.value = "Goblins & Ghosts!"
    
class Halloween:
    
    def __init__(self, arg=None):
        self.testing = arg
    
    def __enter__(self):
        self.where = "Spooky Castle"
        print("Welcome...")
        self.trick_or_treat = ["Trick", "Treat"]
        self.candy = [ ]
        return self
    
    def __exit__(self, *uh_oh):  # catch any exception info
        if uh_oh[0]: 
            print("Trick!")
            print(uh_oh[1].value)  # lets look inside the exception
            return False
        return True

try:
    with Halloween("Testing 1-2-3") as obj:
        print(obj.testing)
        if choice(obj.trick_or_treat) == "Trick":
            raise Trick    
except:
    print("Exception raised!")

import unittest

class TestCastle(unittest.TestCase):
    
    def test_candy(self):
        outer = ""
        with Halloween() as context:
            outer = context.candy
        self.assertEqual(outer, [], "Not OK!")
        
    def test_trick(self):
        outer = ""
        def func():
            with Halloween() as context:
                raise Trick
        self.assertRaises(Trick, func)
        
a = TestCastle()  # the test suite
suite = unittest.TestLoader().loadTestsFromModule(a) # fancy boilerplate
unittest.TextTestRunner().run(suite)  # run the test suite

