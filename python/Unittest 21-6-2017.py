import unittest
import sys

def multiply(x, y):
    return x * y

# Create a test case, inherits from unittest.TestCase

class TestMultiply(unittest.TestCase):
    # Create the unit test
    def test_multiply_two_integers_together(self):
        # Test if 4 equals the output of multiply(2,2)
        self.assertEqual(4, multiply(2,2))

# Run the unit test (and don't shut down the Jupyter Notebook)
unittest.main(argv=['ignored', '-v'], exit=False)



