import unittest

class Temperature:
    pass

class TestTemperature(unittest.TestCase):
    
    def test_to_celcius(self):
        temp = Temperature(32)
        self.assert_equal(0, temp.celcius())

unittest.main()



"""
sum(2, 3)
4

sum(2, 3)
5
"""
def sum(a, b):
    return a + b

import doctest

doctest.testmod()



from fractions import Fraction

Fraction(2, 3) + Fraction(4, 5)



from functools import partial

s = partial(sum, 2)

s(3)



from glob import iglob

for file in iglob('Sess*'):
    print(file)



import hmac

m = hmac.HMAC(b'abc', b'This is message')

dir(m)

m.hexdigest()















