import numpy as np
from nose.tools import assert_true

### STUDENT ANSWER
def square(x):
    return x**2

random_numbers = np.random.randint(0, 50, 50)
assert_true(all(ii ** 2 == square(ii) for ii in random_numbers))

