get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import time
# see also the "datetime" package
import solutions

# This tells you the system time in seconds (from some system-dependent zero point)
time.time()

# here's a simple way to determine how long something takes to evaluate:
t0 = time.time()

# do some random task
x = -1
for k in range(0,100000):
    x = k*x

t1 = time.time()

duration = t1 - t0
print('This task took: ' + str(duration) + ' seconds')

# here's how you might return two numbers or arrays from a function

def myfunc(a):
    out1 = a+2
    out2 = a-2
    return out1, out2

x, y = myfunc(7)
print(x,y)

# an example of random numbers uniformly chosen in the range -5 to 5

# note, other random number distributions exist, and are very useful (e.g. Gaussian!)

N = 20

rmax = 5
rmin = -5

rando_calrissian = np.random.random(N) * (rmax - rmin) + rmin

rando_calrissian

def number_test(input):
    '''
    An example of using a Try/Except statement to catch an error.
    You could imagine doing something more useful than just printing stuff to the screen...
    '''
    try:
        val = float(input)
        print('Yup...')
    except ValueError:
        print("Not a float!")

a = 'hamburger'
b = 123.45

number_test(b)



