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

def constell_christenson(ra,dec):
    '''
    This is a function to determine the constellation in which an object is located from its ra and dec
    Written by @hmchristenson 
    
    Parameters
    -------
    ra: float
        Right ascension
    dec: float
        Declination
        
    Returns
    -------
    output: string
        Name of the constellation in which the object is located
    '''
    RAl, RAu, Decl, = np.loadtxt('data/data.txt', delimiter=',', usecols=(0,1,2), unpack=True)
    names = np.loadtxt('data/data.txt', delimiter=',', usecols=(3,), unpack=True, dtype='str')
    
    count = 0

    while(Decl[count] > dec):
        count = count + 1
    dec_low = Decl[count]
    
    while(RAu[count] <= ra):
        count = count + 1
    ra_up = RAu[count]

    while(RAl[count] > ra or RAu[count] < ra):
        count = count + 1 
    ra_low = RAl[count]
       
    output = names[count]
    
    return output

def constell_runtime_test (N):
    RaArray = np.random.random(N)*24
    DecArray = np.random.random(N)*180 - 90
    
    t = np.zeros(2*N)
    
    for i in range(N):
        t[-i] = time.time()
        for j in range(i):
            
            constell_christenson(RaArray[j], DecArray[j])
            
        t[i] = time.time()
    
    T = np.zeros(2*N)
    
    for i in range(N):
        T[i] = t[i] - t[-i]
    
    i = range(N)
    
    plt.plot(i, T[i])

constell_runtime_test(100)



