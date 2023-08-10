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

def constell_time(N):
    ra = np.random.random(N) * 24.
    dec = np.random.random(N) * 180. - 90.
    
    t0 = time.time()
    tmp = solutions.constell_davenport(ra, dec)
    
    t1 = time.time()
    for k in range(N):
        tmp = solutions.constell_christenson(ra[k], dec[k])
    
    t2 = time.time()
    
    out1 = t1-t0
    out2 = t2-t1
    
    return out1, out2

N = np.array([1,2,3,4,5,6,7,10,20,30,100,200,300])

x1 = np.zeros(len(N))
x2 = np.zeros(len(N))

for i in range(len(N)):
    v1,v2 = constell_time(N[i])
    x1[i] = v1
    x2[i] = v2


plt.plot(N, x1, 'ro-')
# plt.plot(N, x2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('# samples')
plt.ylabel('Runtime (s)')

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



