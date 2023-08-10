#library/config we need
get_ipython().magic('matplotlib inline')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plot

#will allow us to get necessary information from running
numiters = 0
import timeit #timeit.default_timer() times runtime
#import memory_profiler 

start = timeit.default_timer()

# this will print all of our arguments in reverse
def reverse(*args):
    
    if not args[0]:
        return
    
    print args[0][-1] #prints last word
    reverse(args[0][:-1]) #calls itself with one less word
    
reverse(['Bob', 'is', 'name', 'my', 'Hello,'])

stop = timeit.default_timer()

print "Time =",stop-start

def map_print(apple="",**kwargs):
    
    for key in kwargs:
        print kwargs[key]
        
test_dictionary = {"a":"b"}
dictionary={"First":"Edward","Last":"Nusinovich","Middle":"Alexander"}
map_print(**dictionary)
    

# initial setup
get_ipython().magic('matplotlib inline')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plot
import memory_profiler

from scipy.optimize import minimize

import timeit

#all methods to minimize
methods = ['Nelder-Mead','Powell','CG','BFGS','Newton-CG','L-BFGS-B','TNC','COBYLA','SLSQP','dogleg','trust-ncg']

start = np.zeros(0)
stop = np.zeros(0)
num_iters = np.zeros(0)

#runtime code goes here

function = """thefunction()""" #the function we're testing will go here

#testing every minimization method
for method in methods:
    
    start = np.append(start,timeit.default_timer())
    
    guess = [] # add an initial guess
    """
    
    Minimization testing code goes here

    """
    
    
    " scipy.optimize.OptimizeResult contains the number iterations "
    result = minimize(thefunction(),params)
    
    
    stop = np.append(stop,timeit.default_timer())
    


exec_time = stop-start

counter = 0

#could print all of the runtimes as they run but it would be better to print them at the end and store runtimes
for method in methods:
    
    print '{0} took {1} seconds to minimize this function. The result, {2} was found at {3}'.format(method,exec_time[counter],result.x,result.fun)
    counter = counter + 1
    

# Using the format method
#exec_time = 5
#print 'This has taken {} seconds to execute'.format(exec_time)

#print 'Anybody know where {1} {0} is?'.format("Doe","John")

for val in np.arange(1,len(methods)):
    start = np.append(start,timeit.default_timer())
    stop = np.append(stop,timeit.default_timer())

#two numpy arrays can be subtracted from one another and will be subtracted element-wise
print 'This has taken {0} seconds to execute'.format(stop-start)




