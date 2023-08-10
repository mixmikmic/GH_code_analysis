get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
# see also the "datetime" package
import solutions
from solutions import constell_davenport


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 200

plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

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

#take in the number of coordinates we are supposed to test
def test_algorithms(N): 

    import solutions
    
    ras = np.random.random(N)*24
    decs = np.random.random(N)*180 - 90.
    
    davenport_go = time.time()
    constell_davenport(ras, decs)
    davenport_stop = time.time()
    
    davenport_time = davenport_stop - davenport_go
    
    christenson_go = time.time()
    for i in range(N):
        solutions.constell_christenson(ras[i], decs[i])
        
    christenson_stop = time.time()
    
    christenson_time = christenson_stop - christenson_go
    
    return christenson_time, davenport_time

n_trials = 100
trial = np.ones(n_trials)
christenson_times = np.zeros(n_trials)
davenport_times = np.zeros(n_trials)

for i in range(1,n_trials):
    christenson_times[i], davenport_times[i] = test_algorithms(i)
    trial[i] = i
    
hollyplot, = plt.semilogy(trial, christenson_times, linewidth = 4, color = 'b', label = 'Holly')    
jimplot, = plt.semilogy(trial, davenport_times, linewidth = 4, color = 'r', label = 'Jim')   
xlabel = plt.xlabel('N$_{trials}$') 
ylabel = plt.ylabel('time (sec)')
plt.legend(handles=[hollyplot, jimplot], loc=2)
plt.savefig('ConstellationQuickness.png', bbox_inches='tight')

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

#take in the number of coordinates we are supposed to test
def test_algorithm_agreement(N): 

    import solutions
    
    ras = -np.random.random(N)*24
    decs = -np.random.random(N)*180 - 90.

    #ras = 'nope'
    #decs = 'nope'
    
    correct = np.zeros(N)
    
    for i in range(N):
        jim_solution = solutions.constell_davenport(ras[i], decs[i])
        holly_solution = solutions.constell_christenson(ras[i], decs[i])
        if (holly_solution == jim_solution):
            correct[i] = 1
    
    return sum(correct)/N

print(test_algorithm_agreement(1))



