import numpy as np
import matplotlib.pylab as plt
import padasip as pa

get_ipython().magic('matplotlib inline')
plt.style.use('ggplot') # nicer plots
np.random.seed(52102) # always use the same random seed to make results comparable

def measure_x():
    # input vector of size 3
    x = np.random.random(3)
    return x

def measure_d(x):
    # meausure system output
    d = 2*x[0] + 1*x[1] - 1.5*x[2]
    return d
    

filt = pa.filters.FilterNLMS(3, mu=1.)

N = 100
log_d = np.zeros(N)
log_y = np.zeros(N)

for k in range(N):
    # measure input
    x = measure_x()
    
    # predict new value
    y = filt.predict(x)
    
    # do the important stuff with prediction output
    pass    
    
    # measure output
    d = measure_d(x)
    
    # update filter
    filt.adapt(d, x)
    
    # log values
    log_d[k] = d
    log_y[k] = y

plt.figure(figsize=(12.5,6))
plt.plot(log_d, "b", label="target")
plt.plot(log_y, "g", label="prediction")
plt.xlabel("discrete time index [k]")
plt.legend()
plt.tight_layout()
plt.show()

