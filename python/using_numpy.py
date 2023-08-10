import numpy as np

np.array([3,4,5])

np.array([[1, 2],[3,4]])

### linearly spaced 1D array
np.linspace(1.,10.,10)

### log spaced 1D array
np.logspace(0.,1.,10)

### 1D array of zeros
np.zeros(5)

### 2D array of zeros
np.zeros((3,3))

x_int = np.logspace(0.,1.,10).astype('int')   # cast array as int
print(x_int)

x_int[1] = 2.34   # 2.34 is cast as int
print(x_int[1])

array_string = np.array(['a','b','c','d'])
array_string.dtype    # 1 character string

array_string[1]='bbbb'   # 'bbbb' is cast on 1 character string
array_string[1]

array_string = np.array(['a','b','c','d'],dtype=np.dtype('S10'))
array_string[1] = 'bbbb'   # 'bbbb' is cast on 10 character string
array_string[1]

x = np.arange(10)

x[-1]   # last element

x[3:6]  # subarray

x[1::2] # stride

x[::-1] # stride

x = np.array([np.arange(10*i,10*i+5) for i in range(5)])
x

print("first column : ", x[:,0])
print("last row     : ", x[-1,:])

b=x[-1,:]   # This is a view not a copy!
b[:] += 1

print(x) # the initial matrix is changed!

# Fancy indexing 
print(x % 2 == 1)

x[x % 2 == 1] = 0
print(x)

x = np.linspace(1, 5, 5) + 4   # 4 is broadcast to 5 element array
x

y = np.zeros((3, 5)) + x   # x is broadcast to (3,5) array
y

#This is for embedding figures in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')         # Fancy style



# Define the function
def poisson_sample_maximum(mu, N, Ntrials):
    """
    Generate a set of Ntrials random variables defined as the maximum of N 
    random Poisson R.V. of mean mu
    """
    res = np.zeros(Ntrials)
    # Do a loop
    for i in range(Ntrials):
        # Generate N random varslues
        Y = np.random.poisson(mu, size=(N))
        # Take the maximum
        res[i] = np.max(Y)
    return res 

mu = 5
N = 10
Ntrials = 10000

get_ipython().run_line_magic('timeit', 'values = poisson_sample_maximum(mu, N, Ntrials)')

### Define a better function
def poisson_sample_maximum_better(mu, N, Ntrials):
    """
    Generate a set of Ntrials random variables defined as the maximum of N 
    random Poisson R.V. of mean mu
    """
    ### Generate N*Ntrials random values in N x Ntrials matrix
    Y = np.random.poisson(mu,size=(N,Ntrials))
    ### Return the maximum in each row
    return np.max(Y,0)
   
mu = 5
N = 10
Ntrials = 10000
    
get_ipython().run_line_magic('timeit', 'values = poisson_sample_maximum_better(mu, N, Ntrials)')

values = poisson_sample_maximum_better(mu,N,Ntrials)

### Make and plot the normalized histogram
### We define the binning ouselves to have bins for each integer
bins = np.arange(0, 10 * mu)
histo = plt.hist(values, bins=bins, normed=True, log=True)

### Now compare to the analytical solution
from scipy.special import gammaincc

### Define a lambda function to compute analytical solution
proba = lambda nv, Nr, mu_p : gammaincc(nv + 1, mu_p) ** Nr - gammaincc(nv, mu_p) ** Nr

x = 0.5 * (bins[:-1] + bins[1:])
y = proba(bins[:-1], N, mu)
plt.plot(x, y)
plt.ylim(1e-6,1)   # restrict y range

### A solution
def replace_square(n):
    sqrt_n = np.sqrt(n)
    return n + (sqrt_n == sqrt_n.astype(int))*(-n + sqrt_n)

print(replace_square(7.0))
print(replace_square(np.arange(26)))

### or using where
def replace_square2(n):
    sqrt_n = np.sqrt(n)
    return np.where(sqrt_n == sqrt_n.astype(int), 
                    sqrt_n, n)
        
print(replace_square2(7.0))       
print(replace_square2(np.arange(26)))







