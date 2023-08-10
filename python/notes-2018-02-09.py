import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def poly(x,y):
    "Compute x + y**2."
    return x + y**2

poly(1,2)

poly(2,1)

def greeting(first_name,last_name,salutation='Hello, '):
    return "{0}{1} {2}!".format(salutation, first_name, last_name)

greeting('Patrick','Walls')

greeting('Walls','Patrick')

greeting('LeBron','James',salutation='I love you ')

import pandas as pd

get_ipython().magic('pinfo pd.read_csv')

def trapz(f,a=0,b=1,N=50):
    '''Approximate integral f(x) from a to b using trapezoid rule.
    
    The trapezoid rule used below approximates the integral \int_a^b f(x) dx
    using the sum: \sum_{k=1}^N (f(x_k) + f(x_{k-1}))(x_k - x_{k-1})
    where x_0 = a, x_1, ... , x_N = b are evenly spaced x_k - x_{k-1} = (b-a)/N.
    
    Parameters
    ----------
    f : vectorized function of a single variable
    a,b : numbers defining the interval of integration [a,b]
    N : integer setting the length of the partition
        
    Returns
    -------
    Approximate value of integral of f(x) from a to b using the trapezoid rule
    with partition of length N.
        
    Examples
    --------
    >>> trapz(np.sin,a=0,b=np.pi/2,N=1000)
    0.99899979417741058
    '''
    x = np.linspace(a,b,N)
    y = f(x)
    Delta_x = (b - a)/N
    integral = 0.5 * Delta_x * (y[1:] + y[:-1]).sum()
    return integral

trapz(np.sin,a=0,b=np.pi/2,N=1000)

