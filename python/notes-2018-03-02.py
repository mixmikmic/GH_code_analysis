import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def central_diff(f,a,M,e):
    '''Compute the central difference formula for f(x) at x=a.
    
    Parameters
    ----------
    f : function of one variable
    a : number where we compute the derivative f'(a)
    M : number giving an upper bound on the absolute value of
        the third derivative of f(x)
    e : number setting the desired accuracy
    
    Returns
    -------
    An approximation f'(a) = (f(a+h) - f(a-h))/2h where
    h = (6*e/M)**0.5 guaranteeing the error is less
    than e.
    '''
    h = (6*e/M)**0.5
    df_a = (f(a+h) - f(a-h))/(2*h)
    return df_a

central_diff(np.cos,np.pi/2,1,10e-5)

central_diff(lambda x : 1/x, 2, 6, 0.001)

