def simple_linear_regression(X, y):
    '''
    Returns slope and intercept for a simple regression line
    
    inputs- Works best with numpy arrays, though other similar data structures will work fine.
        X - input data
        y - output data
        
    outputs - floats
    '''
    # initial sums
    n = float(len(X))
    sum_x = X.sum()
    sum_y = y.sum()
    sum_xy = (X*y).sum()
    sum_xx = (X**2).sum()
    
    # formula for w0
    slope = (sum_xy - (sum_x*sum_y)/n)/(sum_xx - (sum_x*sum_x)/n)
    
    # formula for w1
    intercept = sum_y/n - slope*(sum_x/n)
    
    return (intercept, slope)

import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
random.seed(199)

# generating some test points
X = np.array(range(10))
y = np.array([random.randint(1, 10) for x in range(10)])

intercept, slope = simple_linear_regression(X, y)

print 'Intercept: %.2f, Slope: %.2f' % (intercept, slope)

def reg_predictions(X, intercept, slope): 
    return ((slope*X) + intercept)

line_x = np.array([x/10. for x in range(100)])
line_y = reg_predictions(line_x, intercept, slope)
plt.plot(X, y, '*', line_x, line_y, '-')

