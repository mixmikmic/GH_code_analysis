get_ipython().magic('matplotlib inline')

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm

# Use the following only if you are on a high definition device
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

def percentile(data, p):
    """Returns the value of the p-ith percentile for the given dataset
    
    Taken from this paper: http://stanford.io/2bk0k8u
    
    """
    data = sorted(data)

    if p >= 100:
        return data[-1]

    n = len(data)
    i = (n*p)/100.0 + 0.5
    
    # If the index is an interger value, return the corresponding
    # value in the dataset as x. Otherwise, interoplate x.
    if int(i) == i:
        i = int(i) - 1 if int(i) - 1 > 0 else int(i)
        x = data[i]
    else:
        k = int(i) - 1 if int(i) - 1 > 0 else int(i)
        f = i - int(i)
        x = (1 - f)*data[k] + f*data[k + 1]
    
    return x

test_data = [5, 1, 9, 3, 14, 9, 7]
for i in range(1, 101):
    print(percentile(test_data, i))

percentile(test_data, 100)

def qqplot(data):
    x = [norm.ppf(p) for p in np.arange(0.0, 1.01, 0.01)]
    y = [percentile(data, p) for p in np.arange(0, 101, 1)]
    plt.scatter(x, y)
    
    qqline(data)
    
    # Add the regression line
#     y = np.percentile(data, 25), np.percentile(data, 75)
#     x = 

qqplot(test_data)

def qqline(data, step=0.01):
    # Calculate the slope and intercept of the...
    y = np.array([np.percentile(data, 25), np.percentile(data, 75)])
    x = np.array([norm.ppf(0.25), norm.ppf(0.75)])
    m = np.diff(y)/np.diff(x)
    b = y[0] - m * x[0]

    # Plot the line with a generated set of x values
    x = np.array([norm.ppf(p) for p in np.arange(0.0, 1.01, step)])
    #x = np.array([norm.ppf(percentileofscore(data, s) * 100) for s in sorted(data)])
    #m, b = np.polyfit(x, data, 1)
    plt.plot(x, x * m + b, 'black')
    

def qqplot(data):
#     x = [norm.ppf(0.00000001)] + \
#         [norm.ppf(p) for p in np.arange(0.01, 1.0, 0.01)] + \
#         [norm.ppf(0.99999999)]
    x = [norm.ppf(p) for p in np.arange(0.0, 1.01, 0.01)]

    y = [np.percentile(data, p * 100) for p in np.arange(0, 1.01, 0.01)]
    plt.scatter(x, y)
    
    qqline(data)
    
    # Add the regression line
#     y = np.percentile(data, 25), np.percentile(data, 75)
#     x = 

qqplot(test_data)

qqplot(np.random.randn(100000))

def qqplot(data):
    sample = np.random.randn(100000)
    x = [np.percentile(sample, p * 100) for p in np.arange(0, 1.01, 0.01)]
    y = [np.percentile(data, p * 100) for p in np.arange(0, 1.01, 0.01)]
    plt.scatter(x, y)

qqplot(test_data)

qqplot(np.random.randn(100000))

from scipy.stats import probplot

probplot(test_data, plot=plt);

