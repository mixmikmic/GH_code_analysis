from __future__ import division
from itertools import izip, count
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
from random import randint
from matplotlib import style
style.use('fivethirtyeight')
get_ipython().magic('matplotlib inline')

# 1. Download sunspot dataset and upload the same to dataset directory
#    Load the sunspot dataset as an Array
get_ipython().system('mkdir -p dataset')
get_ipython().system('wget -c -b http://www-personal.umich.edu/~mejn/cp/data/sunspots.txt -P dataset')
data = loadtxt("dataset/sunspots.txt", float)

# 2. View the data as a table
data_as_frame = pd.DataFrame(data, columns=['Months', 'SunSpots'])
data_as_frame.head()

def moving_average(data, window_size):
    """ Computes moving average using discrete linear convolution of two one dimensional sequences. 
    Args:
    -----
        data (pandas.Series): independent variable 
        window_size (int): rolling window size

    Returns:
    --------
        ndarray of linear convolution
            
    References:
    ------------
    [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
    [2] API Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html     
    """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')

def explain_anomalies_rolling_std(y, window_size, sigma=1.0):
    """ Helps in exploring the anamolies using rolling standard deviation
    Args:
    -----
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation
    
    Returns:
    --------
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value)) 
        containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size)
    avg_list = avg.tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    testing_std = pd.rolling_std(residual, window_size)
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan, 
                                  testing_std_as_df.ix[window_size - 1]).round(3).iloc[:,0].tolist()
    std = np.std(residual)
    return {'stationary standard_deviation': round(std, 3), 
            'anomalies_dict': collections.OrderedDict([(index, y_i) 
                                                       for index, y_i, avg_i, rs_i in izip(count(), 
                                                                                           y, avg_list, rolling_std) 
              if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))])}
              
def explain_anomalies(y, window_size, sigma=1.0):
    """ Helps in exploring the anamolies using stationary standard deviation
    Args:
    -----
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation
    
    Returns:
    --------
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value)) 
        containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size)
    avg_list = avg.tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    std = np.std(residual)
    return {'stationary standard_deviation': round(std, 3), 
            'anomalies_dict': 
            collections.OrderedDict([(index, y_i) for index, y_i, avg_i in izip(count(), y, avg_list) 
              if (y_i > avg_i + (sigma * std)) | (y_i < avg_i - (sigma * std))])}

# This function is repsonsible for displaying how the function performs on the given dataset.
def plot_results(x, y, window_size, title_for_plot, sigma_value=1,
                 text_xlabel="X Axis", text_ylabel="Y Axis", applying_rolling_std=False):
    """ Helps in generating the plot and flagging the anamolies. 
        Supports both moving and stationary standard deviation
    Args:
    -----
        x (pandas.Series): dependent variable
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma_value (int): value for standard deviation
        text_xlabel (str): label for annotating the X Axis
        text_ylabel (str): label for annotatin the Y Axis
    """
    fig = plt.figure(figsize=(15, 8))
    plt.plot(x, y, "k.")
    y_av = moving_average(y, window_size)
    plt.plot(x, y_av, color='green')
    plt.xlim(0, 1000)
    plt.xlabel(text_xlabel)
    plt.ylabel(text_ylabel)
    fig.suptitle(title_for_plot, fontsize=20)
    
    # Query for the anomalies and plot the same
    events = {}
    if applying_rolling_std:
        events = explain_anomalies_rolling_std(y, window_size=window_size, sigma=sigma_value)
    else:
        events = explain_anomalies(y, window_size=window_size, sigma=sigma_value)
            
    x_anomaly = np.fromiter(events['anomalies_dict'].iterkeys(), dtype=int, count=len(events['anomalies_dict']))
    y_anomaly = np.fromiter(events['anomalies_dict'].itervalues(), dtype=float, count=len(events['anomalies_dict']))
    plt.plot(x_anomaly, y_anomaly, "r*", markersize=12)
    
    # add grid and lines and enable the plot
    plt.grid(True)
    plt.show()

x = data_as_frame['Months']
Y = data_as_frame['SunSpots']

# plot the results
plot_results(x, y=Y, window_size=10, title_for_plot="Anomalies in Sun Spots Using Stationary Standard Deviation",
             text_xlabel="Months", sigma_value=3, text_ylabel="Number of Sun Spots")

# Get details about the data flagged as annomalies
events = explain_anomalies(y=Y, window_size=10, sigma=3)

# Display the anomaly dict
print("Information about the anomalies model:{}".format(events))

# Convenience function to add noise
def noise(yval):
    """ Helper function to generate random points """
    np.random.seed(0)
    return 0.2*np.asarray(yval)*np.random.normal(size=len(yval))
    
# Generate a random dataset
def generate_random_dataset(size_of_array=1000, random_state=0):
    """ Helps in generating a random dataset which has a normal distribution
    Args:
    -----
        size_of_array (int): number of data points
        random_state (int): to initialize a random state 

    Returns:
    --------
        a list of data points for dependent variable, pandas.Series of independent variable
    """
    np.random.seed(random_state)
    y = np.random.normal(0, 0.5, size_of_array)
    x = range(0, size_of_array)
    y_new = [y_i + index**((size_of_array - index)/size_of_array) 
             for index, y_i in izip(count(), y)]
    y_new = y_new + noise(y_new)
    return x, pd.Series(y_new)

x1, y1 = generate_random_dataset()
plot_results(x1, y1, window_size=50, title_for_plot="Anomalies in Stock Value Using Stationary Standard Deviation", 
             sigma_value=2, text_xlabel="Time (Days)", text_ylabel="Value (US Dollars)")

plot_results(x1, y1, window_size=50, title_for_plot="Anomalies in Stock Value Using Rolling Standard Deviation", 
             sigma_value=2, text_xlabel="Time (Days)", text_ylabel="Value (US Dollars)", applying_rolling_std=True)

