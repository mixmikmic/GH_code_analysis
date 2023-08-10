get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')

import pandas as pd
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


def fit_line(data, error_func):
    l = np.float32([0,np.mean(data[:,1])])
    
    # Plot initial guess (optional)
    x_ends = np.float32([-5,5])
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth=2.0, label="Initial guess")
    # plt.show()
    
    # Call optimizer to minimize error function
    result = spo.minimize(error_func, l, args=(data,), method="SLSQP", options={'disp':True})
    return result.x


def error(line, data):
    err = np.sum((data[:,1] - (line[0] * data[:,0] + line[1]))**2)
    return err


def run_optimization_2():

    # Define original line
    l_orig = np.float32([4,2])
    print "Original line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1])
    Xorig = np.linspace(0,10,12)
    Yorig = l_orig[0] * Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, "b--", linewidth=2.0, label="Original line")
    # plt.show()
    
    # Generate noisy data
    noise_sigma = 3.0
    noise = np.random.normal(0,noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:,0], data[:,1], 'go', label = "Data points")
    # plt.show()
    
    #try to fit a line to this data
    l_fit = fit_line(data, error)
    print "Fitted line: C0 = {}, C1 = {}".format(l_fit[0], l_fit[1])
    plt.plot(data[:,0], l_fit[0] * data[:,0] + l_fit[1], 'r--', linewidth=2.0, label="Fitted line")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.show()
    
    # add legend and show plot

    
if __name__ == "__main__":
    run_optimization_2()

