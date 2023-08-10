from collections import defaultdict
from IPython.display import Image
from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial.hermite import Hermite
from numpy.polynomial.legendre import Legendre
#from numpy.polynomial. import 
import os
import pandas as pd

get_ipython().magic('matplotlib inline')

PATH_TRAIN = '/home/jpeacock29/TeamDarWin-darwin-cluster/Random_sampler_of_images/labeled_samples.csv'
PATH_TRAIN_CSV = '/data/amnh/darwin/samples/image_csvs/'
PATH_ALL_CSV = '/data/amnh/darwin/image_csvs_clean/'

def plot_example_fit(_i, _polynomial_type, _degree):
    
    for i, edge_file_name in enumerate(os.listdir(PATH_ALL_CSV)):

        if i == _i:
            example_edge = pd.read_csv(PATH_ALL_CSV + edge_file_name, names=['x', 'y'])
            x = example_edge.x.values
            y = example_edge.y.values
            y_fit, sse = polynomial_fit(x, y, _polynomial_type, _degree)
            
            plt.plot(x, y)
            plt.plot(x, y_fit)
            
            plt.show()
            
            print(sse)
            print(edge_file_name)
            print(y_fit[0:10])
            
            break

def polynomial_fit(x, y, polynomial_type, degree):
    """Fit x and y coordinates with polynomial_type of degree."""
    
    # fit the x and y data, returing a new polynomial object and a report on the fit
    fit_polynomial, fit_report = polynomial_type.fit(x, y, degree, full=True)
    
    # predict y values using the fit coefficients
    y_fit = fit_polynomial(x)
    
    # extract sse from list of additonal attributes
    sum_squared_errors = fit_report[0]
    
    return y_fit, sum_squared_errors

def median_SSE_vs_polynomial_degree(path, _polynomial_type, max_degree):

    polynomial_degree__SSEs = defaultdict(list)
    
    for i, edge_file_name in enumerate(os.listdir(path)):
        
        # load edge
        example_edge = pd.read_csv(path + edge_file_name, names=['x', 'y'])
        x = example_edge.x.values
        y = example_edge.y.values
            
        for _degree in range(15):
        
            y_fit, sse = polynomial_fit(x, y, _polynomial_type, _degree)
            
            polynomial_degree__SSEs[_degree].append(sse)
    
    return polynomial_degree__SSEs

def plot_median_SSE_vs_polynomial_degree(_polynomial_degree__SSEs):
   
    plt.semilogy()
    bp = plt.boxplot(list(_polynomial_degree__SSEs.values()))
    
    plt.setp(bp['whiskers'], color='black', linestyle = 'solid')
    plt.setp(bp['fliers'], alpha = 0.5, marker= 'o', markersize = 3)
    
    plt.xlabel('Polynomial degree')
    plt.ylabel('Median SSE')

plot_median_SSE_vs_polynomial_degree(median_SSE_vs_polynomial_degree(PATH_TRAIN_CSV, Hermite, 15))

plot_median_SSE_vs_polynomial_degree(median_SSE_vs_polynomial_degree(PATH_TRAIN_CSV, Legendre, 15))

plot_example_fit(10, Hermite, 3)
plot_example_fit(10, Legendre, 3)

edge_train = pd.read_csv(PATH_TRAIN)

edge_train.head(3)

edge_train.has_north_edge.mean(), edge_train.has_south_edge.mean()

Image(filename='/data/amnh/darwin/segmentations/MS-DAR-00058-00001-000-00035_largest_component.png');

