get_ipython().magic('matplotlib inline')

from irreversible_stressstrain import StressStrain as strainmodel
import test_suite as suite
import graph_suite as plot
import numpy as np
reload(suite)

xtitle = 'Strain'
ytitle = 'Stress'

name1 = 'ref/HSRS/22'
name2 = 'ref/HSRS/222'
name3 = 'ref/HSRS/326'
name4 = 'kolskybar.xml'

data1 = strainmodel(name1).get_experimental_data()
data2 = strainmodel(name2).get_experimental_data()
data3 = strainmodel(name3).get_experimental_data()
data4 = strainmodel(name4, type='xml').get_experimental_data()[:,:2]

yield1 = suite.yield_stress(data1)
yield2 = suite.yield_stress(data2)
yield3 = suite.yield_stress(data3,cutoff = 1) 
yield4 = suite.yield_stress(data4, decreasingend=True)

plot.plotmult2D(data1,yield1, title=name1, xtitle=xtitle, ytitle=ytitle)
plot.plotmult2D(data2,yield2, title=name2, xtitle=xtitle, ytitle=ytitle)
plot.plotmult2D(data3,yield3, title=name3, xtitle=xtitle, ytitle=ytitle)
plot.plotmult2D(data4,yield4, title=name4, xtitle=xtitle, ytitle=ytitle)

from irreversible_stressstrain import StressStrain as strainmodel
import test_suite as suite
import graph_suite as plot
import numpy as np
from scipy.optimize import curve_fit

# finds the yield stress of a dataset automatically
def yield_stress(model, numpoints=1000, cutoff=0.025, startx=None, endx=None):

    """Default interval values"""
    if startx is None:
        startx=min(model[:,0])+0.1

    if endx is None:
        endx=max(model[:,0])

    model = suite.delete_noise(model,cutoff=cutoff)
    model = suite.adjust(model)

    """a and c are parameters"""
    def fit(x, a,c):
        return a*np.log(x)+c

    strain = model[:,0]
    stress = model[:,1]

    optimal_params, cov_matrix = curve_fit(fit,strain,stress)

    a, c = optimal_params

    """The fitted version of the dataset"""
    def bestfit(x):
        return a*np.log(x)+c

    """We look for the place where the slope is average over the dataset"""
    gap_len = (endx-startx)/numpoints

    xs = np.linspace(startx,endx,numpoints)
    ys = bestfit(xs)

    pred_data = suite.combine_data(xs,ys)
    pred_slope = suite.get_slopes(pred_data)

    """Defining average slope by observing clusters in data"""
    #TODO COULD BE BECAUSE DATA4 IS THE REFERENCE HERE
    left, right = suite.kmeanssplit(data4)
    leftmid, rightmid = suite.midpoint(left)[None,:], suite.midpoint(right)[None,:]
    
    ave_slope = (rightmid[0,1]-leftmid[0,1])/(rightmid[0,0]-leftmid[0,0])

    
    """As soon as the slope at a point is less than the average slope, we stop"""
    for ind, slope in enumerate(pred_slope):

        if slope<ave_slope:
            break

    """
    We must take into account that we may not have a 
    data point in the experimental set where we have found a result
    """
    datapointind = ind*gap_len

    for ind, stra in enumerate(model[:,0]):

        if stra > datapointind:
            return model[ind][None,:]

    raise ValueError("The data does not seem to have a yield")
    
data1 = strainmodel('ref/HSRS/22').get_experimental_data()
data2 = strainmodel('ref/HSRS/222').get_experimental_data()
data3 = strainmodel('ref/HSRS/326').get_experimental_data()
data4 = strainmodel('kolskybar.xml',type='xml').get_experimental_data()[:,:2] # got rid of strain rate info
 
# cutoff depends on noise of data, 1 seems to work well for these measurements
yield1 = yield_stress(data1,cutoff=1.)
yield2 = yield_stress(data2)
yield3 = yield_stress(data3,cutoff=1.) 
yield4 = yield_stress(data4)

from irreversible_stressstrain import StressStrain as strainmodel
import test_suite as suite
import graph_suite as plot
import numpy as np

# finds the yield stress of a dataset automatically
def yield_stress(model, numpoints=1000, cutoff=0.025, startx=None, endx=None):

    """Default interval values"""
    if startx is None:
        startx=min(model[:,0])+0.1

    if endx is None:
        endx=max(model[:,0])

    model = suite.delete_noise(model,cutoff=cutoff)
    model = suite.adjust(model)

    """a and c are parameters"""
    def fit(x, a,c):
        return a*np.log(x)+c

    strain = model[:,0]
    stress = model[:,1]

    optimal_params, cov_matrix = curve_fit(fit,strain,stress)

    a, c = optimal_params

    """The fitted version of the dataset"""
    def bestfit(x):
        return a*np.log(x)+c

    """We look for the place where the slope is average over the dataset"""
    gap_len = (endx-startx)/numpoints

    xs = np.linspace(startx,endx,numpoints)
    ys = bestfit(xs)

    pred_data = suite.combine_data(xs,ys)
    pred_slope = suite.get_slopes(pred_data)

    """Defining average slope by observing clusters in data"""
    #TODO THIS RUINS THE SPLIT, BECAUSE SPLITTING BY THESE CRITERIA DOESN'T REPRESENTATIVELY SPLIT
    left, right = suite.kmeanssplit(model)
    leftmid, rightmid = suite.midpoint(left)[None,:], suite.midpoint(right)[None,:]
    
    ave_slope = (rightmid[0,1]-leftmid[0,1])/(rightmid[0,0]-leftmid[0,0])

    
    """As soon as the slope at a point is less than the average slope, we stop"""
    for ind, slope in enumerate(pred_slope):

        if slope<ave_slope:
            break

    """
    We must take into account that we may not have a 
    data point in the experimental set where we have found a result
    """
    datapointind = ind*gap_len

    for ind, stra in enumerate(model[:,0]):

        if stra > datapointind:
            return model[ind][None,:]

    raise ValueError("The data does not seem to have a yield")
    
data1 = strainmodel('ref/HSRS/22').get_experimental_data()
data2 = strainmodel('ref/HSRS/222').get_experimental_data()
data3 = strainmodel('ref/HSRS/326').get_experimental_data()
data4 = strainmodel('kolskybar.xml',type='xml').get_experimental_data()[:,:2] # got rid of strain rate info
 
# cutoff depends on noise of data, 1 seems to work well for these measurements
yield1 = yield_stress(data1,cutoff=1.)
yield2 = yield_stress(data2)
yield3 = yield_stress(data3,cutoff=1.) 
yield4 = yield_stress(data4)

##CHECKING SAMPLE POINTS
reload(plot)
model = data3

left, right = suite.kmeanssplit(model)
leftmid, rightmid = suite.midpoint(left)[None,:], suite.midpoint(right)[None,:]

plot.plot2D(model)
plot.plotmult2D(left,leftmid,marker1='bo')
plot.plotmult2D(right,rightmid,marker1='bo')

reload(suite)
xmldata = strainmodel('kolskybar.xml','xml').get_experimental_data()
plot.plotmult2D(xmldata,suite.yield_stress(xmldata,cutoff=0,decreasingend=True))

from material_analytics import combine_data
import numpy as np

def samplepoints(function, interval, numpoints):
    """Given a function and an interval (two-element list) and a number of points, applies it to the function and gets sample points at even intervals."""

    x_dom = np.linspace(interval[0],interval[1],numpoints)
    return combine_data(x_dom,function(x_dom))

samplepoints(lambda x: x**2,[0.,10.],11)

import graph_suite as plot
from irreversible_stressstrain import StressStrain as strainmodel

"""
Material Analytics
******************

Contains all functionality needed to
**automatically determine the yield stress**
of a material (see *yield_stress()*), even with noisy data, given
a stress-strain curve in the form
[Strain|Stress] in each row.
"""

"""Basic libs"""
import numpy as np
import math
from DataModelDict import DataModelDict as dmd

"""For optimization and model training"""
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.cluster import MiniBatchKMeans as mbkmeans
from sklearn.cluster import KMeans

def yield_stress(model, numpoints=1000, cutoff=0.025, startx=None, endx=None, decreasingend=False):
    """
    Finds the yield stress of a dataset **automatically** using kmeans clustering and covariance analysis.

    In order to use this function, you just need to provide it with a stress/strain curve as a numpy array
    where the data is formatted as a two-column array. The first column is all of the stress values,
    and the second column is all of the strain values.

    This works by fitting a logarithmic model as closely as possible to the experimental data (to reduce noise)
    and then to analyze where the slope begins to be decrease relative to the average slope. In other words,
    where :math:`\partial \sigma/ \partial \epsilon < (f(b)-f(a))/(b-a)` where a and b are the beginning and
    end of the interval, respectively.
    """

    """Default interval values"""
    if startx is None:
        startx=min(model[:,0])+0.1
    if endx is None:
        endx=max(model[:,0])

    """We get rid of the noise in the data, and select only positive values (so that logarithms can be taken)"""
    model = delete_noise(model,cutoff=cutoff)
    model = adjust(model)

    """a,b, and c are parameters"""
    def fit(x, a,b,c):
        return a*np.log(x)/np.log(b)+c

    strain = model[:,0]
    stress = model[:,1]

    """We are fitting a logarithmic curve as closely as possible to the dataset"""
    optimal_params, cov_matrix = curve_fit(fit,strain,stress,bounds=([0,0,0],[10000000,10000000,100000000]))
    #optimal_params, cov_matrix = curve_fit(fit,strain,stress,method='dogbox',bounds=(0, [3., 2., 1.]))
    a,b, c = optimal_params

    """The fitted version of the dataset"""
    def bestfit(x):
        return a*np.log(x)/np.log(b)+c

    """
    We look for the place where the slope is average over
    the domain by taking sample points of the logarithmic curve
    """
    gap_len = (endx-startx)/numpoints

    xs = np.linspace(startx,endx,numpoints)
    ys = bestfit(xs)

    pred_data = combine_data(xs,ys)
    pred_slope = get_slopes(pred_data)

    ave_slope = 0

    """If the slope decreases at the end, we cluster the data to avoid referencing the end of the dataset"""
    if decreasingend is True:
        
        """Defining average slope by observing clusters in data"""
        left, right = kmeanssplit(model)
        leftmid, rightmid = midpoint(left)[None,:], midpoint(right)[None,:]
        ave_slope = (rightmid[0,1]-leftmid[0,1])/(rightmid[0,0]-leftmid[0,0])
            
    else:
        """Otherwise, we get the slope over the whole interval to find where slope begins to decrease overall"""    
        ave_slope = (stress[-1]-stress[0])/(strain[-1]-strain[0])
    
    """As soon as the slope at a point is less than the average slope, we stop"""
    for ind, slope in enumerate(pred_slope):
        if slope<ave_slope:
            break

    """
    We must take into account that we may not have a 
    data point in the experimental set where we have found a result,
    so we find the nearest neighbor in our dataset
    """
    datapointind = ind*gap_len

    """Here we find the nearest neighbor in the dataset"""
    for ind, stra in enumerate(model[:,0]):

        if stra > datapointind:
            return model[ind][None,:]

    raise ValueError("The data does not seem to have a yield")     

def delete_noise(model,cutoff = 0.025):
    """
    Takes an array (assuming it's roughly sorted) and returns it after
    a certain value (cutoff), useful for removing early values which may
    contain high levels of noise.
    """

    cur_index = 0

    # deleting noisy values (possible inaccuracies up to .025 by default)
    for index, num in enumerate(model[:,0]):
        
        if num >= cutoff: 
              
            return model[index:]   
             
def adjust(model):
    """
    Written for convenience because
    it makes all values positive in
    order to be able to take logarithms.
    """
    
    for index, num in enumerate(model[:,1]):
        
        if num<=0:
            model[index,1] = 1
        
    return model

def midpoint(lst):
    """
    Returns the value that is halfway through the
    list (index-wise), left midpoint if
    there are an odd number of points.
    """
    
    length = len(lst)
    return lst[int(length)/2]

def kmeanssplit(data, numclusters=2):
    """Clusters the data into groups (k-means) and returns the split data."""
    
    return splitdata(data,kcluster(data,numclusters=numclusters).predict(data[:,0][:,None]))

def splitdata(data, predictions):
    """Takes predictions from kmeans clustering and split the table into two groups."""
    
    initgroup = predictions[0]
    splitgroup = 0

    for index, val in enumerate(predictions):
        
        # as soon as we reach the new group, we have found our dividing point
        if val != initgroup:
            splitgroup = index
            break
        
    """Instead of creating tuples, we create lists"""
    elastic = combine_data(data[:splitgroup,0],data[:splitgroup,1]) 
    plastic = combine_data(data[splitgroup:,0],data[splitgroup:,1])
    
    return elastic, plastic

def predictlinear(data, step = 0.5):
    """Creates a linear model based on data and predicts its values over the domain, returning the predictions."""
    
    est = linfit(data)
    x_pred = np.arange(min(data[:,0]),max(data[:,0]+1), step)
    y_pred = est.predict(x_pred[:,None])
    
    return combine_data(x_pred,y_pred)

def samplepoints(function, interval, numpoints):
    """Given a function and an interval (two-element list) and a number of points, applies it to the function and gets sample points at even intervals."""

    x_dom = np.linspace(interval[0],interval[1],numpoints)
    return combine_data(x_dom,function(x_dom))

def linfit(data, start=None):
    """Fits a linear regression to the data and returns it."""

    return LinearRegression().fit(*format_data(data,start))
    
def kminicluster(data, numclusters, start=None):
    """Clusters the data using mini batch kmeans."""
    
    return mbkmeans(n_clusters=numclusters).fit(*format_data(data,start))
    
def kcluster(data,numclusters,start=None):
    """Clusters the data using r  category=OptimizeWarning)
egular kmeans clustering."""
    
    return KMeans(n_clusters=numclusters).fit(*format_data(data,start))
    
def format_data(data, start=None):
    """This method will put data in the appropriate format for regression (Scikit-Learn)."""
    
    return (expToTrain(data, start),data[start:,1])
    
def combine_data(data1,data2):
    r"""Given two arrays, returns a combined list where each element is :math:`x_i,y_i`."""

    return np.array([list(a) for a in zip(data1,data2)])

def regularize(data):
    """Converts every non-numerical list value to zero which is useful for analysis later."""
    
    for index, val in enumerate(data):
        if math.isinf(val) or math.isnan(val):
            data[index]=0
        
    return data
    
def get_slopes(model):
    """
    Takes the approximate derivative of a two-column dataset by taking slopes between all of the points.

    The data should be formatted
    [x,y] for each row.
    """
    
    strain = model[:,0]
    stress = model[:,1]

    slopes = []

    """Approximating the partial derivatives of stress/strain"""
    for index in xrange(len(stress)-1):

        rise = (stress[index+1]-stress[index])
        run = (strain[index+1]-strain[index])

        if run==0:
            slopes.append(0)

        else:
            slopes.append(rise/run)

    return np.array(slopes)

def expToTrain(exp,start=None):
    """Converts a bunch of individual domain values to lists, because each domain value must be iterable for training data."""
    
    x_train = []
    
    for data in exp[start:,0]:
        x_train.append([data, ])
    
    return x_train

# must bound from above in order to avoid the issue of log of zero
model = strainmodel('ref/HSRS/22').get_experimental_data()
plot.plotmult2D(model,yield_stress(model))

