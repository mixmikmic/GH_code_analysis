# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

from datetime import datetime

pd.options.display.max_rows = 100

#*********************************************************
#Two datasets are available: a training set and a test set. 
#We'll be using the training set to build our predictive model 
#and the testing set to score it and generate an output file to 
#submit on the Kaggle evaluation system.

# load the basic stock market value dataset, set date (col 0) as index
rawdata = pd.read_csv('stocks-us-adjClose.csv', index_col=0)

# convert date strings to datetimeindex
rawdata.index = pd.DatetimeIndex(pd.to_datetime(rawdata.index))

# look at what we've got:
#data.shape
#data.head()
#data.describe()

# Choose a start date. 
# We need to have a set of stocks that start on or before this date to ensure a complete time series.
# We pick a date that is as early as possible, but includes a good number of stocks.
startdate = datetime(2000, 1, 1)

# take only stock values after the start date
# make a copy of raw data. We will transform this new copy
xfdata = rawdata.loc[rawdata.index >= startdate].copy()

# log transform the stock values 
xfdata = np.log(xfdata)

# delete any stocks with NaN (missing) stock values after the start date
for col in xfdata:
    if any(pd.isnull(xfdata[col])):
        xfdata = xfdata.drop(col,axis=1)

# add any missing dates and use interpolation to estimate the missing days' stock values 
idx = pd.date_range(xfdata.index.min(), xfdata.index.max())
xfdata = xfdata.reindex(idx)
xfdata = xfdata.apply(pd.Series.interpolate, method="cubic")

print("from original dataset of %d stocks and %s days, we are using %d stocks and %d days"%
      (rawdata.shape[1],rawdata.shape[0],xfdata.shape[1],xfdata.shape[0]))

# Generate mean log stock value data to use for training / testing.
# I am using log data for model target output because I think that model fitting 
#  will work better with linearized data.
# Note that we can then convert model output to real stock values using np.exp(stock_mean).
stock_mean = xfdata.mean(axis=1)

# scale each stock to range between 0-1
for col in xfdata:
    valmin = xfdata[col].min()
    valmax = xfdata[col].max()
    xfdata[col] = (xfdata[col]-valmin)/(valmax-valmin)

# set True to plot some log xformed stock data to see what the interpolation did:
if True:
    plt.figure(figsize=(20, 10))
    numdays = xfdata.shape[0] 
    #numdays = 60
    ax1 = plt.plot(xfdata.index[:numdays] ,xfdata.iloc[:numdays,0],"b-")
    ax2 = plt.plot(xfdata.index[:numdays] ,xfdata.iloc[:numdays,1],"g-")
    ax3 = plt.plot(xfdata.index[:numdays] ,xfdata.iloc[:numdays,-1],"m-")      

# Use a difference fn to remove the growth trend and leave the day to day stock variation.
# This is necessary to do before correlation/covariance, etc operations.
diffdata = pd.DataFrame(np.diff(xfdata.values,axis=0),index=xfdata.index[:-1], columns=xfdata.columns)

# set True to plot some differenced stock data:
if False:
    plt.figure(figsize=(20, 10))
    numdays = diffdata.shape[0] 
    #numdays = 60
    ax1 = plt.plot(diffdata.index[:numdays] ,diffdata.iloc[:numdays,0],"b-")
    ax2 = plt.plot(diffdata.index[:numdays] ,diffdata.iloc[:numdays,1],"g-")
    ax3 = plt.plot(diffdata.index[:numdays] ,diffdata.iloc[:numdays,-1],"m-")

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold

# standardize data for model
X = diffdata.copy()
X /= X.std(axis=0)

# Sparse inverse covariance
# note: this is used in the scikit learn example, but it doesn't create a useful matrix.
#edge_model = covariance.GraphLassoCV(verbose=True).fit(X)

# Empirical covariance
edge_model = covariance.EmpiricalCovariance().fit(X)

# set True to show the covariance matrix result
if False:
    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax = ax1.matshow(edge_model.covariance_)

# #############################################################################
# Find a low-dimension embedding for visualization and cluster analysis: 
# find the best position of the nodes (the stocks) on a 2D plane

# alternate mapping models:
#node_position_model = manifold.LocallyLinearEmbedding(
#    n_components=2, eigen_solver='dense', n_neighbors=6)
#node_position_model = manifold.MDS()
#node_position_model = manifold.SpectralEmbedding()

# this seems to work best
# note that the default # dimensions are 2 - you could have more if you want
node_position_model = manifold.TSNE()

# get the 2D coordinates we need
coords = node_position_model.fit_transform(edge_model.covariance_)

from sklearn.cluster import KMeans

names = X.columns
symbols = X.columns

numclusters = 10
kmeans = KMeans(n_clusters=numclusters).fit(coords)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

n_labels = labels.max()

for i in range(n_labels + 1):
    print('Cluster %i: contains %d stocks: %s\n' % 
          ((i + 1), len((names[labels == i])), ', '.join(names[labels == i])))

def plot_stock_map(coords, cluster_labels, symbols, centroids=[None], show_symbols=False):

    # #############################################################################
    # Visualization
    plt.figure(1, facecolor='w', figsize=(15, 10))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')
    
    # Plot the nodes 
    plt.scatter(coords[:,0], coords[:,1], s=100, c=cluster_labels, cmap=plt.cm.spectral)
    
    # plot centroids, if they were passed
    if len(centroids) > 0:
        plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=200, linewidths = 5, zorder = 10)

    # Add a stock symbol label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    if show_symbols:
        for index, (symbol, label, (x, y)) in enumerate(
                zip(symbols, cluster_labels, coords)):

            dx = x - coords[:,0]
            dx[index] = 1
            dy = y - coords[:,1]
            dy[index] = 1
            this_dx = dx[np.argmin(np.abs(dy))]
            this_dy = dy[np.argmin(np.abs(dx))]
            if this_dx > 0:
                horizontalalignment = 'left'
                x = x + .002
            else:
                horizontalalignment = 'right'
                x = x - .002
            if this_dy > 0:
                verticalalignment = 'bottom'
                y = y + .002
            else:
                verticalalignment = 'top'
                y = y - .002
            #plt.text(x, y, name, size=10,
            plt.text(x, y, symbol, size=15,
                     horizontalalignment=horizontalalignment,
                     verticalalignment=verticalalignment,
                     bbox=dict(facecolor='w',
                     #          edgecolor=plt.cm.spectral(label / float(n_labels)),
                               edgecolor='w',
                               alpha=.6)
                    )

    plt.xlim(coords[:,0].min() - .05 * coords[:,0].ptp(),
             coords[:,0].max() + .05 * coords[:,0].ptp(),)
    plt.ylim(coords[:,1].min() - .03 * coords[:,1].ptp(),
             coords[:,1].max() + .03 * coords[:,1].ptp())

    plt.show()

plot_stock_map(coords, labels, symbols, centroids, show_symbols=False)

plot_stock_map(coords, labels, symbols, centroids, show_symbols=True)

from scipy.spatial import distance

labelnums = set(labels)
numclusters =  len(labelnums)
clusternames = list(map(str, labelnums))
serieslen = stock_mean.shape[0]

# create a new dataframe with same time index as original data, and train/test output 
# Use cluster # as cluster name for ease of analysis
cdf = pd.DataFrame(index=stock_mean.index, columns=clusternames)

# for each cluster, 
#  generate a list of stock symbols and a weight for each stock based on distance from centroid
for labelnum in labelnums:
    
    # get indices to stocks in this cluster
    index, = np.where(labels==labelnum)
    
    # number of stocks in this cluster
    numstocks = index.shape[0]

    # test to verify getting cluster data right: 
    #   choose random stock indices - should be greater distance values than cluster selections
    #arr = np.arange(labels.shape[0])
    #np.random.shuffle(arr)
    #index = arr[:numstocks]
    
    # get list of symbols of stocks that belong to this cluster
    symbols = X.columns[index]
    print("cluster %d stocks: %s"%(labelnum,",".join(symbols)))
    
    # Generate weights for each stock in this cluster, based on euclidean distance to centroid.
    # First get distances
    dist = np.zeros((numstocks,))
    for j, stockindex in zip(range(numstocks),index):
        dist[j] = distance.euclidean(coords[stockindex,:],centroids[labelnum, :])
    weight = dist.max() - dist + (dist.max()-dist.min())/5

    # create a time series to contain the feature values for this cluster
    ps = pd.Series(np.zeros((serieslen,)), index=xfdata.index)
    
    # calculate the feature values
    weightsum = weight.sum()
    for i in range(serieslen):
        ps[i] = (xfdata[symbols].iloc[i,:] * weight).sum() / weightsum
    
    print("cluster %d time series: min=%1.3f, max=%1.3f, mean=%1.3f\n"%(labelnum, ps.min(),ps.max(),ps.mean()))
    
    # add the new time series to the feature dataset
    cdf[clusternames[labelnum]] = ps

cdf.plot( y=clusternames, kind="line", figsize=(20,10))

