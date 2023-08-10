#general imports
import matplotlib.pyplot as plt   
import pygslib   
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

#make the plots inline
get_ipython().magic('matplotlib inline')

#get the data in gslib format into a pandas Dataframe
mydata= pygslib.gslib.read_gslib_file('../datasets/cluster.dat')  

# This is a 2D file, in this GSLIB version we require 3D data and drillhole name or domain code
# so, we are adding constant elevation = 0 and a dummy BHID = 1 
mydata['Zlocation']=0
mydata['bhid']=1

# printing to verify results
print ' \n **** 5 first rows in my datafile \n\n  ', mydata.head(n=5)

#view data in a 2D projection
plt.scatter(mydata['Xlocation'],mydata['Ylocation'], c=mydata['Primary'])
plt.colorbar()
plt.grid(True)
plt.show()

print pygslib.gslib.__dist_transf.ns_ttable.__doc__

dtransin,dtransout, error = pygslib.gslib.__dist_transf.ns_ttable(mydata['Primary'],mydata['Declustering Weight'])

dttable= pd.DataFrame({'z': dtransin,'y': dtransout})

print dttable.head(3)
print dttable.tail(3) 
print 'there was any error?: ', error!=0

dttable.hist(bins=30)

transin,transout, error = pygslib.gslib.__dist_transf.ns_ttable(mydata['Primary'],np.ones(len(mydata['Primary'])))

ttable= pd.DataFrame({'z': transin,'y': transout})

print ttable.head(3)
print ttable.tail(3)

ttable.hist(bins=30)

parameters_probplt = {
        'iwt'  : 0,                      #int, 1 use declustering weight
        'va'   : ttable.y,               # array('d') with bounds (nd)
        'wt'   : np.ones(len(ttable.y))} # array('d') with bounds (nd), wight variable (obtained with declust?)

parameters_probpltl =  {
        'iwt'  : 0,                       #int, 1 use declustering weight
        'va'   : dttable.y,               # array('d') with bounds (nd)
        'wt'   : np.ones(len(dttable.y))} # array('d') with bounds (nd), wight variable (obtained with declust?)


binval,cl,xpt025,xlqt,xmed,xuqt,xpt975,xmin,xmax, xcvr,xmen,xvar,error = pygslib.gslib.__plot.probplt(**parameters_probplt)

binvall,cll,xpt025l,xlqtl,xmedl,xuqtl,xpt975l,xminl, xmaxl,xcvrl,xmenl,xvarl,errorl = pygslib.gslib.__plot.probplt(**parameters_probpltl)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot (cl, binval, label = 'gaussian non-declustered')
plt.plot (cll, binvall, label = 'gaussian declustered')
plt.legend(loc=4)
plt.grid(True)
fig.show





