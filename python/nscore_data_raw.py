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

print pygslib.gslib.__dist_transf.nscore.__doc__

transin,transout, error = pygslib.gslib.__dist_transf.ns_ttable(mydata['Primary'],mydata['Declustering Weight'])
print 'there was any error?: ', error!=0

mydata['NS_Primary'] = pygslib.gslib.__dist_transf.nscore(mydata['Primary'],transin,transout,getrank=False)

mydata['NS_Primary'].hist(bins=30)

mydata['NS_Primary'] = pygslib.gslib.__dist_transf.nscore(mydata['Primary'],transin,transout,getrank=True)

mydata['NS_Primary'].hist(bins=30)





