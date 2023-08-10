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

#view data in a 2D projection
plt.scatter(mydata['Xlocation'],mydata['Ylocation'], c=mydata['Primary'])
plt.colorbar()
plt.grid(True)
plt.show()

print pygslib.gslib.__dist_transf.backtr.__doc__

transin,transout, error = pygslib.gslib.__dist_transf.ns_ttable(mydata['Primary'],mydata['Declustering Weight'])
print 'there was any error?: ', error!=0

mydata['NS_Primary'] = pygslib.gslib.__dist_transf.nscore(mydata['Primary'],transin,transout,getrank=False)

mydata['NS_Primary'].hist(bins=30)

mydata['NS_Primary_BT'],error = pygslib.gslib.__dist_transf.backtr(mydata['NS_Primary'],
                                     transin,transout,
                                     ltail=1,utail=1,ltpar=0,utpar=60,
                                     zmin=0,zmax=60,getrank=False)
print 'there was any error?: ', error!=0, error

mydata[['Primary','NS_Primary_BT']].hist(bins=30)

mydata[['Primary','NS_Primary_BT', 'NS_Primary']].head()



