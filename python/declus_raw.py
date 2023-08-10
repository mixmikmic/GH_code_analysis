#general imports
import matplotlib.pyplot as plt   
import pygslib  
import numpy as np

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

#Check the data is ok
a=mydata['Primary'].isnull()
print "Undefined values:", len(a[a==True])
print "Minimum value   :", mydata['Primary'].min()
print "Maximum value   :", mydata['Primary'].max()

parameters_declus = { 
        'x'      :  mydata['Xlocation'],  # data x coordinates, array('f') with bounds (na), na is number of data points
        'y'      :  mydata['Ylocation'],  # data y coordinates, array('f') with bounds (na)
        'z'      :  mydata['Zlocation'],  # data z coordinates, array('f') with bounds (na)
        'vr'     :  mydata['Primary'],    # variable, array('f') with bounds (na)
        'anisy'  :  1.,                   # Y cell anisotropy (Ysize=size*Yanis), 'f' 
        'anisz'  :  1.,                   # Z cell anisotropy (Zsize=size*Zanis), 'f' 
        'minmax' :  0,                    # 0=look for minimum declustered mean (1=max), 'i' 
        'ncell'  :  24,                   # number of cell sizes, 'i' 
        'cmin'   :  1.,                   # minimum cell sizes, 'i' 
        'cmax'   :  25.,                   # maximum cell sizes, 'i'. Will be update to cmin if ncell == 1
        'noff'   :  5,                    # number of origin offsets, 'i'. This is to avoid local minima/maxima
        'maxcel' :  100000}               # maximum number of cells, 'i'. This is to avoid large calculations, if MAXCEL<1 this check will be ignored


wtopt,vrop,wtmin,wtmax,error,xinc,yinc,zinc,rxcs,rycs,rzcs,rvrcr = pygslib.gslib.declus(parameters_declus)

# to know what the output means print the help
help(pygslib.gslib.declus)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.figure(num=None, figsize=(10, 10), dpi=200, facecolor='w', edgecolor='k')

#view data in a 2D projection
plt.scatter(mydata['Xlocation'],mydata['Ylocation'], c=wtopt, s=wtopt*320, alpha=0.5)
plt.plot(mydata['Xlocation'],mydata['Ylocation'], '.', color='k')
l=plt.colorbar()
l.set_label('Declustering weight')
plt.grid(True)
plt.show()


#The declustering size
plt.plot (rxcs, rvrcr)

rvrcr

print '========================================='
print 'declustered mean     :',  vrop
print 'weight minimum       :',  wtmin
print 'weight maximum       :',  wtmax
print 'runtime error        :',  error
print 'cell size increments :',  xinc,yinc,zinc
print 'sum of weight        :',  np.sum(wtopt)
print 'n data               :',  len(wtopt)
print '========================================='

parameters_declus = { 
        'x'      :  mydata['Xlocation'],  # data x coordinates, array('f') with bounds (na), na is number of data points
        'y'      :  mydata['Ylocation'],  # data y coordinates, array('f') with bounds (na)
        'z'      :  mydata['Zlocation'],  # data z coordinates, array('f') with bounds (na)
        'vr'     :  mydata['Primary'],    # variable, array('f') with bounds (na)
        'anisy'  :  1.,                   # Y cell anisotropy (Ysize=size*Yanis), 'f' 
        'anisz'  :  1.,                   # Z cell anisotropy (Zsize=size*Zanis), 'f' 
        'minmax' :  0,                    # 0=look for minimum declustered mean (1=max), 'i' 
        'ncell'  :  1,                   # number of cell sizes, 'i' 
        'cmin'   :  5.,                   # minimum cell sizes, 'i' 
        'cmax'   :  5.,                   # maximum cell sizes, 'i'. Will be update to cmin if ncell == 1
        'noff'   :  5,                    # number of origin offsets, 'i'. This is to avoid local minima/maxima
        'maxcel' :  100000}               # maximum number of cells, 'i'. This is to avoid large calculations, if MAXCEL<1 this check will be ignored


wtopt,vrop,wtmin,wtmax,error,xinc,yinc,zinc,rxcs,rycs,rzcs,rvrcr = pygslib.gslib.declus(parameters_declus)

print '========================================='
print 'declustered mean     :',  vrop
print 'weight minimum       :',  wtmin
print 'weight maximum       :',  wtmax
print 'runtime error        :',  error
print 'cell size increments :',  xinc,yinc,zinc
print 'sum of weight        :',  np.sum(wtopt)
print 'n data               :',  len(wtopt)
print '========================================='





