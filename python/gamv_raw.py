#general imports
import matplotlib.pyplot as plt   
import pygslib                  

#make the plots inline
get_ipython().magic('matplotlib inline')

#get the data in gslib format into a pandas Dataframe
mydata= pygslib.gslib.read_gslib_file('../datasets/cluster.dat')  

# This is a 2D file, in this GSLIB version we require 3D data and drillhole name or domain code
# so, we are adding constant elevation = 0 and a dummy BHID = 1 
mydata['Zlocation']=0.
mydata['bhid']=1.

# printing to verify results
print ' \n **** 5 first rows in my datafile \n\n  ', mydata.head(n=5)

#view data in a 2D projection
plt.scatter(mydata['Xlocation'],mydata['Ylocation'], c=mydata['Primary'])
plt.colorbar()
plt.grid(True)
plt.show()

# these are the parameters we need. Note that at difference of GSLIB this dictionary also stores 
# the actual data (ex, X, Y, etc.). 

#important! python is case sensitive 'bhid' is not equal to 'BHID'

parameters = { 
'x'      :  mydata['Xlocation']   , # X coordinates, array('f') with bounds (nd), nd is number of data points
'y'      :  mydata['Ylocation'],    # Y coordinates, array('f') with bounds (nd)
'z'      :  mydata['Zlocation'],    # Z coordinates, array('f') with bounds (nd)
'bhid'   :  mydata['bhid'],         # bhid for downhole variogram, array('i') with bounds (nd)    
'vr'     :  mydata['Primary'],      # Variables, array('f') with bounds (nd,nv), nv is number of variables
'tmin'   : -1.0e21,                 # trimming limits, float
'tmax'   :  1.0e21,                 # trimming limits, float
'nlag'   :  10,                     # number of lags, int
'xlag'   :  4,                      # lag separation distance, float                
'xltol'  :  2,                      # lag tolerance, float
'azm'    : [0,0,90],                # azimuth, array('f') with bounds (ndir)
'atol'   : [90,22.5,22.5],          # azimuth tolerance, array('f') with bounds (ndir)
'bandwh' : [50,10,10],              # bandwith h, array('f') with bounds (ndir)
'dip'    : [0,0,0],                 # dip, array('f') with bounds (ndir)
'dtol'   : [10,10,10],              # dip tolerance, array('f') with bounds (ndir)
'bandwd' : [10,10,10],              # bandwith d, array('f') with bounds (ndir)
'isill'  : 0,                       # standardize sills? (0=no, 1=yes), int
'sills'  : [100],                   # variance used to std the sills, array('f') with bounds (nv)
'ivtail' : [1,1,1,1,1,1,1],         # tail var., array('i') with bounds (nvarg), nvarg is number of variograms
'ivhead' : [1,1,1,1,1,1,1],         # head var., array('i') with bounds (nvarg)
'ivtype' : [1,3,4,5,6,7,8],         # variogram type, array('i') with bounds (nvarg)
'maxclp' : 50000}                   # maximum number of variogram point cloud to use, input int

'''
Remember this is GSLIB... use this code to define variograms
type 1 = traditional semivariogram
     2 = traditional cross semivariogram
     3 = covariance
     4 = correlogram
     5 = general relative semivariogram
     6 = pairwise relative semivariogram
     7 = semivariogram of logarithms
     8 = semimadogram

'''              

#check the variogram is ok
assert pygslib.gslib.check_gamv_par(parameters)==1 , 'sorry this parameter file is wrong' 

#Now we are ready to calculate the veriogram
pdis,pgam, phm,ptm,phv,ptv,pnump, cldi, cldj, cldg, cldh = pygslib.gslib.gamv(parameters)

nvrg = pdis.shape[0]
ndir = pdis.shape[1]
nlag = pdis.shape[2]-2

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plotting the variogram 1 only
v=1

# in all the directions calculated
for d in range(ndir):
    dip=parameters['dip'][d]
    azm=parameters['azm'][d]
    plt.plot (pdis[v, d, 1:], pgam[v, d, 1:], '-o', label=str(dip) + '-->' + str(azm))

# adding nice features to the plot
plt.legend()
plt.grid(True)
plt.show()

#see if the maximum number of variogram cloud points was reached: 
n_cld= len(cldh)
print n_cld, parameters['maxclp'] 

#plot variogram cloud (note that only was calculated for direction 1, variogram 1)

import numpy
plt.plot (cldh, cldg, '+', label='vcloud')
dip=parameters['dip'][0]
azm=parameters['azm'][0]

#plot the variogram on top for reference
plt.plot (pdis[0, 0, 1:], pgam[0, 0, 1:], '-o', label=str(dip) + '-->' + str(azm))
plt.legend()
plt.grid(True)

plt.show()

#plot all variograms in direction 1 (omnidirectional)

nvrg = len(parameters['ivtype']) 
tpe={1 : 'traditional semivariogram',
     2 : 'traditional cross semivariogram',
     3 : 'covariance',
     4 : 'correlogram',
     5 : 'general relative semivariogram',
     6 : 'pairwise relative semivariogram',
     7 : 'semivariogram of logarithms',
     8 : 'semimadogram',
     9 : 'indicator semivariogram - continuous',
     10 : 'indicator semivariogram - categorical'}

f, sp = plt.subplots(nrows=nvrg, figsize=(8,15))



for v in range(nvrg):
    t=tpe[parameters['ivtype'][v]]
    azm=parameters['azm'][d]
    sp[v].plot (pdis[v, 0, 1:], pgam[v, 0, 1:], '-o')
    sp[v].plot (pdis[v, 1, 1:], pgam[v, 1, 1:], '-o')
    sp[v].plot (pdis[v, 2, 1:], pgam[v, 2, 1:], '-o')
    sp[v].set_title(t)

plt.tight_layout()

plt.show()



