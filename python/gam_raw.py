#general imports
import matplotlib.pyplot as plt  
import pygslib         

#make the plots inline
get_ipython().run_line_magic('matplotlib', 'inline')

#get the data in GSLIB format into a pandas Dataframe
mydata= pygslib.gslib.read_gslib_file('../datasets/true.dat')  

# This is a 2D grid file with two variables in addition we need a 
# dummy BHID = 1 (this is like domain to avoid creating pairs from apples and oranges)
mydata['bhid']=1

# printing to verify results
print ' \n **** 5 first rows in my datafile \n\n  ', mydata.tail(n=5)

# these are the parameters we need. Note that at difference of GSLIB this dictionary also stores 
# the actual data 
#important! python is case sensitive 'bhid' is not equal to 'BHID'

# the program gam (fortran code) use flatten array of 2D arrays (unlike gamv which 
# works with 2D arrays). This is a work around to input the data in the right format
# WARNING: this is only for GAM and make sure you use FORTRAN order. 
vr=mydata[['Primary', 'Secondary']].values.flatten(order='FORTRAN')
U_var= mydata[['Primary']].var()
V_var= mydata[['Secondary']].var()


parameters = { 
        'nx'     :  50,                     # number of rows in the gridded data
        'ny'     :  50,                     # number of columns in the gridded data
        'nz'     :  1,                      # number of levels in the gridded data
        'xsiz'   :  1,                      # size of the cell in x direction 
        'ysiz'   :  1,                      # size of the cell in y direction
        'zsiz'   :  1,                      # size of the cell in z direction
        'bhid'   :  mydata['bhid'],         # bhid for downhole variogram, array('i') with bounds (nd)    
        'vr'     :  vr,     # Variables, array('f') with bounds (nd,nv), nv is number of variables
        'tmin'   : -1.0e21,                 # trimming limits, float
        'tmax'   :  1.0e21,                 # trimming limits, float
        'nlag'   :  10,                     # number of lags, int
        'ixd'    : [1,0],                   # direction x 
        'iyd'    : [0,1],                   # direction y 
        'izd'    : [0,0],                   # direction z 
        'isill'  : 1,                       # standardize sills? (0=no, 1=yes), int
        'sills'  : [U_var, V_var],          # variance used to std the sills, array('f') with bounds (nv)
        'ivtail' : [1,1,2,2],               # tail var., array('i') with bounds (nvarg), nvarg is number of variograms
        'ivhead' : [1,1,2,2],               # head var., array('i') with bounds (nvarg)
        'ivtype' : [1,3,1,3]}               # variogram type, array('i') with bounds (nvarg)

            
#check the variogram is ok
#TODO: assert gslib.check_gam_par(parameters)==1 , 'sorry this parameter file is wrong' 

#Now we are ready to calculate the veriogram
pdis,pgam, phm,ptm,phv,ptv,pnump= pygslib.gslib.gam(parameters)

nvrg = pdis.shape[0]
ndir = pdis.shape[1]
nlag = pdis.shape[2]-2

print 'nvrg: ', nvrg, '\nndir: ', ndir, '\nnlag: ', nlag

import pandas as pd
variogram=0
dir1=0
dir2=1

pd.DataFrame({'dis1': pdis[variogram, dir1, : -2], 
 'gam1': pgam[variogram, dir1, : -2], 
 'npr1': pnump[variogram, dir1, : -2], 
 'dis2': pdis[variogram, dir2, : -2], 
 'gam2': pgam[variogram, dir2, : -2], 
 'npr2': pnump[variogram, dir2, : -2]})

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#plotting the variogram 1 only
v=0

# in all the directions calculated
for d in range(ndir):
    ixd=parameters['ixd'][d]
    iyd=parameters['iyd'][d]
    izd=parameters['izd'][d]
    plt.plot (pdis[v, d, :-2], pgam[v, d, :-2], '-o', label=str(ixd) + '/' + str(iyd) + '/' + str(izd))

# adding nice features to the plot
plt.legend()
plt.grid(True)
plt.show()



