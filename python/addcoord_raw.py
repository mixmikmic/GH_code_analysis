#general imports
import matplotlib.pyplot as plt  
import pygslib      

#make the plots inline
get_ipython().magic('matplotlib inline')

# To put the GSLIB file into a Pandas DataFrame
mydata= pygslib.gslib.read_gslib_file('../datasets/true.dat') 

# This is a 2D grid file with two variables
# To add a dummy variable BHID = 1 use this code
mydata['bhid']=1

# printing to verify results
print ' \n **** 5 first rows in my datafile **** \n\n  ', mydata.tail(n=5)

# GSLIB grids have not coordinates
# To add coordinates use addcoord function

#first prepare a parameter file
par= {  'nx'  : 50,
        'ny'  : 50,
        'nz'  : 1,
        'xmn' : 0.5,
        'ymn' : 0.5,
        'zmn' : 1,
        'xsiz': 1,
        'ysiz': 1,
        'zsiz': 10,
        'grid': mydata }

# then run the function
mydataxyz=pygslib.gslib.addcoord(**par)

# this is to see the output
mydataxyz

# Now we can plot the centroids of the cells
plt.plot(mydataxyz['x'],mydataxyz['y'], '+')



