from matplotlib import rcParams
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')
from nilmtk import DataSet

gjw = DataSet('/Users/GJWood/nilm_gjw_data/HDF5/nilm_gjw_data.hdf5') #load the data from HDF5 file
gjw.set_window(start='2013-11-13 00:00:00', end='2013-11-14 00:00:00') #select a portion of the data
elec = gjw.buildings[1].elec #Select the relevant meter group

house = elec['fridge'] #only one meter so any selection will do

df = house.load().next() #load the first chunk of data into a dataframe
df.info() #check that the data is what we want (optional)
#note the data has two columns and a time index
#The period is one day, containing 86400 entries, one per second

df.plot()

df.ix['2013-11-13 06:30:00':'2013-11-13 07:30:00'].plot()# select a time range and plot it

house



