get_ipython().system('cat kilroy.py')

import kilroy
dir(kilroy)

kilroy.Show('landscape','David_Shean_using_UW_terrestrial_laser_scanner_at_South_Cascade_Glacier.png', 900,700)

# Either from the terminal or via 'bang': These should be installed in the kernel already but here are some installs
# !conda install netcdf4 -y
# !conda install xarray -y

# This is a temporary as-needed install
get_ipython().system('conda install boto -y')

# If this cell throws a 'boto' error run the install in the cell above

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import pandas as pd
import netCDF4
import boto
from boto.s3.key import Key
import xarray as xr

# We 'happen to know' the glodap files reside on the 'himatdata' s3 bucket

connection = boto.connect_s3(anon=True)
bucket = connection.get_bucket('himatdata')
data_dir = './'

# This block of bespoke-y code prints all the root directory names from the AWS S3 'himatdata' bucket
dir_list = []
for key in bucket.list():
    keyname = str(key.name.encode('utf-8'))
    if '/' in keyname:
        cd = keyname.split('/')[0].strip("b'").strip('b"')
        if cd not in dir_list:
            dir_list.append(cd)
            print(cd)

# Now that we see there is a 'glodap' directory: List all the contents of that
file_list = []
for key in bucket.list():
    keyname = str(key.name.encode('utf-8'))
    if 'glodap/' in keyname:
        cd = keyname.strip("b'").strip('b"')
        print(cd)

# Ok, great, we see there are 14 data files available; let's just consider three of them
# This cell assigns filenames based on 'filename includes 'glodap' and data type of interest'

for key in bucket.list(): 
    filename = key.name.encode('utf-8')
    if b'glodap' in filename: 
        if b'salinity.nc' in filename: 
            print ('salinity file is', filename)
            salinityfilename = filename
        if b'temperature.nc' in filename: 
            print ('temperature file is', filename)
            temperaturefilename = filename
        if b'oxygen.nc' in filename: 
            print('oxygen file is', filename)
            oxygenfilename = filename            

local_salinity_filename = data_dir + 'glodap_salinity.nc'
local_temperature_filename = data_dir + 'glodap_temperature.nc'
local_oxygen_filename = data_dir + 'glodap_oxygen.nc'

get_ipython().system('pwd')
get_ipython().system('ls -al ./*.nc')

# This Python code copies three data files from the AWS cloud to the local file system
k = Key(bucket)
k.key = salinityfilename
k.get_contents_to_filename(local_salinity_filename)
k.key = temperaturefilename
k.get_contents_to_filename(local_temperature_filename)
k.key = oxygenfilename
k.get_contents_to_filename(local_oxygen_filename)

# If this works you should see three 100MByte files listed
get_ipython().system('ls -al ./glodap_*.nc')

# This code assigns our data files to three distinct dataset objects
print (local_salinity_filename)
dsSal = xr.open_mfdataset(local_salinity_filename)
dsTemp = xr.open_mfdataset(local_temperature_filename)
dsO2 = xr.open_mfdataset(local_oxygen_filename)
dsO2          # This prints the structure of the oxygen dataset below

# print(dsO2.Comment) # garbled text. Norwegian?
# print('\n')
print(dsO2.Description)             # The description is fine as far as it goes but does not indicate units.
print('\n')
print(dsO2.Citation)

# Run the 'directory' operation on dsO2 to find oxygen; on dsO2.oxygen to find units; then...
#dir(dsO2.oxygen.units)

# print units
print("\n" + dsO2.oxygen.units)

# This code indexes into the Oxygen dataset and prints a few example oxygen values
dsO2['oxygen'][0:2,50:52,170:172].values   # a few dissolved oxygen values near the surface

# These imports give us control sliders that we use for selecting depth slices from the dataset
from ipywidgets import *
from traitlets import dlink

# This creates a 2D color-coded view of oxygen at the surface, attaching a slider to a depth parameter
def plotOxygen(depth_index):
    a=dsO2['oxygen'].sel(depth_surface = depth_index)
    a.plot(figsize=(16, 10),cmap=plt.cm.bwr,vmin=150, vmax=350)
    msg = 'This is for '
    if depth_index == 0: msg += 'surface water'
    else: msg += 'water at ' + str(int(dsO2['Depth'].values[depth_index])) + ' meters depth'
    plt.text(25, -87, msg, fontsize = '20')
    plt.text(28, 50, 'oxygen dissolved in', fontsize = '20')
    plt.text(28, 42, '     ocean water   ', fontsize = '20')

# This is the interactive slider
interact(plotOxygen, depth_index=widgets.IntSlider(min=0,max=32,step=1,value=0, continuous_update=False))

def plotSalinity(depth_index):
    b = dsSal['salinity'].sel(depth_surface = depth_index)
    b.plot(figsize=(16, 10),cmap=plt.cm.bwr,vmin=33, vmax=36)
    msg = 'This is for '
    if depth_index == 0: msg += 'surface water'
    else: msg += 'water at ' + str(int(dsO2['Depth'].values[depth_index])) + ' meters depth'
    plt.text(25, -87, msg, fontsize = '20')
    plt.text(47, 50, 'salinity of', fontsize = '20')
    plt.text(47, 42, 'ocean water', fontsize = '20')
    
interact(plotSalinity, depth_index=widgets.IntSlider(min=0,max=32,step=1,value=0, continuous_update=False))

def plotTemperature(depth_index):
    c=dsTemp['temperature'].sel(depth_surface = depth_index)
    c.plot(figsize=(16, 10),cmap=plt.cm.bwr,vmin=0, vmax=23)
    msg = 'This is for '
    if depth_index == 0: msg += 'surface water'
    else: msg += 'water at ' + str(int(dsO2['Depth'].values[depth_index])) + ' meters depth'
    plt.text(25, -87, msg, fontsize = '20')
    plt.text(47, 50, 'temperature of', fontsize = '20')
    plt.text(47, 42, 'ocean water', fontsize = '20')

interact(plotTemperature, depth_index=widgets.IntSlider(min=0,max=32,step=1,value=0, continuous_update=False))   

# This code pulls out the various available depths (in meters) of the dataset indexed slices
print (dsTemp['Depth'].values[10])
print ('    ')
print (dsTemp['Depth'].values)



