import numpy as np
import StrataPlot as stratVis
get_ipython().magic('matplotlib inline')

get_ipython().system('ls strataGeo')

get_ipython().system('cd strataGeo; strataform delta1.sif')

get_ipython().system('ls strataGeo/delta1/delta1')

filebase ='./strataGeo/delta1/delta1/delta1_99_bas.vtu'
filestrati ='./strataGeo/delta1/delta1/delta1_99.vtu'

# Simulation start and end times (in years)
simStart = -1000000.
simEnd = 0.

# Number of stratigraphic layers recorded
# This is the number at the end of the filestrati name + 1
lays = 100

# Ellapsed time between 2 stratigraphic layers (in years)
dt = (simEnd-simStart)/lays

# Read and create relevant numpy arrays for plotting
minBase,x,y,z,mgz,age = stratVis.read_VTK(filebase,filestrati,simStart)

seafile = './strataGeo/data/sine_2cycle_rsl.sl'
SLtime,sealevel = stratVis.read_seaLevel(seafile)

# Set the regular grid resolution (in meters)
res = 25.0

nlays,xi,yi,zi,mzi,nxi,nyi = stratVis.mapData_Reg(lays,x,y,z,mgz,age,res,simStart,dt)

# X-axis value for the Y cross-section (in meters)
posX = 10000.
# sea-level final sea-level position
slvl = 0.
# Get Y cross-section
Ysec,base,sl,xID = stratVis.crossYsection(posX,res,xi,yi,nxi,nyi,minBase,slvl)

# Size of the figure
figS = [12,6]
# Y and Z axis clipping values
ylim = [10000,20000]
zlim = [-100,10]
# Frequency of layer outputs
layplot = 5
stratVis.plotYtime(figS,Ysec,zi,xID,base,minBase,sl,nlays,layplot,ylim,zlim)

'''
Define the number of points for mean grain size interpolation along the z axis
 - first value is Z bottom, 
 - second value is the maximum elevation of your interpolation grid
 - third parameter is the number of point along the Z axis 
 which in the case below means that we will put a point every 1 m
'''
mgzYGridRes = [-100, 100, 201]
mgzY,zY = stratVis.getYmgz(Ysec,zi,yi,mzi,nlays,xID,mgzYGridRes)

# Plot the mean grain zise along the Y-axis
figS = [15,6]
ylim = [10000,20000]
zlim = [-100,10]
stratVis.plotYmgz(figS,Ysec,zi,zY,mgzY,nlays,xID,ylim,zlim,minBase,sl)





