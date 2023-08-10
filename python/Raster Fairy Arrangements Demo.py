# just some basic setup for the purpose of this demo:
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display
import matplotlib.pyplot as plt
import math

import numpy as np
import rasterfairy

#number of elements to find arrangements for:
totalDataPoints = 478

#in thise case the original point set doesn't matter, zeros will do
xy = np.zeros((totalDataPoints,2))

#rectangular grid:
rectArrangements = rasterfairy.getRectArrangements(totalDataPoints) 

#other arrangements:
arrangements = rasterfairy.getArrangements(totalDataPoints) 
#...which have to be converted to a raster mask
arrangementMasks = rasterfairy.arrangementListToRasterMasks(arrangements) 

print "Found these arrangements for",totalDataPoints,"elements"

# eating my own dog food I'm also using raster fairy to get a good table for showing t
# the available arrangements
totalArrangements = len(arrangementMasks) +  len(rectArrangements)
demoGrid = rasterfairy.getRectArrangements( totalArrangements )[0]

# if the number of arrangements does not split evenly
# and the proportion would be ugly we fall back to an
# incomplete square:
if float(demoGrid[0]) / float(demoGrid[1]) < 0.4:
    rows = int(math.sqrt(totalArrangements))
    cols = int(math.ceil(totalArrangements/float(rows) ))
    demoGrid = (rows,cols)

fig = plt.figure(figsize=(16.0, 16.5 * float(demoGrid[0]) / float(demoGrid[1])))
j = 1
for i in range(len(rectArrangements)):
    grid_xy,(width,height) =  rasterfairy.transformPointCloud2D(xy,target=rectArrangements[i])
    
    # fix for stretching behaviour of matplotlib
    grid_xy[:,0] -= np.min(grid_xy[:,0])
    grid_xy[:,1] -= np.min(grid_xy[:,1])
    w = np.max(grid_xy[:,0])
    h = np.max(grid_xy[:,1])
    dim = max(w,h)
    grid_xy[:,0] += 0.5 * (dim-w)+0.5
    grid_xy[:,1] += 0.5 * (dim-h)+0.5
    
    ax = fig.add_subplot(demoGrid[0], demoGrid[1], j)
    j+=1
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.set_xlim([0,dim*1.1])
    ax.axes.set_ylim([0,dim*1.1])
    ax.scatter(grid_xy[:,0],grid_xy[:,1], c = 'black',  edgecolors='none',s=200.0 / dim)  
    ax.set_title( "rectangular ("+str(rectArrangements[i][0])+"x"+str(rectArrangements[i][1])+")")
        
for i in range(len(arrangementMasks)):
    grid_xy,(width,height) = rasterfairy.transformPointCloud2D(xy,target=arrangementMasks[i])
    
    # fix for stretching behaviour of matplotlib
    grid_xy[:,0] -= np.min(grid_xy[:,0])
    grid_xy[:,1] -= np.min(grid_xy[:,1])
    w = np.max(grid_xy[:,0])
    h = np.max(grid_xy[:,1])
    dim = max(w,h)
    grid_xy[:,0] += 0.5 * (dim-w)+0.5
    grid_xy[:,1] += 0.5 * (dim-h)+0.5
    
    ax = fig.add_subplot(demoGrid[0], demoGrid[1], j)
    j+=1
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.set_xlim([0.0,1.1*dim])
    ax.axes.set_ylim([0.0,1.1*dim])
    
    ax.scatter(grid_xy[:,0],grid_xy[:,1], c = 'black',  edgecolors='none',s=200.0 / dim)  
    ishex = ""
    if arrangementMasks[i]['hex']:
        ishex = "hex "
    ax.set_title( ishex + arrangementMasks[i]['type'] + "("+str(arrangementMasks[i]['width'])+"x"+str(arrangementMasks[i]['height'])+")")

plt.show()

print rectArrangements

radius,adjustmentFactor,count = rasterfairy.getBestCircularMatch(totalDataPoints)
arrangement = rasterfairy.getCircularArrangement(radius,adjustmentFactor)
rasterMask = rasterfairy.arrangementToRasterMask(arrangement)

grid_xy, (width, height) = rasterfairy.transformPointCloud2D(xy,target=rasterMask)

fig = plt.figure(figsize=(10.0,10.0))
ax = fig.add_subplot(1, 1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.set_xlim([-1.0,max(width,height)])
ax.axes.set_ylim([-1.0,max(width,height)])
ax.invert_yaxis()
ax.scatter(grid_xy[:,0],grid_xy[:,1], c = 'black',  edgecolors='none',marker='s',s=50)    
plt.show()





