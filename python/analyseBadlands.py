import warnings
warnings.filterwarnings('ignore')

import glob
import numpy as np
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter

from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

from scripts import stratalAnalyse_basin as strata

# display plots in SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')

# help(strata.stratalSection.__init__)

folder = 'output/h5/'
strat = strata.stratalSection(folder)

stepCounter = len(glob.glob1(folder,"tin.time*"))-1
print "Number of visualisation steps created: ",stepCounter

# help(strat.loadStratigraphy)

strat.loadStratigraphy(stepCounter) 

# Build a cross-section along X axis in the center of the domain
x1 = np.amin(strat.x)
x2 = np.amax(strat.x)
y1 = np.amax(strat.x)/2
y2 = np.amax(strat.x)/2

# Interpolation parameters
nbpts = strat.nx
gfilt = 1

# help(strat.buildSection)

strat.buildSection(xo = x1, yo = y1, xm = x2, ym = y2, pts = nbpts, gfilter = gfilt)

#help(strata.viewSection)

strata.viewSection(width = 800, height = 500, cs = strat, 
            dnlay = 5, rangeX=[8000, 22000], rangeY=[-120,50],
            linesize = 0.5, title='Stratal stacking pattern coloured by time')

time, Sealevel = strata.readSea('data/sealevel.csv') 
# There are 100 (=strat.nz) stratigraphic layers. We need to extract the sea level value of each stratigraphic layer.
dnlay = time.shape[0]/strat.nz
Time = np.array(time[::dnlay])
sealevel = np.array(Sealevel[::dnlay])
timeStep = int(np.amax(time)/strat.nz) 

# Plot sea-level
strata.viewData(x0 = Time/timeStep, y0 = sealevel, width = 600, height = 400, linesize = 3, 
                color = '#6666FF',title='Sea-level curve',xlegend='display steps',ylegend='sea-level position')

cHST = 'rgba(213,139,214,0.8)' 
cFSST = 'rgba(215,171,117,0.8)' 
cLST = 'rgba(39,123,70,0.8)' 
cTST = 'rgba(82,128,233,0.8)' 

FSST1 = np.array([0,25],dtype=int)
LST1 = np.array([25,32],dtype=int)
TST1 = np.array([32,45],dtype=int)
HST1 = np.array([45,50],dtype=int)
FSST2 = np.array([50,75],dtype=int)
LST2 = np.array([75,82],dtype=int)
TST2 = np.array([82,95],dtype=int)
HST2 = np.array([95,100],dtype=int)

# Build the color list
STcolors = []
for k in range(FSST1[0],FSST1[1]):
    STcolors.append(cFSST)
for k in range(LST1[0],LST1[1]):
    STcolors.append(cLST)
for k in range(TST1[0],TST1[1]):
    STcolors.append(cTST)
for k in range(HST1[0],HST1[1]):
    STcolors.append(cHST)
for k in range(FSST2[0],FSST2[1]):
    STcolors.append(cFSST)
for k in range(LST2[0],LST2[1]):
    STcolors.append(cLST)
for k in range(TST2[0],TST2[1]):
    STcolors.append(cTST)
for k in range(HST2[0],HST2[1]):
    STcolors.append(cHST)

# help(strata.viewSectionST)

strata.viewSectionST(width = 800, height = 500, cs=strat, colors=STcolors,
                   dnlay = 2, rangeX=[8000, 22000], rangeY=[-120,50], 
                   linesize=0.5, title='Systems tracts interpreted based on relative sea-level (RSL)')

nbout = stepCounter
nstep = 2

strat_all = [strata.stratalSection(folder)]
strat_all[0].loadStratigraphy(1)
k = 1
for i in range(nstep,nbout+1,nstep):
    strat = strata.stratalSection(folder)
    strat_all.append(strat)
    strat_all[k].loadStratigraphy(i)
    k += 1

# This will take a while...
npts = len(strat_all)
for i in range(0,npts):
    strat_all[i].buildSection(xo = x1, yo = y1, xm = x2, ym = y2, pts = nbpts, gfilter = gfilt)

time, Sealevel = strata.readSea('data/sealevel.csv') 
# Extract the value of sea-level for each stratigraphic layer
dnlay = time.shape[0]/nbout*nstep
Time = np.array(time[::dnlay])
sealevel = np.array(Sealevel[::dnlay])
sealevel[0] = Sealevel[1]
timeStep = int(np.amax(time)/nbout*nstep) 

# Plot sea-level
# strata.viewData(x0 = Time/timeStep, y0 = sealevel, width = 600, height = 400, linesize = 3, 
#                 color = '#6666FF',title='Sea-level curve',xlegend='display steps',ylegend='sea-level position')

# help(strat.buildParameters)

strat.buildParameters(npts = npts, strat_all = strat_all, sealevel = sealevel, gfilter = 0.1, style = 'basin')

# left-hand side
xval = np.linspace(0,strat.nz,npts)
yval = strat_all[0].dist[strat.shoreID_l.astype(int)]
gval = np.gradient(-yval)

# View shoreline position through time
strata.viewData(x0 = xval, y0 = yval, width = 600, height = 400, linesize = 3, color = '#6666FF',
               title='Shoreline trajectory',xlegend='display steps',ylegend='shoreline position in metres')

# Define the gradient evolution
strata.viewData(x0 = xval, y0 = gval, width = 600, height = 400, linesize = 3, color = '#f4a142',
               title='Shoreline trajectory gradient',xlegend='display steps',ylegend='gradient shoreline position')

# right-hand side
xval = np.linspace(0,strat.nz,npts)
yval_r = strat_all[0].dist[strat.shoreID_r.astype(int)]
gval_r = np.gradient(yval_r)

# View shoreline position through time
strata.viewData(x0 = xval, y0 = yval_r, width = 600, height = 400, linesize = 3, color = '#6666FF',
               title='Shoreline trajectory',xlegend='display steps',ylegend='shoreline position in metres')

# Define the gradient evolution
strata.viewData(x0 = xval, y0 = gval_r, width = 600, height = 400, linesize = 3, color = '#f4a142',
               title='Shoreline trajectory gradient',xlegend='display steps',ylegend='gradient shoreline position')

# Default color used: 
TC = 'rgba(56,110,164,0.8)'
STC = 'rgba(60,165,67,0.8)'  
ARC = 'rgba(112,54,127,0.8)'
DRC = 'rgba(252,149,7,0.8)'

# You can change them by specifying new values in the function
STcolors_ST = strata.build_shoreTrajectory(xval, yval, gval, sealevel, strat.nz, cTC=TC, cDRC=DRC, cARC=ARC, cSTC=STC)
# For a different side
STcolors_ST_r = strata.build_shoreTrajectory(xval, yval_r, gval_r, sealevel, strat.nz, cTC=TC, cDRC=DRC, cARC=ARC, cSTC=STC)

strata.viewSectionST(width = 800, height = 500, cs = strat, colors = STcolors_ST, 
                     dnlay = 2, rangeX=[8000, 22000], rangeY=[-120,50], 
                     linesize = 0.5, title = 'Classes interpreted based on shoreline trajectory (ST)')

# left-hand side
xval = np.linspace(0,strat.nz,npts)
ASval = strat.accom_l-strat.sed_l
gval = np.gradient(ASval)

# Accommodation (A) and sedimentation (S) change differences
strata.viewData(x0 = xval, y0 = ASval, width = 600, height = 400, linesize = 3, 
                color = '#6666FF',title='Change between accomodation & sedimentation',xlegend='display steps',
                ylegend='A-S')

# Define the gradient evolution
strata.viewData(x0 = xval, y0 = gval, width = 600, height = 400, linesize = 3, color = '#f4a142',
               title='A&S gradient',xlegend='display steps',ylegend='gradient A&S')

# right-hand side
xval = np.linspace(0,strat.nz,npts)
ASval_r = strat.accom_r-strat.sed_r
gval_r = np.gradient(ASval_r)

# Accommodation (A) and sedimentation (S) change differences
strata.viewData(x0 = xval, y0 = ASval_r, width = 600, height = 400, linesize = 3, 
                color = '#6666FF',title='Change between accomodation & sedimentation',xlegend='display steps',
                ylegend='A-S')

# Define the gradient evolution
strata.viewData(x0 = xval, y0 = gval_r, width = 600, height = 400, linesize = 3, color = '#f4a142',
               title='A&S gradient',xlegend='display steps',ylegend='gradient A&S')

# Default color used: 
R = 'rgba(51,79,217,0.8)' 
APD = 'rgba(252,149,7,0.8)' 
PA= 'rgba(15,112,2,0.8)'

# You can change them by specifying new values in the function
STcolors_AS = strata.build_accomSuccession(xval,ASval,gval,strat.nz,cR=R,cPA=PA,cAPD=APD)
# For a different side
STcolors_AS_r = strata.build_accomSuccession(xval,ASval_r,gval_r,strat.nz,cR=R,cPA=PA,cAPD=APD)

strata.viewSectionST(width = 800, height = 500, cs = strat, colors = STcolors_AS, 
                     dnlay = 2, rangeX=[8000, 22000], rangeY=[-120,50], linesize = 0.5, 
                     title = 'Sequence sets interpreted based on change of accommodation and sedimentation (AS)')

# Specify the range of water depth (relative to sea level, positive = below sea level) of each depositional environment
depthIDs = [-30, 0, 30, 50, 100] 

# Build enviID list
enviID = []
for i in range(len(strat_all)):  # len(cs): number of layers that are read
    nz = strat_all[i].nz-1
    nbpts = strat_all[i].dist.shape[0]  # nbpts: interpolate space
    for j in range(nbpts):  
        if ((strat_all[i].secDep[nz][j]) > (sealevel[i] - depthIDs[0])):
            enviID.append(0)
        elif ((strat_all[i].secDep[nz][j]) > (sealevel[i] - depthIDs[1])):
            enviID.append(1)  # alluvial plain
        elif ((strat_all[i].secDep[nz][j]) > (sealevel[i] - depthIDs[2])):
            enviID.append(2)  # shoreface
        elif ((strat_all[i].secDep[nz][j]) > (sealevel[i] - depthIDs[3])):
            enviID.append(3)  # slope
        elif ((strat_all[i].secDep[nz][j]) > (sealevel[i] - depthIDs[4])):
            enviID.append(4)  # deep marine
        else:
            enviID.append(5)  # ocean basin
# 
enviID = np.array(enviID)
enviID = np.reshape(enviID, (len(strat_all), nbpts))

# fig = plt.figure(figsize = (7,5))
fig, ax = plt.subplots(figsize=(7,6))
plt.rc("font", size=10)
# make a color map of fixed colors
color = ['white','limegreen','sandybrown','khaki','c','teal']
cmap = colors.ListedColormap(color)
bounds=[0,1,2,3,4,5,6]
norm = colors.BoundaryNorm(bounds, cmap.N)
img = plt.imshow(np.flip(enviID, 0), cmap=cmap, norm=norm, interpolation='nearest', extent=[0,30,0,1], aspect=20)
cbar = plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels(['','-30','0','30','50','100'])
cbar.set_label('Paleo-depth (m)')
# 
for j in range(0,npts): 
        plt.axhline(Time[j]/1e6, color='k', linewidth=0.1)
# Plot shoreline trajectory
plt.scatter(strat.dist[strat.shoreID_l.astype(int)]/1000, Time/1e6, s=6, color='k')  # left-side
plt.scatter(strat.dist[strat.shoreID_r.astype(int)]/1000, Time/1e6, s=6, color='k')  # right-side

plt.xlim(7.5, 22.5)
plt.ylim(0, 1.0)

plt.xlabel('Distance (km)')
plt.ylabel('Time (Myr)')
plt.title('Wheeler Diagram')
# 
#fig.savefig("WheelerDiag.png", dpi=400)

# position of wells (km)
posit = np.array([11, 12, 18, 19, 20])  # position of wells /km
positID = [int(x*(nbpts-1)*1000/strat.x.max()) for x in posit]  # find the index of wells in strat.dist
color = ['white','limegreen','sandybrown','khaki','c','teal']
# Build color list for vertical stackings
color_fill = []
for i in positID:
    for j in range(0,strat.nz,nstep):
        if ((strat.secElev[j][i]) > (- depthIDs[0])):
            color_fill.append(color[0])
        elif (strat.secElev[j][i]) > (- depthIDs[1]):
            color_fill.append(color[1])
        elif (strat.secElev[j][i]) > (- depthIDs[2]):
            color_fill.append(color[2])
        elif (strat.secElev[j][i]) > (- depthIDs[3]):
            color_fill.append(color[3])
        elif (strat.secElev[j][i]) > (- depthIDs[4]):
            color_fill.append(color[4])
        else:
            color_fill.append(color[5])
colorFill = np.reshape(color_fill, (len(positID), int(strat.nz/nstep)))
# 
layth = []
for m in positID:
    nz = strat.nz-1
    layth.append(strat.secDep[nz][m])
    for n in range(1,int(strat.nz/nstep)):
        layth.append(-sum(strat.secTh[(nz-n*nstep):(nz-(n-1)*nstep)])[m])
layTh = np.reshape(layth, (len(positID), int(strat.nz/nstep)))

import matplotlib as mpl
fig = plt.figure(figsize = (4,5))
plt.rc("font", size=10)
# 
ax = fig.add_axes([0.18,0.06,0.82,0.91])
data = layTh
bottom1 = np.cumsum(data[0], axis=0)
colors_1 = np.fliplr([colorFill[0]])[0]
plt.bar(0, data[0][0], color = 'w', edgecolor='lightgrey', hatch = '/')
for j in range(1, data[0].shape[0]):
    plt.bar(0, data[0][j], color=colors_1[j], edgecolor='black', bottom=bottom1[j-1])
# 
bottom2 = np.cumsum(data[1], axis=0)
colors_2 = np.fliplr([colorFill[1]])[0]
plt.bar(2, data[1][0], color = 'w', edgecolor='lightgrey', hatch = '/')
for j in range(1, data[1].shape[0]):
    plt.bar(2, data[1][j], color=colors_2[j], edgecolor='black', bottom=bottom2[j-1])
# 
bottom3 = np.cumsum(data[2], axis=0)
colors_3 = np.fliplr([colorFill[2]])[0]
plt.bar(4, data[2][0], color = 'w', edgecolor='lightgrey', hatch = '/')
for j in range(1, data[2].shape[0]):
    plt.bar(4, data[2][j], color=colors_3[j], edgecolor='black', bottom=bottom3[j-1])
#
bottom4 = np.cumsum(data[3], axis=0)
colors_4 = np.fliplr([colorFill[3]])[0]
plt.bar(6, data[3][0], color = 'w', edgecolor='lightgrey', hatch = '/')
for j in range(1, data[3].shape[0]):
    plt.bar(6, data[3][j], color=colors_4[j], edgecolor='black', bottom=bottom4[j-1])
#
bottom5 = np.cumsum(data[4], axis=0)
colors_5 = np.fliplr([colorFill[4]])[0]
plt.bar(8, data[4][0], color = 'w', edgecolor='lightgrey', hatch = '/')
for j in range(1, data[4].shape[0]):
    plt.bar(8, data[4][j], color=colors_5[j], edgecolor='black', bottom=bottom5[j-1])
#
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.axes.get_xaxis().set_visible(False)
ax.tick_params(axis='both', labelsize=10)
ax.yaxis.set_ticks_position('left')
# plt.xlim(-0.4,8)
# plt.ylim(8.5,6.5)
plt.ylabel('Elevation (m)',fontsize=10)
plt.yticks(fontsize=10)
# 
# fig.savefig("/workspace/volume/basin/VerticalStack.png", dpi=400)





