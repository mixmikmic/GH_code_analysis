get_ipython().system('cove')

get_ipython().system('cove 6 1 1. 0.1 25 10')

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

# Customise figure style 
from matplotlib import rc
rc('font',size=8)
rc('ytick.major',pad=5)
rc('xtick.major',pad=5)
padding = 5

# Define wave climate params
Theta = 25
ThetaStd = 10

# Open coastline evolution file
filename = "Spiral_6_1_1._0.1_" + str(Theta) + "_" + str(ThetaStd) + ".xy"

# Read the header info and get the lines
f = open(filename,"r")
Lines = f.readlines()
NoLines = len(Lines)-1
Header = Lines[0]

# Get the shoreline positions
X = np.array(Lines[1].strip().split(" ")[1:],dtype='float64')
Y = np.array(Lines[2].strip().split(" ")[1:],dtype='float64')

TimeIntervals = [0.5,1,2,4,8,16,20]

Time = np.zeros(NoLines-1)
for i in range(0,NoLines-1,2):
    Time[i] = float(Lines[i+1].strip().split(" ")[0])

# Define outputfile name
outputfile = "mean"+ str(Theta) + "std" + str(ThetaStd) + ".png"

# Setup the figure space, axis and line to animate
fig = plt.figure(1, figsize=(10,10))
plt.subplots_adjust(0.2,0.15,0.9,0.85)
plt.plot(X,Y,'k.',ms=1,label="Initial Coastline")
plt.xlabel('X-coordinate (m)')
plt.ylabel('Y-coordinate (m)')
plt.xlim(-500,2000)
plt.ylim(-2200,50)

for TimeInt in TimeIntervals:
    Diff = np.abs(Time-TimeInt)
    Line = np.argmin(Diff)
    Xt = np.array(Lines[Line+1].strip().split(" ")[1:],dtype='float64')
    Yt = np.array(Lines[Line+2].strip().split(" ")[1:],dtype='float64')
    plt.plot(Xt,Yt,'k--',lw=0.5)

xmin = np.min(Xt)-500.
xmax = np.max(Xt)+500
ymin = np.min(Yt)
ymax = np.max(Yt)

SeaX = np.append(Xt,[xmax,xmax,Xt[0]])
SeaY = np.append(Yt,[ymin,ymax,Yt[0]])
BeachX = np.append(Xt,[xmin,xmin,Xt[0]])
BeachY = np.append(Yt,[ymin,ymax,Yt[0]])
    
plt.fill(SeaX, SeaY, color=[0.7,0.9,1.0])
plt.fill(BeachX, BeachY, color=[1.0,0.9,0.6])
plt.plot(X,Y,'k--',lw=0.5,label="1/2, 1, 2, 4, 8, 16 Year Coastlines")
plt.plot(X,Y,'k-',lw=1,label="20 Year Coastline")
plt.plot(X,Y,'ko',ms=3)
plt.plot(X[0:2],Y[0:2],'k-',lw=2)
plt.plot(X[-2:],Y[-2:],'k-',lw=2)

# Display legend
plt.rcParams.update({'legend.labelspacing':0.1}) 
plt.rcParams.update({'legend.numpoints':1}) 
plt.rcParams.update({'legend.frameon':False}) 
plt.rcParams.update({'legend.handlelength':1.5}) 
plt.legend(loc=3)
leg = plt.gca().get_legend()

# Set fontsize to small
ltext  = leg.get_texts()
plt.setp(ltext, fontsize=8) 

# Bin all wave data
Waves = np.random.normal(Theta,ThetaStd,10000)
widths = 15*np.pi/180.0
ax = plt.axes([0.65,0.55,0.25,0.25],polar=True)
hist = np.histogram(Waves*np.pi/180.0,bins=np.arange(0,361*np.pi/180.0,widths))
plt.bar(hist[1][:-1],hist[0],widths,color='white',edgecolor='k')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_axis_bgcolor('none')
plt.axis('off')
plt.yticks([])
plt.xticks([])

plt.savefig(outputfile)
plt.show()

get_ipython().system('rm Spiral_6_1_1._0.1_*.xy')



