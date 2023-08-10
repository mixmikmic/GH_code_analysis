get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from matplotlib.font_manager import FontProperties
from pylab import rcParams
fontP = FontProperties()
fontP.set_size('small')

def loadData(filename):
    return genfromtxt(filename, delimiter=' ')

def plotData(data):
    rcParams['figure.figsize'] = 10, 6 # Set figure size to 10" wide x 6" tall
    t = data[:,0]
    altitude = data[:,1]
    verticalspeed = data[:,2]
    acceleration = data[:,3] # magnitude of acceleration
    gforce = data[:,4]
    throttle = data[:,5] * 100
    dthrottle_p = data[:,6] * 100
    dthrottle_d = data[:,7] * 100
    
    
    # Top subplot, position and velocity up to threshold
    plt.subplot(3, 1, 1)
    plt.plot(t, altitude, t, verticalspeed, t, acceleration)
    plt.axhline(y=100,linewidth=1,alpha=0.5,color='r',linestyle='--',label='goal');
    plt.text(max(t)*0.9, 105, 'Goal Altitude', fontsize=8);
    plt.title('Craft Altitude over Time')
    plt.ylabel('Altitude (meters)')
    plt.legend(['Altitude (m)','Vertical Speed (m/s)', 'Acceleration (m/s^2)'], "best", prop=fontP, frameon=False)
    
    # Middle subplot, throttle & dthrottle
    plt.subplot(3, 1, 2)
    plt.plot(t, throttle, t, dthrottle_p, t, dthrottle_d)
    plt.legend(['Throttle%','P','D'], "best",  prop=fontP, frameon=False) # Small font, best location
    plt.ylabel('Throttle %')
                
    # Bottom subplot, gforce
    plt.subplot(3, 1, 3)
    plt.plot(t, gforce)
    plt.axhline(y=1,linewidth=1,alpha=0.5,color='r',linestyle='--',label='goal');
    plt.text(max(t)*0.9, 1.2, 'Goal g-force', fontsize=8);
    plt.legend(['gforce'], "best", bbox_to_anchor=(1.0, 1.0), prop=fontP, frameon = False) # Small font, best location
    plt.xlabel('Time (seconds)')
    plt.ylabel('G-force');
    plt.show();

data = loadData('collected_data\\gforce.txt')
plotData(data)

data = loadData('collected_data\\vspeed.txt')
plotData(data)

# To run in kOS console: RUN hover3(pos0.txt,20,0.05,0).
data = loadData('collected_data\\pos0.txt')
plotData(data)

# To run in kOS console: RUN hover3(pos1.txt,20,0.05,0).
data = loadData('collected_data\\pos1.txt')
plotData(data)

# To run in kOS console: RUN hover3(pos2.txt,20,0.08,0.04).
data = loadData('collected_data\\pos2.txt')
plotData(data)

# To run in kOS console: RUN hover4(hover0.txt,60,0.01,0.001).
data = loadData('collected_data\\hover0.txt')
plotData(data)

# To run in kOS console: RUN hover4(hover1.txt,60,0.01,0.01).
data = loadData('collected_data\\hover1.txt')
plotData(data)

# To run in kOS console: RUN hover4(hover2.txt,10,0.1,0.1).
data = loadData('collected_data\\hover2.txt')
plotData(data)

# To run in kOS console: RUN hover5(hover3.txt,10,0.1,0.1,300).
data = loadData('collected_data\\hover3.txt')
plotData(data)

