# Import modules
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# directory
try:
    directory
except NameError:
    directory = "F:\\PA_UC\\"
    print("Directory not specified, set to "+directory)

# stub
try:
    stub
except NameError:
    stub = 1
    print("Stub not specified, set to "+str(stub))

# data
try:
    data
except NameError:
    print("No data available, running ImportData:")
    get_ipython().magic('run ./ImportData.ipynb')
    print("-----")

def ClassifyUranium(threshold=5):
    cls = np.zeros(len(data))
    clsN = np.zeros(3)
    
    for i in range(len(data)):
        if data.iloc[i]["MinCnts"]==0:
            cls[i] = 0
            clsN[0] += 1
        elif data.iloc[i]["UM"]>threshold:
            cls[i] = 1
            clsN[1] += 1
        else:
            cls[i] = 2
            clsN[2] += 1
            
    data["classification"] = cls;
    
    print("0: "+str(clsN[0])+" particles with no EDX data")
    print("1: "+str(clsN[1])+" uranium particles")
    print("2: "+str(clsN[2])+" non-uranium particles")
        
    return cls;

ClassifyUranium()

plt.scatter(
    x = data["X"], 
    y = data["Y"], 
    c = data["classification"],
    s = 5,
    alpha = 1,
    lw = 0,
    cmap = "rainbow_r")
plt.colorbar()
plt.show()

def ClassifyUranylChloride(ClThreshold=1):
    cls = np.zeros(len(data))
    clsN = np.zeros(5)
    
    for i in range(len(data)):
        if data.iloc[i]["MinCnts"]==0:
            # Other
            cls[i] = 4
            clsN[4] += 1
        elif data.iloc[i]["d"]>2.8:
            if data.iloc[i]["ClK"]>ClThreshold:
                # UCl droplet
                cls[i] = 0
                clsN[0] += 1
            else :
                # UO agglomerate
                cls[i] = 3
                clsN[3] += 1
        else:
            if data.iloc[i]["ClK"]>ClThreshold:
                # UCl Particle
                cls[i] = 1
                clsN[1] += 1
            else :
                # UO particle
                cls[i] = 2
                clsN[2] += 1
            
    data["classification"] = cls;
    
    print("0: "+str(clsN[0])+" particles with no EDX data")
    print("1: "+str(clsN[1])+" uranyl chloride droplets")
    print("2: "+str(clsN[2])+" uranyl chloride particles")
    print("3: "+str(clsN[3])+" uranium oxide particles")
    print("4: "+str(clsN[4])+" uranium oxide agglomerates")
        
    return cls;

if False:
    ClassifyUranylChloride()

    plt.scatter(
        x = data["X"], 
        y = data["Y"], 
        c = data["classification"],
        s = 5,
        alpha = 1,
        lw = 0,
        cmap = "rainbow_r")
    plt.colorbar()
    plt.show()



