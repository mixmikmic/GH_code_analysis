# Import modules
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# dist_required
try:
    dist_required
except:
    dist_required = 75
    print("dist_required not specified, set to "+str(dist_required))

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

# Array containing the particle positions
pos = np.transpose(np.array([data["StgX"], data["StgY"]]))

# Calculate distances
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pos)
distances, indices = nbrs.kneighbors(pos)

#print(indices)
#print(distances[:,1])
data["Dist"] = distances[:,1]*1000

dist_N = data['Dist'].between(dist_required, max(data["Dist"]), inclusive=True)
dist_N = len(dist_N[dist_N==True])/len(data)*100

print("Median interparticle distance: "+str(np.median(data["Dist"]))+" um")
print("Fraction of particles further than "+str(dist_required)+" um apart: "+str(dist_N)+"%")



