# Import modules
from os import listdir
import numpy as np
import pandas as pd

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

for i in listdir(directory):
    if i.endswith(".csv"):
        file = directory+i

print("File: "+file)

with open(file) as f:
    for i in range(0, 20):
        line = f.readline().replace("\n", "").split(',')
        if(line[0].strip()=="Part#"):
            columns = line
            skip=i+1
        elif(line[0].strip()=="Date"):
            print("Date: "+str(line[2].strip())+"."+str(line[1].strip())+"."+str(line[3].strip()))
        elif(line[0].strip()=="Acc."):
            print("Voltage: "+str(line[2].strip())+" kV")
        elif(line[0].strip()=="Magn:"):
            print("Magnification: "+str(line[1].strip())+"x")
        elif(line[0].strip()=="Preset"):
            print("Measurement time: "+str(line[2].strip())+" s")

print('Number of columns: '+str(len(columns)))
for i in range(len(columns)):
    columns[i] = columns[i].strip()

data = np.loadtxt(file, delimiter=',', skiprows=skip, comments='_')
print("Number of particles: "+str(len(data)))

# Transfer data into a dataframe
data = pd.DataFrame(data, columns=columns)
    
#print(data)

# Get stub for each particle
data["stub"] = np.rint(np.floor(data["Field#"]/10000))

# Only select particles from selected strub
data = data[data["stub"]==stub]

# Get field number
data["fieldnum"] = np.rint(np.floor(data["Field#"]-10000*stub))

print("Particles on stub "+str(stub)+": "+str(len(data)))

#print("Number of fields: "+str(len()))

data["X"] = data["StgX"]
data["Y"] = data["StgY"]
data["d"] = data["AvgDiam"]
data["A"] = data["Aspe"]

# Just to test some output options:
#from IPython.display import HTML
#HTML(data.to_html())



