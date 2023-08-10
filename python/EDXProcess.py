# Import modules
import numpy as np
import pandas as pd
import struct
from os import listdir, path

if False:
    del directory, stub, data
    del EDX_save, EDX_channels
    
# directory
try:
    directory
except NameError:
    directory = "E:\\PA_UC\\"
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
    
# EDX_save
try:
    EDX_save
except NameError:
    EDX_save = False
    print("EDX_save not specified, set to "+str(EDX_save))
    
# EDX_channels
try:
    EDX_channels
except NameError:
    EDX_channels = 2086
    print("EDX_channels not specified, set to "+str(EDX_channels))

strStub = 'stub{num:02d}'.format(num=stub)
fld_prev = 1
partnum = 0
fldpart = np.array([])

for i in range(len(data)):
    fld = int(data.iloc[i]["fieldnum"])
    if fld!=fld_prev:
        # New field, reset particle counter
        partnum=1
        fld_prev = fld
    else:
        # Field with more than 1 particle
        partnum += 1
    fldpart = np.append(fldpart, "{0:04d}{1:04d}".format(fld, partnum))

#print(fldpart)    

def EDXimport(PartID=0):
    global N_EDX
    EDXfile = directory+strStub+"\\spc\\"+fldpart[PartID]+".spc"
    
    if(path.isfile(EDXfile)==False):
        return np.zeros(EDX_channels)
    
    N_EDX += 1
    with open(EDXfile, 'rb') as f:
        spectrum = np.zeros([EDX_channels],dtype="uint32")
        i = 0

        f.seek(3848)
        byte = f.read(4)    
        while byte and i<EDX_channels:
            spectrum[i] = struct.unpack('I', byte)[0]
            byte = f.read(4)
            i = i + 1
        
    return spectrum

#print(EDXimport())

EDX = np.zeros([EDX_channels])
N_EDX = 0
for i in range(len(data)):
    EDX = np.column_stack((EDX, EDXimport(i)))

EDX = np.delete(EDX, (0), axis=1)
EDX = np.transpose(EDX)
data["EDX"] = EDX.tolist()

print("Number of EDX spectra: "+str(N_EDX))

if EDX_save:
    np.savetxt(directory+strStub+"\\EDX.csv", EDX, fmt="%1.1d", delimiter=",")
    print("EDX spectra saved to "+directory+strStub+"\\EDX.csv")

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def EDXGetElement(E, dx=9):
    I = np.sum(EDX[:,E-dx:E+dx], axis=1)

    ax.add_patch(
        patches.Rectangle(
            (E-dx,0), 
            width = (2*dx), 
            height = max(y),
            linewidth=1,
            edgecolor='b',
            facecolor='none'
        )
    )
    
    return I

y = EDX[4,0:500]
fig,ax = plt.subplots(1)
ax.plot(range(len(y)), y, 'r-')

data["C"] = EDXGetElement(25, 6)
data["O"] = EDXGetElement(50, 7)
data["Na"] = EDXGetElement(102)
data["Si"] = EDXGetElement(172)
data["S"] = EDXGetElement(230)
data["Cl"] = EDXGetElement(265)
data["U"] = EDXGetElement(315)
data["Ba"] = EDXGetElement(448)
data["Ce"] = EDXGetElement(482, 12)
data["Fe"] = EDXGetElement(638)

plt.show()

from pandas.tools.plotting import scatter_matrix

columns = ["d", "A", "C", "O", "Na", "Si", "S", "Cl", "U", "Ba", "Ce", "Fe"]
test = data.loc[:,columns]

grr = pd.scatter_matrix(
    test, 
    figsize=(15, 15), 
    marker='o',
    hist_kwds={'bins': 20}, 
    s=10, 
    alpha=.8, 
)



