#--------------------------------------------------------------------------------------------
# Import operator numpy and matplotlib
#--------------------------------------------------------------------------------------------
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from numpy import array as npa

#--------------------------------------------------------------------------------------------
# Define a find_str() function
# Notice, the string should be the only one in the whole text.
#--------------------------------------------------------------------------------------------
def find_str(str, arr):
    line = 0
    for ln in arr:
        line = line + 1
        if str in ln:
            return(line-1)
            break

#--------------------------------------------------------------------------------------------
# Read data from OUTCAR
#--------------------------------------------------------------------------------------------
file_1 = open ('data/OUTCAR', 'r')
out = [line.split() for line in file_1]
file_1.close()

lf = find_str("E-fermi", out)
Ef = float(out[lf][2])                         # Fermi level
lk = find_str("NKPTS", out)
nk = int(out[lk][3])                           # Number of K-point
nb = int(out[lk][-1])                          # Number of bands

#--------------------------------------------------------------------------------------------
# Get band structure data from PROCAR
# Add the high symmetry points as the x-axis as you are interested.
#--------------------------------------------------------------------------------------------
file_2 = open ('data/PROCAR', 'r')
band = [line.split() for line in file_2]
file_2.close()

eng = npa([float(band[i][4]) for i, ln in enumerate(band) if "energy" in ln])
data = np.reshape(eng, (nk,nb)).T-Ef

#--------------------------------------------------------------------------------------------
# Plot the band
#--------------------------------------------------------------------------------------------
x = np.arange(1,nk+1)
for i in range(nb):
    plt.plot(x,data[i],'k-')
plt.show()

#--------------------------------------------------------------------------------------------
# Import modules
#--------------------------------------------------------------------------------------------
import numpy as np
from numpy import array as npa
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
get_ipython().magic('matplotlib inline')

#--------------------------------------------------------------------------------------------
# Define a find_str() function
# Notice, the string should be the only one in the whole text.
#--------------------------------------------------------------------------------------------
def find_str(str, arr):
    line = 0
    for ln in arr:
        line = line + 1
        if str in ln:
            return(line-1)
            break

#--------------------------------------------------------------------------------------------
# Define a function myFloat transform myList to float
#--------------------------------------------------------------------------------------------
def myFloat(myList):
    return map(float, myList)

#--------------------------------------------------------------------------------------------
# Read data from OUTCAR
#--------------------------------------------------------------------------------------------
file_1 = open ('data/OUTCAR', 'r')
out = [line.split() for line in file_1]
file_1.close()

lfermi = find_str("E-fermi", out)
efermi = float(out[lfermi][2])                         # Fermi level

lkpt = find_str("NKPTS", out)
nkpt = int(out[lkpt][3])                               # Number of k-points
nband = int(out[lkpt][-1])                             # Number of bands

lions = find_str("NIONS", out)
nions = int(out[lions][-1])                            # Number of ions

#--------------------------------------------------------------------------------------------
# Read data from PROCAR
#--------------------------------------------------------------------------------------------
file_2 = open ('data/PROCAR', 'r')
band = [line.split() for line in file_2]
file_2.close()
ln_kpt = npa([i for i, ln in enumerate(band) if "k-point" in ln])
n_interval = ln_kpt[1] - ln_kpt[0]

# ps, pp, pd -- project to s, p and d orbitals
energy = [];ps = []; pp = []; pd = []
for i in range(nkpt):
    kp = band[ln_kpt[i]:ln_kpt[i]+n_interval]
    energy.append([float(kp[j][4]) for j, ln in enumerate(kp) if "energy" in ln])
    ps.append([float(kp[j+nions+1][1]) for j, ln in enumerate(kp) if "ion" in ln])
    pp.append([sum(myFloat(kp[j+nions+1][2:5])) for j, ln in enumerate(kp) if "ion" in ln])
    pd.append([sum(myFloat(kp[j+nions+1][5:10])) for j, ln in enumerate(kp) if "ion" in ln])

x = np.arange(1,nkpt+1)
y = npa(energy).T - efermi

#--------------------------------------------------------------------------------------------
# define seg() function to make segment for LineCollection
#--------------------------------------------------------------------------------------------
def seg(x, y):
    points = npa([x, y]).T.reshape(-1, 1, 2)
    segment = np.concatenate([points[:-1], points[1:]], axis=1)
    return segment

#--------------------------------------------------------------------------------------------
# Plot the projected band structure in colors
#--------------------------------------------------------------------------------------------
ax = plt.axes()
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())

for i in range(nband):
    segment = seg(x, y[i])
    c = [colors.rgb2hex((npa(ps).T[i][j], npa(pp).T[i][j], npa(pd).T[i][j])) for j in range(nkpt)]
    lc = LineCollection(segment, color=c, lw=3)
    ax.add_collection(lc)
plt.show()

