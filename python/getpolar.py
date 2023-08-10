## Input file -- OUTCAR

#--------------------------------------------------------------------------------------------
# Import operator numpy and matplotlib
#--------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# Define functions to transform myList to float/int
#--------------------------------------------------------------------------------------------
# Define a function myFloat transform myList to float
#--------------------------------------------------------------------------------------------
def myFloat(myList):
    return map(float, myList)
def myInt(myList):
    return map(int, myList)

#--------------------------------------------------------------------------------------------
# Set print precision
#--------------------------------------------------------------------------------------------
np.set_printoptions(precision=6,suppress=True)


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
# Read Ionic and electronic dipole moment -- ip and ep (unit: e*Angst)
# Read lattice parameter and volume
# calc quota and total polarization in unit um/cm^2
#--------------------------------------------------------------------------------------------
out = [line.split() for line in open ("data/OUTCAR", 'r')]
lni = find_str("p[ion]=(", out)
lne = find_str("p[elc]=(", out)
ip = myFloat(out[lni][4:7])                  # ionic dipole moment
ep = myFloat(out[lne][5:8])                  # electronic dipole moment

lnl = find_str("VOLUME", out)
lp = myFloat(out[lnl+10][0:3])               # lattice parameter
v = np.prod(np.array(lp))                    # volume

quota = np.divide(lp,v)*1600                 # quota of polarization (unit: um/cm^2)

tp = np.add(ip, ep)*1600/v                   # total polarization along three directions (unit: um/cm^2)

# Reduce quota from polarization
for i in range(3):
    for n in range(100):
        if abs(tp[i]) - quota[i] > -0.1:
            tp[i] = tp[i] - quota[i]
        else:
            break

# Print all
print "The ionic and electronic dipole moment: \n", ip, "e*Angst\n", ep, "e*Angst\n"
print "The lattice parameters and volume: \n", lp, "Angst\n", v, "Angst^3\n"
print "The quota of polarization: \n", quota, "um/cm^2\n"
print "The total polarization: \n", tp, "um/cm^2\n"

