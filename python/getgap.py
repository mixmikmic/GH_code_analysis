## Input file -- DOSCAR

#--------------------------------------------------------------------------------------------
# Import operator numpy and matplotlib
#--------------------------------------------------------------------------------------------
import numpy as np
from numpy import array as npa
from numpy import concatenate as npc


#--------------------------------------------------------------------------------------------
# Define a function myFloat transform myList to float
#--------------------------------------------------------------------------------------------
def myFloat(myList):
    return map(float, myList)


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
# Read the DOS
#--------------------------------------------------------------------------------------------
file_1 = open ('data/Si/DOSCAR', 'r')
out = [line.split() for line in file_1]
file_1.close()

ef = float(out[5][3])
nedos = int(out[5][2])
tdos = npa(map(myFloat, out[6:nedos+6])).T

for i in range(nedos):
    if tdos[0][i] > ef:
        c = i
        v = i
        break

if tdos[1][i]>0.0001 and tdos[1][i-1]>0.0001:
    print 'Sorry, No gap!'
    
else:
    while tdos[1][c]<0.0001:
        c = c + 1
    c = c - 1
    while tdos[1][v]<0.0001:
        v = v - 1
    v = v + 1
    
    print 'VBM is %s and CBM is %s' %(tdos[0][v], tdos[0][c])
    print 'The band gap is %s eV' %(tdos[0][c]-tdos[0][v])

# Input file -- PROCAR

#--------------------------------------------------------------------------------------------
# Read the PROCAR
#--------------------------------------------------------------------------------------------
file_2 = open ('data/Si/PROCAR', 'r')
band = [line.split() for line in file_2]
file_2.close()

nk = int(band[1][3])
nb = int(band[1][7])

eng = npa([float(band[j][4]) for j, ln in enumerate(band) if "energy" in ln])
data = np.reshape(eng, (nk,nb)).T-ef

##  Test if it is a metal
for i in range(nb):
    if max(data[i])>0 and min(data[i])<0:
        print 'This is a metal!!'
        sys.exit()
    else:
        continue

c = []; v = []
data = data.T

for i in range(nk):
    for j in range(nb):
        if data[i][j] > 0:
            c = c + [data[i][j]]
            v = v + [data[i][j-1]]
            break

m_dir = min(np.subtract(npa(c),npa(v)))
m_indir = min(npa(c))-max(npa(v))

if m_indir < 0:
    print "There is no band gap!!!"
elif m_dir <= m_indir:
    print "The direct band gap is %s." % m_dir
else:
    print "The indirect band gap is %s." % m_indir

