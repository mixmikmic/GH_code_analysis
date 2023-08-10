get_ipython().run_cell_magic('HTML', '', '<iframe width="560" height="315" src="https://www.youtube.com/embed/0JpOJ4F4984" frameborder="0" allowfullscreen></iframe>')

get_ipython().run_cell_magic('HTML', '', '<iframe width="560" height="315" src="https://www.youtube.com/embed/ipyGVh7JKvw" frameborder="0" allowfullscreen></iframe>')

import numpy as np
import random as rd

# to make this notebook's output stable across runs
rd.seed(42)

data0=np.zeros(500)
steps = 10
loc = sorted(rd.sample(xrange(500),steps))
bckg = 1.0
flevel = 10.0
counter = steps
for i in range (0,500,1):
    if i in loc:
        counter -= 1
        data0[i] = flevel*counter + bckg
    else:
        data0[i] = flevel*counter + bckg

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.plot(data0)
plt.xlabel('time (in seconds)')
plt.ylabel('intensity (in a.u.)')
plt.show()

# to make this notebook's output stable across runs
rd.seed(42)

data1=np.zeros(500)
steps = 10
loc = sorted(rd.sample(xrange(500),steps))
bckg = 10.0
flevel = 20.0
std_bckg = 1.0
std_flevel = 2.5
counter = steps
for i in range (0,500,1):
    if i in loc:
        counter -= 1
        data1[i] = counter*rd.gauss(flevel,std_flevel) + rd.gauss(bckg,std_bckg)
    else:
        data1[i] = counter*rd.gauss(flevel,std_flevel) + rd.gauss(bckg,std_bckg)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.plot(data1)
plt.xlabel('time (in seconds)')
plt.ylabel('intensity (in a.u.)')
plt.show()

#We import the Kalafut-Visscher algorithm
from KalafutPUB import KalafutC
#We import some data:
zignal = np.loadtxt("TestDataSet00.txt", unpack=True)  
#This is real experimental data from the lab of Y.Krishnan at U. Chicago. 
#This means we have no idea where the steps actually are.
#We reverse the plot here; see part 4 below for the reason.
signal = zignal[::-1]
Try1 = KalafutC(signal)
stats = Try1.stats
tzero = Try1.tzero

print "number of datapoints:",len(signal)
print "First step:", tzero
print "mean and variance of background:", int(stats[0]),"and",int(stats[1])
print "mean and variance of single fluorophore:", int(stats[2]),"and",int(stats[3])

#%matplotlib inline
#import matplotlib.pyplot as plt
plt.plot(signal)
plt.xlabel('time (in seconds)')
plt.ylabel('intensity (in a.u.)')
plt.show()

mb = np.mean(signal[100:250])
vb = np.var(signal[100:250])
mf = np.mean(signal[280:320]) - mb
vf = np.var(signal[280:500]) - vb
print "mean and variance of background:", int(mb),"and",int(vb)
print "mean and variance of single fluorophore:", int(mf),"and",int(vf)

from SeekerPUB import Slicer

#We pass into the precursor algorithm the data, window size, the number of fluorophores the reversed trace starts with 
#(0, since all have photobleached) and the statistics found by Kalafut-Visscher above.
#Note that the number of data points must always be an integer multiple of the window size (windsz)
windsz = 100
Prelook = Slicer(signal,windsz,0,stats) 

#The precursor algorithm will now divide the 1200 data points by 100, and look into each of the resulting 12 windows.
#It will print "Step 2 'window number' " for each successfully processed window. 
#The 12 steps are numbered in the pythonic way, 0 to 11.

#Here is the approximate number of fluorophores active in each window:
print Prelook.fosfor

#We can use these to see an estimate of what the precursor algorithm thinks the data ought to look like without noise,
#and compare this to the actual data
precurs = np.zeros(len(signal))
for i in range(0,len(precurs),1):
    j = i//windsz
    precurs[i] = Prelook.fosfor[j]*stats[2] +stats[0] 

dx = np.arange(0,len(signal))
    
plt.figure(1)
plt.plot(dx, signal)
plt.plot(dx, precurs, linewidth=3.0)
plt.xlabel('reversed time (in seconds)')
plt.ylabel('intensity (in a.u.)')
plt.show()

from LeffFinderPUB import LbarFind, PriorSlicer
from SeekerPUB import mSICer

#This finds an array of estimates for the parameter λ. The main code then automatically chooses the best estimate
#to use. This parameter plays only a minor role and can be way off without repercussions to our method accuracy.
calba = LbarFind(signal, stats, tzero, 0)                                   
lbar = calba.Lbar                                                
zamm = PriorSlicer(signal, lbar, tzero, 0)   
lefaray = zamm.leffarray  

widd = len(signal)/windsz

fluor = 0
stepsC = np.zeros(1)
for i in range(0,widd,1):
    mSIClook = mSICer(signal, i, fluor, windsz, stats, len(signal), lefaray)                                                                     
    steps_found = mSIClook.SIClocs                                            
    stepsCi = np.zeros_like(steps_found)
    levelz = fluor
    for j in range(0,len(steps_found),1):                                     
        stepsCi[j] = levelz
        levelz += steps_found[j]
    fluor = mSIClook.fluorOUT
    stepsC = np.concatenate((stepsC,stepsCi))  
stepsC = stepsC[1:]

#We can now see the results of our effort:
result = np.zeros(len(signal))
for i in range(0,len(stepsC),1):
    result[i] = stepsC[i]*stats[2] +stats[0] 

plt.figure(1)
plt.plot(dx, signal)
plt.plot(dx, result, linewidth=3.0)
plt.xlabel('reversed time (in seconds)')
plt.ylabel('intensity (in a.u.)')
plt.show()

#We can also count the number of steps:
diffcount = np.diff(stepsC)
acount = np.where(diffcount!=0)
stepcount = len(acount[0])

print "number of steps:", stepcount
#and find what the maximum number of fluorophores that were ever active was:
maxfluors = np.nanmax(stepsC)
print "total fluorophores:", int(maxfluors)

