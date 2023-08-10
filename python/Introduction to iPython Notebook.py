get_ipython().magic('pylab inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
#this is just a setting that lets us plot figures within the notebook. Don't worry about this for now

print ('I can count to 10. Watch!\n')

for x in range(1,11):
    print (str(x))

#here are some common modules:
import scipy as sp #library of scientific functions
import scipy.io 
from scipy.signal import welch
import numpy as np #library of math functions
import pandas as pd #library of data analysis functions
import matplotlib.pyplot as plt #functions to plot data
import os #This lets python talk to your opperating system to open and save files.

filename = 'emodat.mat'; #just a string for the name of the file

filename = os.path.join('./', filename) #this is how we tell python where the file is

datafile = sp.io.loadmat(filename) #this is how we load the file.
#there are lots of variables in this .mat file. Let's just look at the voltages at each time point. 
#In this particular file, that is stored as 'data'. 
data = datafile['data']; 
data = data[0];
print 'There are ' + str(len(data)) +' time samples in this data';

plt.plot(data[0:1000]); #this is what about 1 second of ECoG data looks like



