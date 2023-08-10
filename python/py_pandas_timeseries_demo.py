#imports (yawn) we'll get them out of the way
import numpy as np              #python's array proccesing / linear algebra library
import pandas as  pd            #data processing / stats library
import matplotlib.pyplot as plt #data visualization
import matplotlib.dates as dates
import csv
import datetime
from py_utils import printme 	#home-made formatting utilities
#$ipython notebook    #strart a server on 8888

#directive to allow in-line plotting 
get_ipython().magic('matplotlib inline')

#... and a few marker codes cf. matplotlib.org/api/colors_api.html
red='r'; blue='b'; green='g'; greenish='chartreuse'; magenta='m';black='b'
circle='o'; x='x';

#set this to True to show all the plots; False for dev/debugging
live=True

#get some stuff to plot - permutations of a sine wave here...
#   ... these use list comprehensions and scalar operations
s1=pd.Series([np.sin(x/10) for x in range(0, 300)])
s2=s1.copy()
s2.index=s2.index-50
s3=s1*2
s1=pd.Series([np.cos(x/10) for x in range(0, 300)])

#create a new figure (high-level container), add a plot to it
plt.figure()
plt.plot(s1)
if live: plt.show()

plt.figure()
plt.plot(s1, red+circle, s1, black,
         s2, green+x, s2, blue,
         s3, greenish)
if live: plt.show()
plt.close()

rows=2; cols=2 #subplots fill row-wise

#we'll stack 'em in one at a time
plt.subplot(rows, cols, 1)
plt.plot(s1, red+circle, s1, black,
         s2, green+x, s2, blue)

plt.subplot(rows, cols, 2)
plt.plot(s2, green+x, s2, blue)

plt.subplot(rows, cols, 3)
plt.plot(s3, greenish, s1, black+circle)

plt.subplot(rows, cols, 4)
plt.scatter(s1, s2)

if live: plt.show()
plt.close()







