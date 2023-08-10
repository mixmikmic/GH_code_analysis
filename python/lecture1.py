print 1+13

my_name = "Tom"

print "Hello " + my_name

x=1

print x+1

#words behind a hashtag are comments

#enter your information here
my_height_feet=5
my_height_inches=12

#now convert feet and inches to just inches
my_height_just_inches=my_height_feet*12 + my_height_inches
print "my height in inches = " + str(my_height_just_inches)

#now convert from inches to cm
my_height_cm = my_height_just_inches * 2.54
print "my height in cm = " + str(my_height_cm)

''' (three apostrophes defines a block of comments)

#This generates some random numbers if you just want to see the graph
import numpy as np
class_size=41
h=np.random.randn(class_size)*6.1+162
'''



h=[177.8	,
163.83	,
172.72	,
160	,
172.72	,
182.88	,
187.96	,
172.72	,
182.88	,
157.48	,
186.7	,
172.72	,
187.96	,
162.56	,
132.08	,
169	,
157	,
175	,
180.34	,
168	,
173.7	]

print "height data = " + str(h)

#In python if you want to do something complicated, you often import some new commands (called a library)

import pylab as plt #get graph plotting commands from pylab library
#draw graphs in this notebook rather than in separate windows
get_ipython().magic('matplotlib inline')

plt.hist(h) #draw a histrogram
plt.xlabel('height') #label the x axis
plt.ylabel('frequency') #label the y axis
plt.title('Histogram of class heights') #give it a title

from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "http://www.neuroscience.cam.ac.uk/uploadedFiles/pcb10_phppYaWHE.jpg", width=600)

