get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages



data=pd.read_csv('CoreEM09GC01.csv')

#This is my plot with axis from 0-50 cm
fig,ax=plt.subplots()
fig.set_size_inches(4,8)
ax.scatter(data.Pb,data.DepthCm)
ax.set_ylim([50,0])

#This makes a second axis so we can add the years.  I makee a twin axis and then give it the range in years.  
ax2=ax.twinx()
ax2.set_ylim([1936,2009.6])

#This reads in an image.  We set it to im.  Then we plot that image in its owns as axis.  This way we can show the core picture!
im = plt.imread('EM09-GC01-Cropped.jpg')
newax = fig.add_axes([1.0, 0.10, 0.85, 0.85])  #x,y,width and height.  
newax.imshow(im)
newax.axis('off')

data=pd.read_csv('CoreEM09GC01.csv')

#This makes our first plot.
fig,ax=plt.subplots()
fig.set_size_inches(4,8)
ax.plot(data.Fe,data.DepthCm,'.g-',markersize=10)
ax.set_ylim([50,0])
ax.set_xlabel('Iron (mg/Kg)',color='g')
ax.set_ylabel('Depth (cm)')
ax.locator_params(nbins=4,axis='x') #I did this to limit the number of tick marks.  


#This makes the second set with a twiny axis.  So it basically duplictes the y axis and then you an plot a new x.   
# This is probably all you need. It gives you a second axis with a anice plt in just a couple lines.  
ax2=ax.twiny()
ax2.plot(data.Mn,data.DepthCm,'.r-',markersize=10)
ax2.set_xlabel('Manganese (mg/Kg)',color='r')

#To get really fancy I added a third parameter and made a third offset axis at the bottom.
# You need the two extra lines to force it to offset at the bottom.  
ax3=ax.twiny()
ax3.plot(data.Bi,data.DepthCm,'.b-',markersize=10)
ax3.spines["bottom"].set_position(("axes", -.1))
ax3.set_xlabel('Bismuth (mg/kg)',color='b')
ax3.xaxis.set_ticks_position('bottom')
ax3.xaxis.set_label_position('bottom')

#make a fourth axis on top....
ax4=ax.twiny()
ax4.plot(data.Cl,data.DepthCm,'.y-',markersize=10)
ax4.spines["top"].set_position(("axes", 1.1))
ax4.set_xlabel('Chloride (mg/kg)',color='y')

ax4.set_title('This adds an offset title',y=1.2)

data=pd.read_csv('CoreEM09GC01-extra-line.csv')

#This makes our first plot.
fig,ax=plt.subplots()
fig.set_size_inches(4,8)
ax.plot(data.Fe,data.DepthCm,'.g-',markersize=10)
ax.set_ylim([50,0])
ax.set_xlabel('Iron (mg/Kg)',color='g')
ax.set_ylabel('Depth (cm)')
ax.locator_params(nbins=4,axis='x') #I did this to limit the number of tick marks.  


#This makes the second set with a twiny axis.  So it basically duplictes the y axis and then you an plot a new x.   
# This is probably all you need. It gives you a second axis with a anice plt in just a couple lines.  
ax2=ax.twiny()
ax2.plot(data.Mn,data.DepthCm,'.r-',markersize=10)
ax2.set_xlabel('Manganese (mg/Kg)',color='r')

#This is a horizontal line
ax.hlines(40,0,20000,colors='grey')

#You could make the line really thick to hide certain areas and make a box....
ax.hlines(50,0,20000,colors='grey',linewidth=120)





