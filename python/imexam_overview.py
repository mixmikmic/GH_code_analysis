#show plots in the notebook, and make them interactive
#This is the best solution if you are going to use the interactive image examination loop, imexam() 
#remember to close the interaction by clicking on the plots upper right close button
get_ipython().magic('matplotlib notebook')

import imexam

imexam.display_help() #pull up the help documents for the installed version

a=imexam.connect() #starts up a DS9 window by default when no options are specified

a.load_fits('/Users/sosey/test_images/iacs01t4q_flt.fits') #display fits image in ds9

a.scale() #scale the image using a zscale algorithm (the default)

a.grab() # Optional if you want to save a view in the notebook for reference

a.imexam()

a.jimexam()

get_ipython().magic('pinfo a.set_plot_pars')

a.eimexam()

#You can customize the plotting parameters (or any function in the imexam loop)
a.set_plot_pars('e','title','This is my favorite galaxy')
a.set_plot_pars('e','ncontours',4)
a.set_plot_pars('e','cmap','YlOrRd') #see http://matplotlib.org/users/colormaps.html

a.imexam()

a.unlearn() #you can always go back to the default plot settings
a.eimexam()

#maybe we want to change the colormap on the DS9 display? You can see the available maps:
a.cmap()

a.cmap(color='heat')

data=a.get_data() #grab the data to play with it from the command line

data

#you can also get the header, which will be returned as a string
header=a.get_header()

print(header)

#any numpy array can be displayed in the viewer
import numpy as np
data=np.random.rand(100,100) 
a.view(data)
a.zoomtofit()

a.get_viewer_info() #display information on what is being displayed in the viewer

a.frame(2)#open another frame. This can also be used to switch frames

#or you can use astropy nddata arrays (which really are numpy arrays with meta data)
from astropy.nddata import NDData
array = np.random.random((12, 12, 12))  # a random 3-dimensional array
ndd = NDData(array)
a.view(ndd.data[5])
a.zoom()

a.close() #disconnect and close DS9 window. This only works for DS9 process started from imexam

get_ipython().magic('matplotlib notebook')
import imexam

a=imexam.connect(viewer='ginga') #stars up a new tab with the ginga HTML5 viewer

a.load_fits('/Users/sosey/test_images/iacs01t4q_flt.fits') #display fits image in a separate browser window, same as in DS9

#No list of commands is printed with the event driven imexam, but you can always 
#see what the available commands are by issuing the imexam() call:
a.imexam()

# you can  save a copy of the current viewing window
a.window.ginga_view.show()

#besides making plots you can also get basic aperture photometry using the "a" key, try that now

#if you are using the ginga viewer, you can return the full Ginga image object and use any
#of the methods which are enabled for it. You can look here for the Ginga quick reference: 
#http://ginga.readthedocs.org/en/latest/quickref.html
img=a.get_image()

type(img)

img.height, img.width, img.pixtoradec(100,100)

img.pixtoradec(100,100)

canvas=a.window.ginga_view.add_canvas()
canvas.delete_all_objects()
canvas.set_drawtype('rectangle')

#now you can go to the viewer and draw a rectangle selection box

a.window.ginga_view.show()

from ginga.util import iqcalc
iq = iqcalc.IQCalc()

#find all the peaks in the rectangle area
r = canvas.objects[0]
data = img.cutout_shape(r)
peaks = iq.find_bright_peaks(data)

peaks[-10:] #show the last 10 peaks detected in the cutout

objs = iq.evaluate_peaks(peaks, data)

o1=objs[0]
o1

# pixel coords are for cutout, so add back in origin of cutout
#  to get full data coords RA, DEC of first object
x1, y1, x2, y2 = r.get_llur()
img.pixtoradec(x1+o1.objx, y1+o1.objy)

# Draw circles around all objects
Circle = canvas.get_draw_class('circle')
stars=[]
for obj in objs:
    x, y = x1+obj.objx, y1+obj.objy
    if r.contains(x, y):
        canvas.add(Circle(x, y, radius=10, color='yellow'))
        stars.append((x,y))
        
        
# set pan and zoom to center
a.panto_image((x1+x2)/2, (y1+y2)/2)
a.window.ginga_view.scale_to(0.75, 0.75)

a.window.ginga_view.show()

#lets look at one of the stars closer
a.zoom(6)
a.panto_image(stars[0][0],stars[0][1])

a.close() #for ginga, there isn't an automatic window close for the HTML5 canvas, this will just stop the server

a.reopen() #but if you close the window by hand and want it back, you can reopen it!

#Ginga also has colormaps
a.cmap(color="spiffy")

a.cmap(color='smooth2')

b=imexam.connect(viewer='ginga', port=5478)

b.view(data)

from imexam.imexamine import Imexamine
from astropy.io import fits

plots=Imexamine() #the plots object now has all the imexam functions available

#now, grab some data, associate it with the plot object
data=fits.getdata('/Users/sosey/test_images/iacs01t4q_flt.fits')
plots.set_data(data)

radii,flux=plots.radial_profile(532,415,genplot=False) #save the radial profile data

plots.radial_profile(532,415) #save the radial profile data

radii

flux

#now, if you decide to make a plot, you can still do that
import matplotlib.pyplot as plt

plt.plot(radii,flux,'D')

plots.aper_phot(532,415) #just return the photometry

cog=plots.curve_of_growth(532,415,genplot=False) #curve of growth

cog #just an example illustrating that functions return tuples

#you can separate them afterwards too:
radius,flux=cog
print(radius)
print(flux)

gauss=plots.line_fit(532,415,genplot=False) #return the gaussian fit model

gauss.stddev #you can check out the model parameters

#if you want the fwhm you can pull in the function imexam uses or create your own
from imexam import math_helper
math_helper.gfwhm(gauss.stddev) 



