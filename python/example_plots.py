'''
Standard things to import...
'''
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

'''
Lets import some other new things!
'''
import matplotlib # going to use this to change some settings (next cell)

from matplotlib.colors import LogNorm # this lets us scale color outputs using Log instead of Linear

import matplotlib.cm as cm # this gives access to the standard colormaps (besides rainbow)

'''
this is how you change the default properties of plot text
search Goolge for more examples of changing rcParams to get other fonts, styles, etc
'''
matplotlib.rcParams.update({'font.size':11}) 
matplotlib.rcParams.update({'font.family':'serif'})

# remember how to open FITS tables from last week (or go back and review)
dfile = 'data.fit'

# our data comes from the HIPPARCOS mission: http://adsabs.harvard.edu/abs/1997ESASP1200.....E
# I used Vizier to make a smaller version of the table for ease of reading

hdulist2 = fits.open(dfile)
hdulist2.info() # print the extensions

tbl = hdulist2[1].data # get the data from the 2nd extension
hdulist2.close() # close the file
tbl.columns # print the columns available (can be called by name!)

# you can make plots by calling columns by name!
plt.plot(tbl['col1'], tbl['col2'], alpha=0.2)

'''
Find stars with "good" data
I required errors for B-V greater than 0 and less than or equal to 0.05mag
I required errors on parallax to be greater than 0 and less than or equal to 5

Finally, I required the absolute magnitudes to be real numbers (no Nulls, NaN's, Infs, etc)
'''

# here is most of what you need. Finish it!
ok = np.where((tbl['e_B-V'] <= 0.05) &
              (tbl['e_Plx'] > 0) & 
              np.isfinite(Mv))

plt.figure( figsize=(7,5) ) 
# here's a freebie: I used a 10x8 figsize

plt.hist2d(x, y, 
           bins=(10,20), # set the number of bins in the X and Y direction. You'll have to guess what I used
           norm=LogNorm(), # scale the colors using log, not linear (default)
           cmap = cm.Spectral) # change the colormap

# the B-V color of the Sun is 0.635 mag

# use plt.annotate to put words on the plot, set their colors, fontsizes, and rotation

plt.ylabel('$m_{V}$') # you can put (some) LaTeX math in matplotlib titles/labels

cb = plt.colorbar() # make a colorbar magically appear


# more freebies: this is the exact resolution and padding I used to make the figure file
plt.savefig('FILENAME.png', 
            dpi=300, # set the resolution
            bbox_inches='tight', # make the figure fill the window size
            pad_inches=0.5) # give a buffer so text doesnt spill off side of plot



