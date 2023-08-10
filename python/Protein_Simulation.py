import ionize
from ionize import Protein
from Bio import PDB
import numpy
from matplotlib.pyplot import *
get_ipython().magic('matplotlib inline')

DB = PDB.PDBList().get_all_entries()

pH = numpy.linspace(0, 14)
x=[]
y=[]
for idx, entry in enumerate(DB): 
    try:
        my_prot = Protein(entry)
        mob = [my_prot.mobility(p) for p in pH]
        print(mob)
        x.extend(pH)
        y.extend(mob)
        plot(pH, mob)
    except Exception as e:
        print(e)
    if idx>=1000:
        break
ylim(-1e-8, 1e-8)
show()

nbins = 500
H, xedges, yedges = numpy.histogram2d(x,y,bins=nbins)
 
# H needs to be rotated and flipped
H = numpy.rot90(H)
H = numpy.flipud(H)
 
# Mask zeros
Hmasked = numpy.ma.masked_where(H==0,H) # Mask pixels with a value of zero
 
# Plot 2D histogram using pcolor
fig2 = figure()
pcolormesh(xedges,yedges,Hmasked)
xlabel('pH')
ylabel('mobility (m^2/V/s)')
cbar = colorbar()
cbar.ax.set_ylabel('Counts')
ylim(-1e-8, 1e-8)
show()



