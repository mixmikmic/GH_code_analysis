get_ipython().magic('matplotlib inline')
import os
import re
import sys
import numpy
import pickle

sys.path.append('/Users/mpagani/Projects/hmtk/')
sys.path.append('/Users/mpagani/Projects/original/oq-engine/')

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

cat = pickle.load(open( "./../data/catalogue_ext_cac.p", "rb" ))

fin = open('./../data/trench.xy', 'r')
trench = []
for line in fin: 
    aa = re.split('\s+', re.sub('^\s+', '', line))
    trench.append((float(aa[0]), float(aa[1])))
fin.close()
trc = numpy.array(trench)

def plot_gmt_multi_line(proj, filename):
    trenches = [] 
    fin = open(filename, 'r')
    trench = []
    for line in fin: 
        if re.search('^>', line):
            if len(trench):
                trc = numpy.array(trench)
                trenches.append(trc)
                x, y = proj(trc[:,0], trc[:,1])
                plt.plot(x, y, '--r') 
            name = line
            trench = []
        else:
            aa = re.split('\s+', re.sub('^\s+', '', line))
            trench.append((float(aa[0]), float(aa[1])))
    fin.close()

import matplotlib.patheffects as PathEffects

fig = plt.figure(figsize=(15,10))

midlo = (min(cat.data['longitude'])+max(cat.data['longitude']))/2
midla = (min(cat.data['latitude'])+max(cat.data['latitude']))/2
minlo = min(cat.data['longitude'])
minla = min(cat.data['latitude'])
maxlo = max(cat.data['longitude'])
maxla = max(cat.data['latitude'])

m = Basemap(llcrnrlon=minlo, llcrnrlat=minla,
            urcrnrlon=maxlo, urcrnrlat=maxla,
            resolution='i', projection='tmerc', 
            lon_0=midlo, lat_0=midla)

m.drawcoastlines()
x, y = m(cat.data['longitude'], cat.data['latitude'])
plt.plot(x, y, 'x')

# Plotting large earthquakes
idx = numpy.nonzero((cat.data['magnitude'] > 7.4) & (cat.data['year'] > 1990))
plt.plot(x[idx], y[idx], 'or')
mags = cat.data['magnitude']
years = cat.data['year']
effect = [PathEffects.withStroke(linewidth=3,foreground="w")]
for iii in idx[0]:
    lab = '%.1f - %d' % (mags[iii], years[iii])
    plt.text(x[iii], y[iii], lab, path_effects=effect)

# Parallels
delta = 10
parallels = numpy.arange(numpy.floor(minla/delta)*delta,
                         numpy.ceil(maxla/delta)*delta, delta)
# labels = [left,right,top,bottom]
m.drawparallels(parallels, labels=[False,True,True,False])
meridians = numpy.arange(numpy.floor(minlo/delta)*delta,
                         numpy.ceil(maxlo/delta)*delta, delta)
# labels = [left,right,top,bottom]
m.drawmeridians(meridians, labels=[True, False, False, True])

# Plotting trench axis
x, y = m(trc[:,0], trc[:,1])
plt.plot(x, y, '--b', linewidth=2)
plot_gmt_multi_line(m, './trench.gmt')



