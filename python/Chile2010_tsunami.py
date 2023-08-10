get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from numpy import ma # masked arrays
from __future__ import print_function

from data_tools import read_asc_file

topo_file = '../geoclaw_output/chile2010/topo.asc'
topo_data_dict = read_asc_file(topo_file)

X = topo_data_dict['X']
Y = topo_data_dict['Y']
topo = topo_data_dict['values']

fig = plt.figure(figsize=(5,5))
ax = plt.axes()

#plt.contour(X,Y,topo, np.linspace(-5000,0,6), colors='b', linestyles='-')
#plt.contour(X,Y,topo, np.linspace(0,5000,6), colors='g', linestyles='-')
plt.contourf(X,Y,topo)
cb = plt.colorbar()
cb.set_label('meters')
ax.set_aspect(1./np.cos(30*np.pi/180.))
ax.ticklabel_format(format='plain',useOffset=False)
plt.xticks(rotation=20);
plt.title('Topography / Bathymetry')

frame_times = np.linspace(0,5.5,12)
print("The data is at the following times (in hours after the earthquake):\n   ", frame_times)

frameno = 9
filename = 'eta_%s.asc' % str(frameno).zfill(4)  # pads with zeros to fill 4 characters
eta_file = '../geoclaw_output/chile2010/%s' % filename
print('Reading file %s\nTsunami simulation at time %4.1f hours after quake\n' % (eta_file, frame_times[frameno]))
eta_data_dict = read_asc_file(eta_file)
Eta = eta_data_dict['values']

plt.contourf(X,Y,Eta)
plt.colorbar()

Eta_offshore = ma.masked_where(topo>0, Eta)

fig = plt.figure(figsize=(5,5))
ax = plt.axes()
plt.contourf(X,Y,Eta_offshore)
cb = plt.colorbar()
cb.set_label('meters elevation relative to sea level')

ax.set_aspect(1./np.cos(30*np.pi/180.))
ax.ticklabel_format(format='plain',useOffset=False)
ax.set_title('%4.1f hours after quake' % frame_times[frameno])

eta_colors = [[0,0,1], [.5,.5,1], [0,1,1], [1,.5,.5], [1,0,0]]
eta_levels = [-0.5,-0.1,-0.05,0.05,0.1,0.5]

fig = plt.figure(figsize=(5,5))
ax = plt.axes()

plt.contourf(X,Y,Eta_offshore, eta_levels, colors=eta_colors, extend='both')

cb = plt.colorbar()
cb.set_label('meters elevation relative to sea level')

# also plot the coastline in green:
# (do this *after* making the colorbar for the contourf plot)
plt.contour(X,Y,topo,[0], colors='g')

ax.set_aspect(1./np.cos(30*np.pi/180.))
ax.ticklabel_format(format='plain',useOffset=False)
ax.set_title('%4.1f hours after quake' % frame_times[frameno])

from mpl_toolkits.basemap import Basemap, shiftgrid, cm

m = Basemap(projection='ortho',lon_0=-80,lat_0=-20,resolution='l')

m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary(fill_color='aqua')

# draw parallels and meridians.
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,420.,60.))

plt.title("Full Disk Orthographic Projection")
plt.show()

Xortho, Yortho = m(X,Y)

m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,420.,30.))
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')

plt.contourf(Xortho,Yortho,Eta_offshore, eta_levels,              colors=eta_colors, extend='both');
cb = plt.colorbar()
cb.set_label('meters elevation relative to sea level')
plt.title('%4.1f hours after quake' % frame_times[frameno])

for frameno in [0,4,8]:
    eta_file = '../geoclaw_output/chile2010/eta_%s.asc'                 % str(frameno).zfill(4)
    eta_data_dict = read_asc_file(eta_file, verbose=False)
    Eta = eta_data_dict['values']
    Eta_offshore = ma.masked_where(topo>0, Eta)
    
    plt.figure()
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,30.))
    m.drawcoastlines()
    m.fillcontinents(color='coral',lake_color='aqua')
    plt.contourf(Xortho,Yortho,Eta_offshore, eta_levels,                  colors=eta_colors, extend='both')
    plt.title('%4.1f hours after quake' % frame_times[frameno])

import animation_tools

figs = []  # to collect set of figures we will animate

m = Basemap(projection='ortho',lon_0=-80,lat_0=-20,resolution='l')
Xortho, Yortho = m(X,Y)

for frameno in range(12):
    eta_file = '../geoclaw_output/chile2010/eta_%s.asc'                 % str(frameno).zfill(4)
    eta_data_dict = read_asc_file(eta_file, verbose=False)
    Eta = eta_data_dict['values']
    Eta_offshore = ma.masked_where(topo>0, Eta)
    
    fig = plt.figure()
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,30.))
    m.drawcoastlines()
    m.fillcontinents(color='coral',lake_color='aqua')
    plt.contourf(Xortho,Yortho,Eta_offshore, eta_levels,                  colors=eta_colors, extend='both')
    plt.title('%4.1f hours after quake' % frame_times[frameno])
    figs.append(fig)
    plt.close(fig)
    

animation_tools.interact_animate_figs(figs)

images = animation_tools.make_images(figs, dpi=150)

animation_tools.JSAnimate_images(images, figsize=(8,4))

anim = animation_tools.JSAnimate_images(images, figsize=(10,8))

file_name = 'chile2010_on_sphere.html'
animation_tools.make_html(anim, file_name=file_name, title="Chile 2010 Tsunami", 
                          raw_html="Illustration of html file created by <tt>make_html</tt>")

m = Basemap(projection='robin',lon_0=-180,resolution='c')
Xortho, Yortho = m(X,Y)

m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,420.,30.))
m.drawcoastlines()
m.fillcontinents(color='coral')
plt.contourf(Xortho,Yortho,Eta_offshore, eta_levels, colors=eta_colors)
plt.title("Robinson Projection")

