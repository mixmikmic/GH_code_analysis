def download(url, file):
    import urllib3
    import shutil
    import os
    if os.path.isfile(file):
        return
    urllib3.disable_warnings()
    http = urllib3.PoolManager()
    print('downloading {:} ...'.format(file))
    with http.request('GET', url, preload_content=False) as r, open(file, 'wb') as out_file:
        shutil.copyfileobj(r, out_file)

# download the example data
import os
if not os.path.exists('_openPMDdata'):
    os.mkdir('_openPMDdata')
    download('https://github.com/openPMD/openPMD-example-datasets/'
        + 'raw/776ae3a96c02b20cfae56efafcbda6ca76d4c78d/example-2d.tar.gz',
            '_openPMDdata/example-2d.tar.gz')

import tarfile
tar = tarfile.open('_openPMDdata/example-2d.tar.gz')
tar.extractall('_openPMDdata')
print('done.')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import postpic as pp
pp.__version__

pp.chooseCode('openpmd')
# open a specific dump
dr = pp.readDump('_openPMDdata/example-2d/hdf5/data00000300.h5')
print(dr)
# the dumpreader knwos all the information of the simulation
print('The simulations was running on {} spatial dimensions.'.format(dr.simdimensions()))

# if needed, postpic can be bypassed and the underlying datastructer (h5 in this case)
# can be accessed directly via keys
print(dr.keys())
dr['fields']

ez = dr.Ez()
print(ez)
fig, ax = plt.subplots()
ax.imshow(ez.T, origin='lower', extent=ez.extent*1e6)
ax.set_xlabel('y [$\mu m$]');
ax.set_ylabel('z [$\mu m$]');

# the multispecies Object is used to access particle data
print(dr.listSpecies())
ms = pp.MultiSpecies(dr, 'electrons')
# now, ms is representing the species "electrons"
print(ms)

# we can access the properties for each individual particle, like the x coordinate
x = ms('x')
print(len(x))

# or do something more complicated such as:
pr = ms('sqrt(px**2 + py**2)')
# actually ridiculous things will work:
pr = ms('x + x**2 + (gamma**2 - 2) - sin(px/pz)')
# you can look at  for a list of predefined values
pp.particle_scalars

# we can use the particle properties to create a Field for plotting.
# Particle Shapes as in the Simulation will be included
# calculate the number density
nd = ms.createField('z', 'x', bins=[200, 50])
# note, that particle weights are included automatically
# and plot
fig, ax = plt.subplots()
ax.imshow(nd.T, origin='lower', extent=nd.extent*1e6)
ax.set_xlabel('z [$\mu m$]');
ax.set_ylabel('x [$\mu m$]');

# create a Field object nd holding the charge density
# note, that particle weights are included automatically
qd = ms.createField('z', 'x', weights='charge', bins=[200, 50])
# and plot
fig, ax = plt.subplots()
ax.imshow(qd.T, origin='lower', extent=qd.extent*1e6)
ax.set_xlabel('z [$\mu m$]');
ax.set_ylabel('x [$\mu m$]');

# average kinetic energy on grid
ekin = ms.createField('z', 'x', weights='Ekin_MeV', bins=[200, 50])
ekinavg = ekin/nd
# and plot
fig, ax = plt.subplots()
ax.imshow(ekinavg.T, origin='lower', extent=ekinavg.extent*1e6)
ax.set_xlabel('z [$\mu m$]');
ax.set_ylabel('x [$\mu m$]');

f = ms.createField('z', 'p', bins=[200,50])
# and plot
fig, ax = plt.subplots()
ax.imshow(f.T, origin='lower', extent=f.extent, aspect='auto')
ax.set_xlabel('z [m]');
ax.set_ylabel('p');

f = ms.createField('z', 'gamma', bins=[200,50])
# and plot
fig, ax = plt.subplots()
ax.imshow(f.T, origin='lower', extent=f.extent, aspect='auto')
ax.set_xlabel('z [$\mu m$]');
ax.set_ylabel('$\gamma$');

f = ms.createField('z', 'beta', bins=[200,50])
# and plot
fig, ax = plt.subplots()
ax.imshow(f.T, origin='lower', extent=f.extent, aspect='auto')
ax.set_xlabel('z [$\mu m$]');
ax.set_ylabel(r'$\beta$');

sr = pp.readSim('_openPMDdata/example-2d/hdf5/*.h5')
print('There are {:} dumps in this simulationreader object.'.format(len(sr)))
print(sr)

for dr in sr:
    print('Simulation time of current dump t = {:.2e} s'.format(dr.time()))

fig, axs = plt.subplots(1,5)
for dr, ax in zip(sr, axs):
    f = dr.Ez()
    ax.imshow(f.T, origin='lower', extent=f.extent*1e6)
    ax.set_xlabel('y [$\mu m$]');
    ax.set_ylabel('z [$\mu m$]');
    ax.set_title('{:.2f} fs'.format(dr.time()*1e15))
fig.set_size_inches(13,3)
fig.tight_layout()



