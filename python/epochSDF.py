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
if not os.path.exists('_epochexample'):
    os.mkdir('_epochexample')
    download('https://gist.github.com/skuschel/9c78c47130a181a1e2bec13900ab4293/raw/'
           + '6588b96090caa9032a6a9ed8d785f2034f446ad2/epoch_exampledata.tar.xz',
            '_epochexample/epoch_exampledata.tar.xz')

import tarfile
tar = tarfile.open('_epochexample/epoch_exampledata.tar.xz')
tar.extractall('_epochexample/')
print('done.')

get_ipython().run_line_magic('pylab', 'inline')
import postpic as pp
pp.__version__

pp.chooseCode('epoch')
# open a specific dump
dr = pp.readDump('_epochexample/epoch_exampledata/0420.sdf')
print(dr)
# the dumpreader knwos all the information of the simulation
print('The simulations was running on {} spatial dimensions.'.format(dr.simdimensions()))

# if needed, postpic can be bypassed and the underlying datastructure (sdf in this case)
# can be accessed directly via keys
print(dr.keys())
dr['Header']

def addcolorbar(ax, im):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = colorbar(im, cax=cax)
    return cbar

ez = dr.Ex()
print(ez)
fig, ax = subplots()
im = ax.imshow(ez.T, origin='lower', extent=ez.extent*1e6)
addcolorbar(ax, im)
ax.set_xlabel('y [$\mu m$]');
ax.set_ylabel('z [$\mu m$]');

# the multispecies Object is used to access particle data
print(dr.listSpecies())
ms = pp.MultiSpecies(dr, 'electron')
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
nd = ms.createField('x', 'y', bins=[240, 150], shape=3)
# note, that particle weights are included automatically
# and plot
fig, ax = subplots()
im = ax.imshow(nd.T/1e6, origin='lower', extent=nd.extent*1e6)
addcolorbar(ax, im)
ax.set_xlabel('x [$\mu m$]');
ax.set_ylabel('y [$\mu m$]');

# create a Field object qd holding the charge density
# note, that particle weights are included automatically
qd = ms.createField('x', 'y', weights='charge', bins=[240, 150], shape=3)
# and plot
fig, ax = subplots()
im = ax.imshow(qd.T, origin='lower', extent=qd.extent*1e6)
addcolorbar(ax, im)
ax.set_xlabel('x [$\mu m$]');
ax.set_ylabel('y [$\mu m$]');

# average kinetic energy on grid
ekin = ms.createField('x', 'y', weights='Ekin_MeV', bins=[240, 150])
ekinavg = ekin/nd
# and plot
fig, ax = subplots()
im = ax.imshow(ekinavg.T, origin='lower', extent=ekinavg.extent*1e6, vmax=20)
addcolorbar(ax, im)
ax.set_title('Avg kin. Energy on Grid [MeV]')
ax.set_xlabel('x [$\mu m$]');
ax.set_ylabel('y [$\mu m$]');

f = ms.createField('x', 'p', bins=[240, 150])
# and plot
fig, ax = subplots()
im = ax.imshow(f.T, origin='lower', extent=f.extent, aspect='auto', vmax=0.5e41)
addcolorbar(ax, im)
ax.set_xlabel('x [m]');
ax.set_ylabel('p');

f = ms.createField('x', 'gamma', bins=[240, 150])
# and plot
fig, ax = subplots()
im = ax.imshow(f.T, origin='lower', extent=f.extent, aspect='auto', vmax=1e19)
addcolorbar(ax, im)
ax.set_xlabel('z [$\mu m$]');
ax.set_ylabel('$\gamma$');

f = ms.createField('x', 'beta', bins=[240, 150])
# and plot
fig, ax = subplots()
im = ax.imshow(f.T, origin='lower', extent=f.extent, aspect='auto', vmax=5e21)
addcolorbar(ax, im)
ax.set_xlabel('z [$\mu m$]');
ax.set_ylabel(r'$\beta$');

sr = pp.readSim('_epochexample/epoch_exampledata/normal.visit')
print('There are {:} dumps in this simulationreader object.'.format(len(sr)))
print(sr)

for dr in sr:
    print('Simulation time of current dump t = {:.2e} s'.format(dr.time()))

fig, axs = subplots(1,5)
for dr, ax in zip(sr, axs):
    f = dr.Ex()
    im = ax.imshow(f.T, origin='lower', extent=f.extent*1e6)
    #addcolorbar(ax, im)
    ax.set_xlabel('x [$\mu m$]');
    ax.set_ylabel('y [$\mu m$]');
    ax.set_title('{:.2f} fs'.format(dr.time()*1e15))
fig.set_size_inches(13,3)
fig.tight_layout()



