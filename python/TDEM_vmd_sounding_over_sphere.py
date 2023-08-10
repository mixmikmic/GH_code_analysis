# Basic python packages
import numpy as np 
import matplotlib.pyplot as plt
from scipy.constants import mu_0

# Modules of SimPEG we will use for forward modelling
from SimPEG import Mesh, Utils, Maps
from SimPEG.EM import TDEM
from SimPEG import SolverLU as Solver

# Set a nice colormap! 
plt.set_cmap(plt.get_cmap('viridis'))

get_ipython().magic('matplotlib inline')

# electrical conductivities in S/m
sig_halfspace = 1e-6
sig_sphere = 1e-2
sig_air = 1e-8

# depth to center, radius in m
sphere_z = -50.
sphere_radius = 30. 

boom_height = 20. 
rx_offset = 8. 

times = np.logspace(-7, -4, 30)

# source and receiver location in 3D space
src_loc = np.r_[0., 0., boom_height]
rx_loc = np.atleast_2d(np.r_[rx_offset, 0., boom_height])

# print the min and max diffusion distances to make sure mesh is 
# fine enough and extends far enough 

def diffusion_distance(sigma, time):
    return 1.28*np.sqrt(time/(sigma * mu_0))

print(
    'min diffusion distance: {:.2e} m'.format(diffusion_distance(sig_sphere, times.min()))
)
print(
    'max diffusion distance: {:.2e} m'.format(diffusion_distance(sig_halfspace, times.max()))
)

# x-direction
csx = 2  # core mesh cell width in the x-direction
ncx = np.ceil(1.2*sphere_radius/csx)  # number of core x-cells (uniform mesh slightly beyond sphere radius)
npadx = 30  # number of x padding cells

# z-direction
csz = 1  # core mesh cell width in the z-direction
ncz = np.ceil(1.2*(boom_height - (sphere_z-sphere_radius))/csz) # number of core z-cells (uniform cells slightly below bottom of sphere)
npadz = 32  # number of z padding cells

# padding factor (expand cells to infinity)
pf = 1.3

# cell spacings in the x and z directions
hx = Utils.meshTensor([(csx, ncx), (csx, npadx, pf)])
hz = Utils.meshTensor([(csz, npadz, -pf), (csz, ncz), (csz, npadz, pf)])

# define a SimPEG mesh
mesh = Mesh.CylMesh([hx, 1, hz], x0 = np.r_[0.,0., -hz.sum()/2.-boom_height])

# X and Z limits we want to plot to. Try 
xlim = np.r_[0., 2.5e4]
zlim = np.r_[-2.5e4, 2.5e4]

fig, ax = plt.subplots(1,1)
mesh.plotGrid(ax=ax)

ax.set_title('Simulation Mesh')
ax.set_xlim(xlim)
ax.set_ylim(zlim)

print(
    'The maximum diffusion distance (in background) is: {:.2e} m. '
    'Does the mesh go sufficiently past that?'.format(
        diffusion_distance(sig_halfspace, times.max())
    )
)

# create a vector that has one entry for every cell center
sigma = sig_air*np.ones(mesh.nC)  # start by defining the conductivity of the air everwhere
sigma[mesh.gridCC[:,2] < 0.] = sig_halfspace  # assign halfspace cells below the earth

sigma_background = sigma.copy()

# indices of the sphere (where (x-x0)**2 + (z-z0)**2 <= R**2)
sphere_ind = (mesh.gridCC[:,0]**2 + (mesh.gridCC[:,2] - sphere_z)**2) <= sphere_radius**2 
sigma[sphere_ind] = sig_sphere  # assign the conductivity of the sphere

# Plot a cross section of the conductivity model
fig, ax = plt.subplots(1,1)
cb = plt.colorbar(mesh.plotImage(np.log10(sigma), ax=ax, mirror=True)[0])

# plot formatting and titles
cb.set_label('$\log_{10}\sigma$', fontsize=13)
ax.axis('equal')
ax.set_xlim([-120., 120.])
ax.set_ylim([-100., 30.])
ax.set_title('Conductivity Model')

# Define the receivers, we will sample the real secondary magnetic flux density as well as the 
# imaginary magnetic flux density 

dbdt_z = TDEM.Rx.Point_dbdt(locs=rx_loc, times=times, orientation='z')  # vertical db_dt
rxList = [dbdt_z]  # list of receivers

# Define the list of sources - one source for each frequency. The source is a point dipole oriented
# in the z-direction

srcList = [
    TDEM.Src.MagDipole(
        rxList, loc=src_loc, orientation='z', waveform=TDEM.Src.StepOffWaveform()
    )
]

print(
    'There is {nsrc} source. Each source has {nrx} receivers '
    'sampling the resulting db/dt-field at {ntimes} times'.format(
        nsrc = len(srcList), 
        nrx = len(rxList),
        ntimes = len(times)
    )
)

# solve the problem at these times
timeSteps = [(1e-8, 20), (1e-7, 20), (2e-7, 10), (1e-6, 10), (2e-6, 10), (1e-5, 10)] 

# define a problem - the statement of which discrete pde system we want to solve
prob = TDEM.Problem3D_e(mesh, timeSteps = timeSteps, sigmaMap=Maps.IdentityMap(mesh))  
prob.solver = Solver

survey = TDEM.Survey(srcList)

# tell the problem and survey about each other - so the RHS can be constructed for 
# the problem and the resulting fields and fluxes can be sampled by the receiver. 
prob.pair(survey)  

print('The maximum time is {:1.1e}'.format(prob.times[-1]))

get_ipython().run_cell_magic('time', '', "print('solving with sphere ... ')\nfields = prob.fields(sigma)\nprint('... done ')")

get_ipython().run_cell_magic('time', '', "print('solving without sphere ... ')\nfields_background = prob.fields(sigma_background)\nprint('... done ')")

# log-scale the colorbar
from matplotlib.colors import LogNorm 
import ipywidgets

def plot_dbdtSphere(
    time_ind=0 # which frequency would you like to look at?
#     ax=ax
):
    fig, ax = plt.subplots(1,1, figsize=(6,5))
    
    # Plot the magnetic flux
    field_to_plot = fields[srcList, 'dbdt', time_ind]
    max_field = np.abs(field_to_plot).max() #use to set colorbar limits
    cb_range = 5e2  # dynamic range of colorbar
    
    cb = plt.colorbar(mesh.plotImage(
        field_to_plot, 
        vType='F', view='vec', 
        range_x=[-120., 120.], range_y=[-180., 80.],
        pcolorOpts={
                'norm': LogNorm(), 
                'cmap': plt.get_cmap('viridis')
            },
        streamOpts={'color': 'w'}, mirror=True, ax=ax, 
        clim=[max_field/cb_range, max_field]
    )[0], ax=ax)

    ax.set_xlim([-120., 120.])
    ax.set_ylim([-180., 70.])
    cb.set_label('|db/dt|')

    # plot the outline of the sphere
    x = np.linspace(-sphere_radius, sphere_radius, 100)
    ax.plot(x, np.sqrt(sphere_radius**2 - x**2) + sphere_z, color='k')
    ax.plot(x, -np.sqrt(sphere_radius**2 - x**2) + sphere_z, color='k')

    # plot the source and receiver locations
    ax.plot(src_loc[0],src_loc[2],'co', markersize=6)
    # ax.plot(rx_loc[0,0],rx_loc[0,2],'co', markersize=6)

    # plot the surface of the earth
    ax.plot(np.r_[-200, 200], np.r_[0., 0.], 'w:')

    # give it a title
    ax.set_title(
        'db/dt, {time:10.2e} s'.format( 
            time=prob.times[time_ind]
        )
    )
    plt.show()
    return ax

ipywidgets.interact(
    plot_dbdtSphere, 
    time_ind=ipywidgets.IntSlider(min=1, max=len(prob.timeSteps)-1, value=1) 
)

dpred = survey.dpred(sigma, f=fields)
dpred_background = survey.dpred(sigma_background, f=fields_background)

# Plot
fig, ax = plt.subplots(1,1)

ax.loglog(1e3*times, -dpred)
ax.loglog(1e3*times, -dpred_background, '--k')
ax.grid(True, color='k',linestyle="-", linewidth=0.1)
ax.legend(['with sphere', 'background'])

ax.set_title('Sounding over Sphere')
ax.set_ylabel('-$db_z/dt$')
ax.set_xlabel('time (ms)')



