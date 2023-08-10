import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

from devito import configuration
configuration['log_level'] = 'WARNING'

# Define true and initial model
from examples.seismic import demo_model, plot_velocity, plot_perturbation

shape = (101, 101)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # Need origin to define relative source and receiver locations

model = demo_model('circle-isotropic', vp=3.0, vp_background=2.5,
                    origin=origin, shape=shape, spacing=spacing, nbpml=40)

model0 = demo_model('circle-isotropic', vp=2.5, vp_background=2.5,
                     origin=origin, shape=shape, spacing=spacing, nbpml=40)

# Define time discretization according to grid spacing
t0 = 0.
tn = 1000.  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing
nt = int(1 + (tn-t0) / dt)  # Discrete time axis length
time = np.linspace(t0, tn, nt)  # Discrete modelling time
f0 = 0.010 

nshots = 21  # Number of shots to create gradient from
nreceivers = 101  # Number of receiver locations per shot 

# Define true and intiial model
from examples.seismic import plot_velocity, plot_perturbation
from scipy import ndimage


# Plot the true and initial model and the perturbation between them
plot_velocity(model)
plot_velocity(model0)
plot_perturbation(model0, model)

# Define acquisition geometry: source
from examples.seismic import RickerSource

src = RickerSource(name='src', grid=model.grid, f0=f0, time=np.linspace(t0, tn, nt))

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(model.domain_size) * .05
src.coordinates.data[0, -1] = 500.  # Depth is 20m

# We can plot the time signature to see the wavelet
src.show()

# Define acquisition geometry: receivers
from examples.seismic import Receiver

# Initialize receivers for synthetic and observed data
rec = Receiver(name='rec', npoint=nreceivers, ntime=nt, grid=model.grid)
rec.coordinates.data[:, 0] = 980.
rec.coordinates.data[:, 1] = np.linspace(0, model.domain_size[0], num=nreceivers)

# For plotting only
src_plot = Receiver(name='src', npoint=21, ntime=nt, grid=model.grid)

# First, position source centrally in all dimensions, then set depth
src_plot.coordinates.data[:, 0] = 20.
src_plot.coordinates.data[:, 1] = np.linspace(0, model.domain_size[0], num=21)


import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig1 = plt.figure(figsize=(10, 10))
l = plt.imshow(model.vp, vmin=2.5, vmax=3, cmap=cm.jet, aspect=1,
               extent=[model.origin[0], model.origin[0] + 1e-3*model.shape[0] * model.spacing[0],
                       model.origin[1] + 1e-3*model.shape[1] * model.spacing[1], model.origin[1]])
plt.xlabel('X position (km)',  fontsize=20)
plt.ylabel('Depth (km)',  fontsize=20)
plt.tick_params(labelsize=20)
plt.scatter(1e-3*rec.coordinates.data[:, 0], 1e-3*rec.coordinates.data[:, 1],
                    s=25, c='green', marker='D')
plt.scatter(1e-3*src_plot.coordinates.data[:, 0], 1e-3*src_plot.coordinates.data[:, 1],
                    s=25, c='red', marker='o')
# Ensure axis limits
plt.xlim(model.origin[0], model.origin[0] + 1e-3*model.domain_size[0])
plt.ylim(model.origin[1] + 1e-3*model.domain_size[1], model.origin[1])
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(l, cax=cax)
cbar.set_label('Velocity (km/s)')
plt.savefig("../Figures/camembert_true.pdf")
plt.show()

fig1 = plt.figure(figsize=(10, 10))
l = plt.imshow(model0.vp, vmin=2.5, vmax=3, cmap=cm.jet, aspect=1,
               extent=[model.origin[0], model.origin[0] + 1e-3*(model.shape[0]-1) * model.spacing[0],
                       model.origin[1] + 1e-3*(model.shape[1]-1) * model.spacing[1], model.origin[1]])
plt.xlabel('X position (km)',  fontsize=20)
plt.ylabel('Depth (km)',  fontsize=20)
plt.tick_params(labelsize=20)
plt.scatter(1e-3*rec.coordinates.data[:, 0], 1e-3*rec.coordinates.data[:, 1],
                    s=25, c='green', marker='D')
plt.scatter(1e-3*src_plot.coordinates.data[:, 0], 1e-3*src_plot.coordinates.data[:, 1],
                    s=25, c='red', marker='o')
# Ensure axis limits
plt.xlim(model.origin[0], model.origin[0] + 1e-3*model.domain_size[0])
plt.ylim(model.origin[1] + 1e-3*model.domain_size[1], model.origin[1])
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(l, cax=cax)
cbar.set_label('Velocity (km/s)')
plt.savefig("../Figures/camembert_init.pdf")
plt.show()

# Compute synthetic data with forward operator 
from examples.seismic.acoustic import AcousticWaveSolver

solver = AcousticWaveSolver(model, src, rec, space_order=4)
true_d , _, _ = solver.forward(src=src, m=model.m)

# Compute initial data with forward operator 
smooth_d, _, _ = solver.forward(src=src, m=model0.m)

# Plot shot record for true and smooth velocity model and the difference
from examples.seismic import plot_shotrecord

plot_shotrecord(true_d.data, model, t0, tn, colorbar=False)
plot_shotrecord(smooth_d.data, model, t0, tn, colorbar=False)
plot_shotrecord(smooth_d.data - true_d.data, model, t0, tn, colorbar=False)

# Define gradient operator
from devito import Backward, Operator, TimeFunction, Eq
from examples.seismic import PointSource

from sympy import solve

def GradientOperator(model, grad, rec, save=False):
    # Define the wavefield with the size of the model and the time dimension
    # In practice, v does not need to be saved, we however keep it for plotting
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4,
                     save=save, time_dim=nt)

    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=4,
                     save=True, time_dim=nt)
    
    # Define the wave equation, but with a negated damping term
    eqn = model.m * v.dt2 - v.laplace - model.damp * v.dt

    # Use SymPy to rearranged the equation into a stencil expression
    stencil = Eq(v.backward, solve(eqn, v.backward)[0])
    
    # Define residual injection at the location of the forward receivers
    dt = model.critical_dt
    residual = PointSource(name='residual', ntime=nt, coordinates=rec.coordinates.data,
                          grid=model.grid)    
    res_term = residual.inject(field=v.backward, expr=residual * dt**2 / model.m, offset=model.nbpml)

    # Correlate u and v for the current time step and add it to the gradient
    grad_update = Eq(grad, grad - u * v.dt2)

    return Operator([stencil] + res_term + [grad_update],
                    time_axis=Backward)

from devito import Function

grad = Function(name='grad', grid=model.grid)
op_grad = GradientOperator(model, grad, rec, save=True)
# Generate synthetic data from true model
true_d, _, _ = solver.forward(src=src, m=model.m)

# Compute smooth data and full forward wavefield u0
smooth_d, u0, _ = solver.forward(src=src, m=model0.m, save=True)

# Compute gradient from the data residual  
v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4,
                 save=True, time_dim=nt)
op_grad(u=u0, v=v, m=model0.m, residual=smooth_d.data - true_d.data, dt=model.critical_dt)



# Adjoint wavefield movie

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from IPython.display import HTML

fig = plt.figure()
im = plt.imshow(np.transpose(v.data[0,40:-40,40:-40]), animated=True, vmin=-1e1, vmax=1e1, cmap=cm.RdGy, aspect=1,
                 extent=[model.origin[0], model.origin[0] + 1e-3 * model.shape[0] * model.spacing[0],
                         model.origin[1] + 1e-3*model.shape[1] * model.spacing[1], model.origin[1]])
plt.xlabel('X position (km)',  fontsize=20)
plt.ylabel('Depth (km)',  fontsize=20)
plt.tick_params(labelsize=20)
im2 = plt.imshow(np.transpose(model0.vp), vmin=1.5, vmax=4.5, cmap=cm.jet, aspect=1,
                 extent=[model.origin[0], model.origin[0] + 1e-3 * model.shape[0] * model.spacing[0],
                         model.origin[1] + 1e-3*model.shape[1] * model.spacing[1], model.origin[1]], alpha=.4)
def updatefig(i):
    im.set_array(np.transpose(v.data[-i,40:-40,40:-40]))
    im2.set_array(np.transpose(model0.vp))
    return im, im2

ani = animation.FuncAnimation(fig, updatefig, frames=np.linspace(1, nt, nt, dtype=np.int64), blit=True, interval=100)
# plt.close(ani._fig)
# HTML(ani.to_html5_video())

from examples.seismic import plot_image
# Plot the FWI gradient
import matplotlib.pyplot as plt
from matplotlib import cm

fig1 = plt.figure(figsize=(10,10))
l = plt.imshow(np.transpose(grad.data[40:-40, 40:-40]), vmin=-1e3, vmax=1e3, cmap=cm.jet, aspect=1,
               extent=[model.origin[0], model.origin[0] + 1e-3*model.shape[0] * model.spacing[0],
                       model.origin[1] + 1e-3*model.shape[1] * model.spacing[1], model.origin[1]])
plt.xlabel('X position (km)',  fontsize=20)
plt.ylabel('Depth (km)',  fontsize=20)
plt.tick_params(labelsize=20)
plt.savefig("../Figures/banana.pdf")

# Prepare the varying source locations sources
source_locations = np.empty((nshots, 2), dtype=np.float32)
source_locations[:, 0] = 20.
source_locations[:, 1] = np.linspace(0., 1000, num=nshots)

plot_velocity(model, source=source_locations)

# Run gradient loop over shots

# Create gradient symbol and instantiate the previously defined gradient operator
grad = Function(name='grad', grid=model.grid, dtype=model.m.dtype)
op_grad2 = GradientOperator(model, grad, rec)
u0 = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=4,
              save=True, time_dim=nt)
residual = Receiver(name='residual', ntime=nt, coordinates=rec.coordinates.data, grid=model.grid)
for i in range(nshots):
    print('Source %d out of %d' % (i+1, nshots))
    
    # Update source location
    src.coordinates.data[0, :] = source_locations[i, :]

    # Generate synthetic data from true model
    true_d, _, _ = solver.forward(src=src, m=model.m)
    
    # Compute smooth data and full forward wavefield u0
    smooth_d, _, _ = solver.forward(src=src, m=model0.m, u=u0, save=True)
    
    # Compute gradient from the data residual  
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)
    residual.data[:] = smooth_d.data[:] - true_d.data[:]
    op_grad2(u=u0, v=v, m=model0.m, residual=residual, grad=grad, dt=model.critical_dt)

# Plot FWI gradient and model update
from examples.seismic import plot_image

# Plot the FWI gradient
# Plot the FWI gradient
import matplotlib.pyplot as plt
from matplotlib import cm

fig1 = plt.figure(figsize=(10,10))
l = plt.imshow(np.transpose(grad.data[40:-40, 40:-40]), vmin=-1e4, vmax=1e4, cmap=cm.jet, aspect=1,
               extent=[model.origin[0], model.origin[0] + 1e-3*(model.shape[0]-1) * model.spacing[0],
                       model.origin[1] + 1e-3*(model.shape[1]-1) * model.spacing[1], model.origin[1]])
plt.xlabel('X position (km)',  fontsize=20)
plt.ylabel('Depth (km)',  fontsize=20)
plt.tick_params(labelsize=20)
# Ensure axis limits
plt.xlim(model.origin[0], model.origin[0] + 1e-3*model.domain_size[0])
plt.ylim(model.origin[1] + 1e-3*model.domain_size[1], model.origin[1])
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(l, cax=cax)
plt.savefig("../Figures/simplegrad.pdf")
plt.show()

# Plot the difference between the true and initial model.
# This is not known in practice as only the initial model is provided.
plot_image(model0.m.data[40:-40, 40:-40] - model.m.data[40:-40, 40:-40], vmin=-1e-1, vmax=1e-1, cmap="jet")

# Show what the update does to the model
alpha = .05 / np.max(grad.data)
plot_image(model0.m.data[40:-40, 40:-40] - alpha*grad.data[40:-40, 40:-40], vmin=.1, vmax=.2, cmap="jet")

# Create FWI gradient kernel 
from devito import Function, clear_cache

def fwi_gradient(m_in):
    # Important: We force previous wavefields to be destroyed,
    # so that we may reuse the memory.
    clear_cache()
    
    # Create symbols to hold the gradient and residual
    grad = Function(name="grad", grid=model.grid)
    residual = Receiver(name='rec', ntime=nt, coordinates=rec.coordinates.data, grid=model.grid)
    u0 = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=4,
                      save=True, time_dim=nt)
    objective = 0.
    
    for i in range(nshots):
        # Update source location
        src.coordinates.data[0, :] = source_locations[i, :]
        
        # Generate synthetic data from true model
        # In practice, this line would be replace by some I/O, shot = load(shot)
        true_d, _, _ = solver.forward(src=src, m=model.m)
        
        # Compute smooth data and full forward wavefield u0
        smooth_d, _, _ = solver.forward(src=src, m=m_in, u=u0, save=True)
        
        # Compute gradient from data residual and update objective function 
        residual.data[:] = smooth_d.data[:] - true_d.data[:]
        objective += .5*np.linalg.norm(residual.data.reshape(-1))**2
        solver.gradient(rec=residual, u=u0, m=m_in, grad=grad)
    
    grad.data[0:25,:] = 0.
    grad.data[-25:,:] = 0.
    return objective, grad.data

# Compute gradient of initial model
ff, update = fwi_gradient(model0.m)
print('Objective value is %f ' % ff)

# Plot FWI gradient and model update
from examples.seismic import plot_image

# Plot the FWI gradient
plot_image(update[40:-40, 40:-40], vmin=-1e4, vmax=1e4, cmap="jet")

# Plot the difference between the true and initial model.
# This is not known in practice as only the initial model is provided.
plot_image(model0.m.data[40:-40, 40:-40] - model.m.data[40:-40, 40:-40], vmin=-1e-1, vmax=1e-1, cmap="jet")

# Show what the update does to the model
alpha = .05 / np.max(update)
plot_image(model0.m.data[40:-40, 40:-40] - alpha*update[40:-40, 40:-40], vmin=.1, vmax=.2, cmap="jet")

# Create bounds constraint
def bound_constr(m):
    m[m<.1] = .08 # Maximum accepted velocity is 3.5 km/sec (true is 3 km/sec)
    m[m>.2] = .25 # Minimum accepted velocity is 2 km/sec (true is 2.5 km/sec)
    return m

# Run FWI with gradient descent
fwi_iterations = 8
history = np.zeros((fwi_iterations, 1))
for i in range(0, fwi_iterations):
    # Compute the functional value and gradient for the current
    # model estimate
    phi, direction = fwi_gradient(model0.m)
    
    # Store the history of the functional values
    history[i] = phi
    
    # Artificial Step length for gradient descent
    # In practice this would be replaced by a Linesearch (Wolfe, ...)
    # that would guaranty functional decrease Phi(m-alpha g) <= epsilon Phi(m)
    # where epsilon is a minimum decrease constant
    alpha = .005 / np.max(direction)
    
    # Update the model estimate and inforce minimum/maximum values
    model0.m.data[:] = bound_constr(model0.m.data - alpha * direction)
    
    # Log the progress made
    print('Objective value is %f at iteration %d' % (phi, i+1))

# Plot inverted velocity model

# First, update velocity from computed square slowness
nbpml = model.nbpml
model0.vp = np.sqrt(1. / model0.m.data[nbpml:-nbpml, nbpml:-nbpml])

plot_velocity(model0)

# Plot objective function decrease
import matplotlib.pyplot as plt

plt.figure()
plt.loglog(history)
plt.xlabel('Iteration number')
plt.ylabel('Misift value Phi')
plt.title('Convergence')
plt.show()

