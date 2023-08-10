get_ipython().magic('pylab inline')

import seaborn as sns

sns.set_context('poster', font_scale=1.25)

sns.set_style('white')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from LB_D2Q9.dimensionless import opencl_dim as lb

N = 25 # resolution of obstacle: number of grid points over radius

D = 1. # meter
rho = 1. # kg/m^3
nu = 1. # Viscosity, m^2/s

pressure_grad = -100 # Pa/m

pipe_length = 3*D # meter

# initialize cylinder
cylinder_center = [pipe_length/4, D/2]
cylinder_radius = D/25

sim = lb.Pipe_Flow_Cylinder(diameter=D, rho=rho, viscosity=nu, pressure_grad=pressure_grad, pipe_length=pipe_length,
                             N=N, time_prefactor=1.,
                             cylinder_center=cylinder_center, cylinder_radius=cylinder_radius,
                             two_d_local_size = (32, 32), three_d_local_size = (32, 32, 1))

print 'Total num grid points:' , '%.3E' % (sim.nx*sim.ny)

plt.imshow(sim.obstacle_mask_host.T)

total_time_to_run=20

time_per_iteration = .1 #In dimensionless units
steps_per_iteration = int(time_per_iteration/sim.delta_t)
print 'Running for', steps_per_iteration, 'iterations between frames'
print 'Corresponds to a non-dimensional time of ' , time_per_iteration

total_num_steps = int(np.ceil(total_time_to_run/time_per_iteration))

from IPython import display

from mpl_toolkits.axes_grid1 import make_axes_locatable

offset = 1200

for i in range(total_num_steps):
    plt.clf()

    fields = sim.get_nondim_fields()
    
    u = fields['u']
    u[sim.obstacle_mask_host.astype(np.bool)] = np.nan
    
    cmap = cm.coolwarm    
    cmap.set_bad(color='k', alpha=1.)
    
    im = plt.imshow(fields['u'].T, cmap=cmap, interpolation=None)

    plt.clim(-3, 3)
    plt.title('Dimensionless Horizontal Velocity')
    
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='5%', pad=0.3)
    plt.colorbar(im, cax=cax, ticks=[-3,-1.5,0,1.5,3])

    plt.savefig('{0:05}'.format(offset + i) + '.png', dpi=200, bbox_inches='tight')
    
    print 'Updated!'
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    
    sim.run(steps_per_iteration)

