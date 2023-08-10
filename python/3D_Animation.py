get_ipython().magic('matplotlib inline')
from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython.display import HTML

# First set up the figure, the axis, and the plot element we want to animate
fig = figure()
ax = axes(projection='3d')

# we need to fix some parameters, describing size of the inner radus of the torus/knot
r = .4

# We set the parameterization for the circle and the knot
u = linspace(0, 2*pi, 100)
v = linspace(0, 2*pi, 100)
u,v = meshgrid(u,v)
x_torus = 2*sin(u) + r*sin(u)*cos(v)
y_torus = 2*cos(u) + r*cos(u)*cos(v)
z_torus = r*sin(v)
x_knot =   sin(u) + 2*sin(2*u) + r*sin(u)*cos(v)
y_knot =   cos(u) - 2*cos(2*u) + r*cos(u)*cos(v)
z_knot = -sin(3*u) + r*sin(v)

ax.plot_surface(x_torus, y_torus, z_torus, color='c')
ax.set_xlim([-2*(1+r), 2*(1+r)])
ax.set_ylim([-2*(1+r), 2*(1+r)])
ax.set_zlim([-(1+r), (1+r)])

# initialization function: plot the background of each frame
def init():
    thingy = ax.plot_surface([0], [0], [0], color='c')
    return (thingy,)

# animation function. This is called sequentially
def animate(i):
    a = sin(pi*i/100)**2  # this is an interpolation parameter. a = 0 is torus, a=1 is knot
    x = (1-a)*x_torus + a*x_knot
    y = (1-a)*y_torus + a*y_knot
    z = (1-a)*z_torus + a*z_knot
    ax.clear()
    ax.set_xlim([-2*(1+r), 2*(1+r)])
    ax.set_ylim([-2*(1+r), 2*(1+r)])
    ax.set_zlim([-(1+r), (1+r)])
    thingy = ax.plot_surface(x, y, z, color='c')
    return (thingy,)

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=50, blit=True)

HTML(anim.to_html5_video())

anim.save('knot.mp4')

2+2



