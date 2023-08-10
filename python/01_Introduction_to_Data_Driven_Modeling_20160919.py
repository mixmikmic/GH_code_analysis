#Code from: https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/


# import numpy as np
# from scipy import integrate

# # Note: t0 is required for the odeint function, though it's not used here.
# def lorentz_deriv((x, y, z), t0, sigma=10., beta=8./3, rho=28.0):
#     """Compute the time-derivative of a Lorenz system."""
#     return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# x0 = [1, 1, 1]  # starting vector
# t = np.linspace(0, 3, 1000)  # one thousand time steps
# x_t = integrate.odeint(lorentz_deriv, x0, t)





import numpy as np
from scipy import integrate

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
get_ipython().magic('matplotlib inline')
N_trajectories = 10


#dx/dt = sigma(y-x)
#dy/dt = x(rho-z)-y
#dz/dt = xy-beta*z

def lorentz_deriv((x, y, z), t0, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorentz system."""
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


# Choose random starting points, uniformly distributed from -15 to 15
np.random.seed(1)
x0 = -15 + 30 * np.random.random((N_trajectories, 3))

# Solve for the trajectories
t = np.linspace(0, 7, 1000)
x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t)
                  for x0i in x0])

# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('off')
plt.set_cmap(plt.cm.YlOrRd_r)
plt.set_cmap(plt.cm.hot)
# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

# prepare the axes limits
ax.set_xlim((-25, 25))
ax.set_ylim((-35, 35))
ax.set_zlim((5, 55))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=500, interval=10, blit=True)

# Save as mp4. This requires mplayer or ffmpeg to be installed
anim.save('./Images/lorentz_attractor.mp4', fps=15, extra_args=['-vcodec', 'libx264'],dpi=200)

plt.close()

from IPython.display import HTML
HTML("""
<video width="600" height="400" controls>
  <source src="files/Images/lorentz_attractor.mp4" type="video/mp4">
</video>
""")

from IPython.display import YouTubeVideo
YouTubeVideo('JZoGO0MrZPA',width=700, height=600)

from IPython.display import YouTubeVideo
YouTubeVideo('0aQxJgHknGs',width=700, height=600)

from IPython.display import YouTubeVideo
YouTubeVideo('VQ1f312SVqg',width=700, height=600)

from IPython.display import YouTubeVideo
YouTubeVideo('D6XTyLbO13w',width=700, height=600)

