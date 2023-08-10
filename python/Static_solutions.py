import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
get_ipython().magic('matplotlib inline')

Q0 = 0.2
aR = 1.0
f = 0.2
H0 = 1.0

nx = 100
nt = 20

xs = np.linspace(-5.0, 5.0, nx, endpoint=True)
ts = np.linspace(0.0, 10, nt+1, endpoint=True)
qs = np.zeros((nt+1, nx, 3))

for i in range(nt+1):
    qs[i,:,0] = (Q0 * ts[i] / (2.0 * aR)) * np.exp(-np.abs(xs) / aR)
    qs[i,:,1] = np.sign(xs) * (Q0 / (2.0 * H0)) * np.exp(-np.abs(xs) / aR)
    qs[i,:,2] = - np.sign(xs) * (Q0 * ts[i] * f / (2.0 * H0)) * np.exp(-np.abs(xs) / aR)

fig = plt.figure(figsize=(8,6))
ax = fig.gca()
plt.rc("font", size=15)

def animate(i):
    ax.clear()
    ax.plot(xs, qs[i,:,0], xs, qs[i,:,1], '--', xs, qs[i,:,2], '-.', lw=2)
    title = 't = ' + str(ts[i])
    ax.set_title(title)
    ax.set_xlabel(r'$x$')
    ax.legend([r'$h$', r'$v_x$', r'$v_y$'])#, loc='center left', bbox_to_anchor=(1, 0.5))
    ax
    ax.set_ylim([-0.2, 0.8])    

anim = animation.FuncAnimation(fig, animate, frames=nt+1, interval=150)

HTML(anim.to_html5_video())



