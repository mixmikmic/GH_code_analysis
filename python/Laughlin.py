import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

N = 200
m = 7.0
# set the size of the initial random distribution
min_lengthscale = 80.0
# set the window size
max_lengthscale = 300.0

def initialise():
    positions = [min_lengthscale*(r.random()*np.exp(2.0j*np.pi*r.random())) for i in range(N)]
    return positions

def move_particle(positions, index):
    z_new = positions[index] + min_lengthscale*(r.random()*np.exp(1.0j*2*np.pi*r.random()))
    p = 1.0
    for i in range(N):
        if i != index:
            p *= abs((z_new - positions[i])/(positions[index] - positions[i]))**(2*m)
    p *= np.exp(0.5*(abs(positions[index])**2 - abs(z_new)**2))
    if r.random() < p:
        positions[index] = z_new

def separations(positions):
    dr = 1.0
    separations = []
    for ref in positions:
        for pos in positions:
            if ref != pos:# and np.absolute(pos) < 0.2*min_lengthscale:
                separations.append(np.absolute(pos-ref))
    weighted_frequencies = np.bincount(separations)
    for i in range(weighted_frequencies.shape[0]):
        if i != 0:
            weighted_frequencies[i] = weighted_frequencies[i]/(i*dr)
    return weighted_frequencies        
        
positions = initialise()

fig = plt.figure()
plt.suptitle("Laughlin Wavefunction for the m = " + str(m) + " State")

ax1 = fig.add_subplot(121, adjustable = 'box', aspect = 1.0)
ax2 = fig.add_subplot(122, adjustable = 'box', aspect = 1.0)
ax1.set_xlim(-0.2*max_lengthscale, 0.2*max_lengthscale)
ax1.set_ylim(-0.2*max_lengthscale, 0.2*max_lengthscale)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax2.set_xlim(0, 0.2*min_lengthscale)
ax2.set_ylim(0,40)
ax2.set_xlabel("Separation")
ax2.set_ylabel("Frequency/Separation")# \n Unnormalised Radial Correlation Function")

points = ax1.scatter([], [], s = 2)
line, = ax2.plot([], [])
p = [points, line]

def animate(i):
    p[0].set_offsets([[z.real for z in positions], [z.imag for z in positions]])
    
    index = 0
    for j in range(N):
        move_particle(positions, index)
        index += 1
    
    # start plotting once the particles have
    # settled into the Laughlin wavefunction
    if i > 20:
        s = separations(positions) 
        xd, yd = p[1].get_data()
        xd2, yd2 = xd.tolist(), yd.tolist()
        for i in range(len(s)):
            xd2.append(i)
            yd2.append(s[i])
        xd2, yd2 = np.array(xd2), np.array(yd2)
        p[1].set_data(xd2, yd2)
        ax2.figure.canvas.draw()
        
    ax1.figure.canvas.draw()
    
    return p

anim = animation.FuncAnimation(fig, animate, 500, repeat = False, blit = True, interval = 100)
html_video = anim.to_html5_video()
HTML(html_video)

