get_ipython().magic('matplotlib inline')

import pygimli as pg

import numpy as np
import matplotlib.pyplot as plt

grid = pg.createGrid(x=np.linspace(-1.0, 1.0, 10),
                     y=np.linspace(-1.0, 1.0, 10))

u = pg.solver.solve(grid, f=1.,
                    uB=[grid.findBoundaryByMarker(1, 5), 0.0],
                    verbose=True)

ax, cbar = pg.show(grid, data=u, colorBar=True, label='P1 Solution $u$')

pg.mplviewer.drawMesh(ax, grid)

gridh2 = grid.createH2()

uh = pg.solver.solve(gridh2, f=1.,
                     uB=[gridh2.findBoundaryByMarker(1, 5), 0.0],
                     verbose=True)

ax, cbar = pg.show(gridh2, data=uh, colorBar=True, label='H2 Solution $u$')

pg.mplviewer.drawMesh(ax, gridh2)

gridp2 = grid.createP2()

up = pg.solver.solve(gridp2, f=1.,
                     uB=[gridp2.findBoundaryByMarker(1, 5), 0.0],
                     verbose=True)

def uAna(r):
    x = r[0]
    y = r[1]

    ret = 0
    for k in range(1, 151, 2):
        kp = k*np.pi
        s = np.sin(kp * (1. + x)/2) / (k**3 * np.sinh(kp)) *             (np.sinh(kp * (1. + y)/2) + np.sinh(kp * (1. - y)/2))
        ret += s
    return (1. - x**2)/2 - 16./(np.pi**3) * ret

x = np.linspace(-1.0, 1.0, 100)

probe = np.zeros((len(x), 3))
probe[:, 0] = x

uH1 = pg.interpolate(srcMesh=grid, inVec=u, destPos=probe)
uH2 = pg.interpolate(srcMesh=gridh2, inVec=uh, destPos=probe)
uP2 = pg.interpolate(srcMesh=gridp2, inVec=up, destPos=probe)

plt.figure()
plt.plot(x, np.array(list(map(uAna, probe))), 'black', linewidth=2,
         label='analytical')
plt.plot(x, uH1, label='linear (H1)')
plt.plot(x, uH2, label='linear (H2)')
plt.plot(x, uP2, label='quadratic (P2)')

plt.xlim([-0.4, 0.4])
plt.ylim([0.25, 0.3])
plt.legend()


plt.show()

