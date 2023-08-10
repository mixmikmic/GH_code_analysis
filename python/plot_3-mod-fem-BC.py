get_ipython().magic('matplotlib inline')

import numpy as np
import pygimli as pg

from pygimli.solver import solve
from pygimli.viewer import show
from pygimli.mplviewer import drawStreams

grid = pg.createGrid(x=np.linspace(-1.0, 1.0, 21),
                     y=np.linspace(-1.0, 1.0, 21))

def uDirichlet(b):
    """
        Return a solution value for coordinate p.
    """
    return 4.0

dirichletBC = [[1, 1.0],                                    # left
               [grid.findBoundaryByMarker(2), 2.0],         # right
               [grid.findBoundaryByMarker(3),
                lambda p: 3.0 + p.center()[0]],  # top
               [grid.findBoundaryByMarker(4), uDirichlet]]  # bottom

u = solve(grid, f=1., uB=dirichletBC)

ax = show(grid, data=u, colorBar=True,
          orientation='vertical', label='Solution $u$',
          levels=np.linspace(1.0, 4.0, 17), hold=1)[0]

show(grid, axes=ax)

ax.text(0, 1.01, '$u=3+x$', ha='center')
ax.text(-1.01, 0, '$u=1$', va='center', ha='right', rotation='vertical')
ax.text(0, -1.01, '$u=4$', ha='center', va='top')
ax.text(1.02, 0, '$u=2$', va='center', ha='left',  rotation='vertical')

ax.set_title('$\\nabla\cdot(1\\nabla u)=1$')

ax.set_xlim([-1.1, 1.1])  # some boundary for the text
ax.set_ylim([-1.1, 1.1])

neumannBC = [[1, -0.5],  # left
             [grid.findBoundaryByMarker(4), 2.5]]  # bottom

dirichletBC = [3, 1.0]  # top

u = solve(grid, f=0., duB=neumannBC, uB=dirichletBC)

ax = show(grid, data=u, filled=True, colorBar=True,
          orientation='vertical', label='Solution $u$',
          levels=np.linspace(min(u), max(u), 14), hold=1)[0]

drawStreams(ax, grid, u)

ax.text(0.0, 1.01, '$u=1$',
        horizontalalignment='center')  # top -- 3
ax.text(-1.0, 0.0, '$\partial u/\partial n=-0.5$',
        va='center', ha='right', rotation='vertical')  # left -- 1
ax.text(0.0, -1.01, '$\partial u/\partial n=2.5$',
        ha='center', va='top')  # bot -- 4
ax.text(1.01, 0.0, '$\partial u/\partial n=0$',
        va='center', ha='left', rotation='vertical')  # right -- 2

ax.set_title('$\\nabla\cdot(1\\nabla u)=0$')

ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])

pg.wait()

