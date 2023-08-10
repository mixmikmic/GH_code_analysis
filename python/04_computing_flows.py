get_ipython().run_cell_magic('writefile', 'fdm_b.py', '\nimport numpy as np\nimport pdb\nimport scipy.sparse as sp\nfrom scipy.sparse.linalg import spsolve # to use its short name\nfrom collections import namedtuple\n\nclass InputError(Exception):\n    pass\n\ndef quivdata(Out, x, y, z=None, iz=0):\n    """Returns coordinates and velocity components to show velocity with quiver\n    \n    Compute arrays to display velocity vectors using matplotlib\'s quiver.\n    The quiver is always drawn in the xy-plane for a specific layer. Default iz=0\n    \n    Parameters\n    ----------\n    `Out` : namedtuple holding arrays `Qx`, `Qy`, `Qz` as defined in `fdm3`\n        `Qx` : ndarray, shape: (Ny, Nx-1, Nz), [L3/T]\n            Interfacial flows in finite difference model in x-direction from `fdm3\'\n        `Qy` : ndarray, shape: (Ny-1, Nx, Nz), [L3/T]\n            Interfacial flows in finite difference model in y-direction from `fdm3`\n        `Qz` : ndarray, shape: (Ny, Nx, Nz-1), [L3/T]\n            Interfacial flows in finite difference model in z-direction from `fdm3`            \n    `x` : ndarray, [m]\n        Grid line coordinates of columns\n    \'y\' : ndarray, [m]\n        Grid line coordinates of rows\n    `z` : ndaray [L] | int [-]\n        If z == None, then iz must be given (default = 0)\n        If z is an ndarray vector of floats\n            z will be interpreted as the elvations of uniform layers.\n            iz will be ignored\n        If z is a full 3D ndarray of floats\n            z will be interpreted as the elevations of the tops and bottoms of all cells.\n            iz will be ignored\n    `iz` : int [-]\n            iz is ignored if z ~= None\n            iz is the number of the layer for which the data are requested,\n            and all output arrays will be 2D for that layer.\n\n    Returns\n    -------\n    `Xm` : ndarray, shape: (Nz, Ny, Nx), [L]\n        x-coordinates of cell centers\n    `Ym` : ndarray, shape: (Nz, Ny, Nx), [L]\n        y-coodinates of cell centers\n    `ZM` : ndarray, shape: (Nz, Ny, Nx), [L]\n        `z`-coordinates at cell centers\n    `U` : ndarray, shape: (Nz, Ny, Nx), [L3/d]\n        Flow in `x`-direction at cell centers\n    `V` : ndarray, shape: (Nz, Ny, Nx), [L3/T]\n        Flow in `y`-direction at cell centers\n    `W` : ndarray, shape: (Nz, Ny, Nx), [L3/T]\n        Flow in `z`-direction at cell centers.\n    \n    """\n    Ny = len(y)-1\n    Nx = len(x)-1\n    \n    xm = 0.5 * (x[:-1] + x[1:])\n    ym = 0.5 * (y[:-1] + y[1:])\n    \n    X, Y = np.meshgrid(xm, ym) # coordinates of cell centers\n    \n    # Flows at cell centers\n    U = np.concatenate((Out.Qx[iz, :, 0].reshape((1, Ny, 1)), \\\n                        0.5 * (Out.Qx[iz, :, :-1].reshape((1, Ny, Nx-2)) +\\\n                               Out.Qx[iz, :, 1: ].reshape((1, Ny, Nx-2))), \\\n                        Out.Qx[iz, :, -1].reshape((1, Ny, 1))), axis=2).reshape((Ny,Nx))\n    V = np.concatenate((Out.Qy[iz, 0, :].reshape((1, 1, Nx)), \\\n                        0.5 * (Out.Qy[iz, :-1, :].reshape((1, Ny-2, Nx)) +\\\n                               Out.Qy[iz, 1:,  :].reshape((1, Ny-2, Nx))), \\\n                        Out.Qy[iz, -1, :].reshape((1, 1, Nx))), axis=1).reshape((Ny,Nx))\n    return X, Y, U, V\n\n\ndef unique(x, tol=0.0001):\n    """return sorted unique values of x, keeping ascending or descending direction"""\n    if x[0]>x[-1]:  # vector is reversed\n        x = np.sort(x)[::-1]  # sort and reverse\n        return x[np.hstack((np.diff(x) < -tol, True))]\n    else:\n        x = np.sort(x)\n        return x[np.hstack((np.diff(x) > +tol, True))]\n\n    \ndef fdm3(x, y, z, kx, ky, kz, FQ, HI, IBOUND):\n    \'\'\'Steady state 3D Finite Difference Model returning computed heads and flows.\n        \n    Heads and flows are returned as 3D arrays as specified under output parmeters.\n    \n    Parameters\n    ----------\n    `x` : ndarray,[L]\n        `x` coordinates of grid lines perpendicular to rows, len is Nx+1\n    `y` : ndarray, [L]\n        `y` coordinates of grid lines along perpendicular to columns, len is Ny+1\n    `z` : ndarray, [L]\n        `z` coordinates of layers tops and bottoms, len = Nz+1\n    `kx`, `ky`, `kz` : ndarray, shape: (Ny, Nx, Nz), [L/T]\n        hydraulic conductivities along the three axes, 3D arrays.\n    `FQ` : ndarray, shape: (Ny, Nx, Nz), [L3/T]\n        prescrived cell flows (injection positive, zero of no inflow/outflow)\n    `IH` : ndarray, shape: (Ny, Nx, Nz), [L]\n        initial heads. `IH` has the prescribed heads for the cells with prescribed head.\n    `IBOUND` : ndarray, shape: (Ny, Nx, Nz) of int\n        boundary array like in MODFLOW with values denoting\n        * IBOUND>0  the head in the corresponding cells will be computed\n        * IBOUND=0  cells are inactive, will be given value NaN\n        * IBOUND<0  coresponding cells have prescribed head\n    \n    outputs\n    -------    \n    `Out` : namedtuple containing heads and flows:\n        `Out.Phi` : ndarray, shape: (Ny, Nx, Nz), [L3/T] \n            computed heads. Inactive cells will have NaNs\n        `Out.Q`   : ndarray, shape: (Ny, Nx, Nz), [L3/T]\n            net inflow in all cells, inactive cells have 0\n        `Out.Qx   : ndarray, shape: (Ny, Nx-1, Nz), [L3/T] \n            intercell flows in x-direction (parallel to the rows)\n        `Out.Qy`  : ndarray, shape: (Ny-1, Nx, Nz), [L3/T] \n            intercell flows in y-direction (parallel to the columns)\n        `Out.Qz`  : ndarray, shape: (Ny, Nx, Nz-1), [L3/T] \n            intercell flows in z-direction (vertially upward postitive)\n        the 3D array with the final heads with `NaN` at inactive cells.\n    \n    TO 160905\n    \'\'\'\n\n    # define the named tuple to hold all the output of the model fdm3\n    Out = namedtuple(\'Out\',[\'Phi\', \'Q\', \'Qx\', \'Qy\', \'Qz\'])\n    Out.__doc__ = """fdm3 output, <namedtuple>, containing fields Phi, Qx, Qy and Qz\\n \\\n                    Use Out.Phi, Out.Q, Out.Qx, Out.Qy and Out.Qz"""                            \n\n    x = unique(x)\n    y = unique(y)[::-1]  # unique and descending\n    z = unique(z)[::-1]  # unique and descending\n        \n    # as well as the number of cells along the three axes\n    SHP = Nz, Ny, Nx = len(z)-1, len(y)-1, len(x)-1\n    \n    Nod = np.prod(SHP)\n \n    if Nod == 0:\n        raise AssetError(\n            "Grid shape is (Ny, Nx, Nz) = {0}. Number of cells in all 3 direction must all be > 0".format(SHP))\n                                \n    if kx.shape != SHP:\n        raise AssertionError("shape of kx {0} differs from that of model {1}".format(kx.shape,SHP))\n    if ky.shape != SHP:\n        raise AssertionError("shape of ky {0} differs from that of model {1}".format(ky.shape,SHP))\n    if kz.shape != SHP:\n        raise AssertionError("shape of kz {0} differs from that of model {1}".format(kz.shape,SHP))\n        \n    # from this we have the width of columns, rows and layers\n    dx = np.abs(np.diff(x)).reshape(1, 1, Nx)\n    dy = np.abs(np.diff(y)).reshape(1, Ny, 1)\n    dz = np.abs(np.diff(z)).reshape(Nz, 1, 1)\n    \n    active = (IBOUND>0).reshape(Nod,)  # boolean vector denoting the active cells\n    inact  = (IBOUND==0).reshape(Nod,) # boolean vector denoting inacive cells\n    fxhd   = (IBOUND<0).reshape(Nod,)  # boolean vector denoting fixed-head cells\n\n    # half cell flow resistances\n    Rx = 0.5 * dx / (dy * dz) / kx\n    Ry = 0.5 * dy / (dz * dx) / ky\n    Rz = 0.5 * dz / (dx * dy) / kz\n    \n    # set flow resistance in inactive cells to infinite\n    Rx = Rx.reshape(Nod,); Rx[inact] = np.Inf; Rx=Rx.reshape(SHP)\n    Ry = Ry.reshape(Nod,); Ry[inact] = np.Inf; Ry=Ry.reshape(SHP)\n    Rz = Rz.reshape(Nod,); Rz[inact] = np.Inf; Rz=Rz.reshape(SHP)\n    \n    # conductances between adjacent cells\n    Cx = 1 / (Rx[:, :, :-1] + Rx[:, :, 1:])\n    Cy = 1 / (Ry[:, :-1, :] + Ry[:, 1:, :])\n    Cz = 1 / (Rz[:-1, :, :] + Rz[1:, :, :])\n    \n    NOD = np.arange(Nod).reshape(SHP)\n    \n    IE = NOD[:, :, 1:]  # east neighbor cell numbers\n    IW = NOD[:, :, :-1] # west neighbor cell numbers\n    IN = NOD[:, :-1, :] # north neighbor cell numbers\n    IS = NOD[:, 1:, :]  # south neighbor cell numbers\n    IT = NOD[:-1, :, :] # top neighbor cell numbers\n    IB = NOD[1:, :, :]  # bottom neighbor cell numbers\n    \n    R = lambda x : x.ravel()  # generate anonymous function R(x) as shorthand for x.ravel()\n\n    # notice the call  csc_matrix( (data, (rowind, coind) ), (M,N))  tuple within tupple\n    # also notice that Cij = negative but that Cii will be postive, namely -sum(Cij)\n    A = sp.csc_matrix(( -np.concatenate(( R(Cx), R(Cx), R(Cy), R(Cy), R(Cz), R(Cz)) ),\\\n                        (np.concatenate(( R(IE), R(IW), R(IN), R(IS), R(IB), R(IT)) ),\\\n                         np.concatenate(( R(IW), R(IE), R(IS), R(IN), R(IT), R(IB)) ),\\\n                      )),(Nod,Nod))\n    \n    # to use the vector of diagonal values in a call of sp.diags() we need to have it aa a \n    # standard nondimensional numpy vector.\n    # To get this:\n    # - first turn the matrix obtained by A.sum(axis=1) into a np.array by np.array( .. )\n    # - then take the whole column to loose the array orientation (to get a dimensionless numpy vector)\n    adiag = np.array(-A.sum(axis=1))[:,0]\n    \n    Adiag = sp.diags(adiag)  # diagonal matrix with a[i,i]\n    \n    RHS = FQ.reshape(Nod,1) - A[:,fxhd].dot(HI.reshape(Nod,1)[fxhd]) # Right-hand side vector\n    \n    Out.Phi = HI.flatten() # allocate space to store heads\n    \n    Out.Phi[active] = spsolve( (A+Adiag)[active][:,active] ,RHS[active] ) # solve heads at active locations\n    \n    # net cell inflow\n    Out.Q  = (A+Adiag).dot(Out.Phi).reshape(SHP)\n\n    # set inactive cells to NaN\n    Out.Phi[inact] = np.NaN # put NaN at inactive locations\n    \n    # reshape Phi to shape of grid\n    Out.Phi = Out.Phi.reshape(SHP)\n    \n    #Flows across cell faces\n    Out.Qx =  -np.diff( Out.Phi, axis=2) * Cx\n    Out.Qy =  +np.diff( Out.Phi, axis=1) * Cy\n    Out.Qz =  +np.diff( Out.Phi, axis=0) * Cz\n    \n    return Out # all outputs in a named tuple for easy access')

import fdm_b
from importlib import reload
reload(fdm_b)

import numpy as np
reload(fdm_b) # make sure we reload because we edit the file regularly

# specify a rectangular grid
x = np.arange(-1000.,  1000.,  25.)
y = np.arange( 1000., -1000., -25.)
z = np.array([20, 0 ,-10, -100.])

Nz, Ny, Nx = SHP = len(z)-1, len(y)-1, len(x)-1

k = 10.0 # m/d uniform conductivity
kx = k * np.ones(SHP)
ky = k * np.ones(SHP)
kz = k * np.ones(SHP)

IBOUND = np.ones(SHP)
IBOUND[:, -1, :] = -1  # last row of model heads are prescribed
IBOUND[:, 40:45, 20:70]=0 # inactive

FQ = np.zeros(SHP)    # all flows zero. Note SHP is the shape of the model grid
FQ[1, 30, 25] = -1200  # extraction in this cell

HI = np.zeros(SHP)

Out = fdm_b.fdm3( x, y, z, kx, ky, kz, FQ, HI, IBOUND)

# when not using some of the parameters
#Phi, _, _, _ = fdm.fdm3( x, y, z, kx, ky, kz, FQ, HI, IBOUND) # ignore them using the _
#Phi, _Qx, _Qy, _Qz = fdm.fdm3( x, y, z, kx, ky, kz, FQ, HI, IBOUND)  # make them private

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt # combines namespace of numpy and pyplot

print('Out.Phi.shape = {0}'.format(Out.Phi.shape))
print('Out.Q.shape = {0}'.format(Out.Q.shape))
print('Out.Qx.shape = {0}'.format(Out.Qx.shape))
print('Out.Qy.shape = {0}'.format(Out.Qy.shape))
print('Out.Qz.shape = {0}'.format(Out.Qz.shape))

xm = 0.5 * (x[:-1] + x[1:])
ym = 0.5 * (y[:-1] + y[1:])

layer = 2 # contours for this layer
nc = 50   # number of contours in total

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title("Contours (%d in total) of the head in layer %d with inactive section" % (nc, layer))
plt.contour(xm, ym, Out.Phi[layer], nc)

#plt.quiver(X, Y, U, V) # show velocity vectors
X, Y, U, V = fdm_b.quivdata(Out, x, y, iz=0)
plt.quiver(X, Y, U, V)

print('\nSum of the net inflow over all cells is sum(Q) = {0:g} [m3/d]\n'.format(np.sum(Out.Q.ravel())))

print('\nThe indivdual values for the top layer are shown here:')
print(np.round(Out.Q[:,:,0].T,2))

plt.figure(); plt.title('Q of cells in top layer [m3/d]')
plt.imshow(Out.Q[0, :, :], interpolation='None')
plt.colorbar()

plt.figure(); plt.title('Q of cells in second layer [m3/d]')
plt.imshow(Out.Q[1, :, :], interpolation='None')
plt.colorbar()

plt.figure(); plt.title('Q of cells in bottom layer with well [m3/d]')
plt.imshow(Out.Q[1, :, :], interpolation='None')
plt.colorbar()



