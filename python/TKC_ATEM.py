import numpy as np
from scipy.constants import mu_0
import matplotlib.pyplot as plt
import ipywidgets

from SimPEG import EM, Mesh, Utils, Maps

get_ipython().magic('matplotlib inline')

# import a solver. If you want to re-run the forward simulation or inversion, 
# make sure you have pymatsolver (https://github.com/rowanc1/pymatsolver) 
# installed. The default solverLU will be painfully slow. 

try:
    from pymatsolver import Pardiso as Solver
except Exception:
    from SimPEG import SolverLU as Solver
    print(
        'Using the default solver. Install pymatsolver '
        'if you want to re-run the forward simulation or inversion'
    )

download_dir = './TKC_ATEM/'  # name of the local directory to create and put the files in. 

root_url = 'https://storage.googleapis.com/simpeg/tkc_synthetic/atem/' 
files = ['VTKout.dat']
urls = [root_url + f for f in files]

downloads = Utils.download(url=urls, folder=download_dir, overwrite=True)
downloads = dict(zip(['sigma_model'], downloads))  # create a dict

cs, npad = 35., 13
ncx, ncy = 19, 15
nczd, nczu, nczm = 9, 9, 6
hx = [(cs,npad, -1.3),(cs,ncx),(cs,npad, 1.3)]
hy = [(cs,npad, -1.3),(cs,ncy),(cs,npad, 1.3)]
hz = [(cs,npad, -1.3),(cs,nczd), (cs/2,nczm) ,(cs,nczu),(cs,npad, 1.3)]
mesh = Mesh.TensorMesh([hx, hy, hz], x0="CCC")
xc = 300+5.57e5
yc = 600+7.133e6
zc = 425.

print(mesh)

sigma = mesh.readModelUBC(downloads["sigma_model"])

# functions for visualizing the model 
def vizsection(sigma, indy=20, ax=None):
    """
    Plot a cross section of the conductivity model
    """
    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize = (5,2.5))
    mesh.plotSlice(
        np.log10(sigma), ind=indy, grid=True, normal="Y", clim=(-4, -1), ax=ax
    )
    ax.axis("equal")
    ax.set_title(("Northing at %.1fm")%(mesh.vectorCCy[indy]))
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")    
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 0)
    
def vizplan(sigma, indz=21, ax=None):
    """
    Plot a plan view of the conductivity model
    """
    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize = (5,5))
    mesh.plotSlice(np.log10(sigma), grid=True, ind=indz, clim=(-4, -1), ax=ax)
    ax.set_title(("Elevation at %.1fm")%(mesh.vectorCCz[indz]))
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)

# for plotting, we will turn the air indices to nan's 
sigma_masked = sigma.copy()

airind = (sigma==1e-8)
sigma_masked[airind] = np.nan

vizplan(sigma_masked)
vizsection(sigma_masked)

def gettopoCC(mesh, airind):
    """
    Get topography from active indices of mesh.
    """
    mesh2D = Mesh.TensorMesh([mesh.hx, mesh.hy], mesh.x0[:2])
    zc = mesh.gridCC[:,2]
    AIRIND = airind.reshape((mesh.vnC[0]*mesh.vnC[1],mesh.vnC[2]), order='F')
    ZC = zc.reshape((mesh.vnC[0]*mesh.vnC[1], mesh.vnC[2]), order='F')
    topo = np.zeros(ZC.shape[0])
    topoCC = np.zeros(ZC.shape[0])
    for i in range(ZC.shape[0]):
        ind  = np.argmax(ZC[i,:][~AIRIND[i,:]])
        topo[i] = ZC[i,:][~AIRIND[i,:]].max() + mesh.hz[~AIRIND[i,:]][ind]*0.5
        topoCC[i] = ZC[i,:][~AIRIND[i,:]].max()
    XY = Utils.ndgrid(mesh.vectorCCx, mesh.vectorCCy)
    return mesh2D, topoCC

# easting
x = mesh.vectorCCx[np.logical_and(mesh.vectorCCx>-250., mesh.vectorCCx<250.)]

# northing - use slightly wider spacing
nskip = 2
y = mesh.vectorCCy[np.logical_and(mesh.vectorCCy>-250., mesh.vectorCCy<250.)][::nskip]

# height
z = np.r_[30.]

# grid of locations
xyz = Utils.ndgrid(x, y, z)

fig, ax = plt.subplots(1, 1, figsize=(5,5))
vizplan(sigma_masked, ax=ax)
ax.plot(xyz[:,0], xyz[:,1], 'wo', ms=3)
ax.set_xlim(-400, 400)
ax.set_ylim(-400, 400)

ntx = xyz.shape[0]
srcList = []
times = np.logspace(-4, np.log10(2e-3), 10)

for itx in range(ntx):
    rx = EM.TDEM.Rx.Point_b(xyz[itx,:], times, orientation='z')
    src = EM.TDEM.Src.CircularLoop(
        [rx], waveform=EM.TDEM.Src.StepOffWaveform(), 
        loc=xyz[itx,:], radius=13.
    )       
    srcList.append(src)
    
print(
    'In this survey, there are {nsrc} soundings'.format(
        nsrc=len(srcList)
    )
)
print(
    'There are {ntimes} time channels, '
    'and we are sampling the {comp}-component '
    'of the {field}-field'.format(
        ntimes = len(times),
        comp=rx.projComp,
        field=rx.projField
    )
)

# setup and run the forward simulation 
timeSteps_fwd = [(1e-5, 5), (1e-4, 10),(5e-4, 10)]

survey = EM.TDEM.Survey(srcList)    
problem = EM.TDEM.Problem3D_b(
    mesh, timeSteps=timeSteps_fwd, verbose=False, 
    sigmaMap=Maps.IdentityMap(mesh)
)
problem.pair(survey)
problem.Solver = Solver

# %timeit
# dpred = survey.dpred(sigma)
# TKCATEMexample = {
#     "mesh": mesh, "sigma": sigma, "xyz": xyz, "ntx": tx,
#     "times": times, "timeSteps": problem.timeSteps, "dpred": dpred
# }
# pickle.dump(
#     TKCATEMexample, open(download_dir + "/TKCATEMfwd.p", "wb" )
# )

