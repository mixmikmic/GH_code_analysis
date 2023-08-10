import cPickle as pickle
from SimPEG import EM, Mesh, Utils, Maps
from SimPEG.Survey import Data
get_ipython().magic('pylab inline')
import numpy as np
from pymatsolver import PardisoSolver
from matplotlib.colors import LogNorm
from ipywidgets import interact, IntSlider

url = "https://storage.googleapis.com/simpeg/kevitsa_synthetic/"
files = ['dcipdata_12150N.txt', 'dc_mesh.txt', 'dc_sigma.txt', 'dc_topo.txt']
keys = ['data', 'mesh', 'sigma', 'topo']
downloads = Utils.download([url + f for f in files], folder='./KevitsaDC', overwrite=True)
downloads = dict(zip(keys, downloads))

mesh = Mesh.TensorMesh.readUBC(downloads["mesh"])
sigma = mesh.readModelUBC(downloads["sigma"])
topo = np.loadtxt(downloads["topo"])
dcipdata = np.loadtxt(downloads["data"])
actind = ~np.isnan(sigma)
mesh.plotGrid()

figsize(8, 4)
indy = 6
temp = 1./sigma.copy()
temp[~actind] = np.nan
out = mesh.plotSlice(temp, normal="Y", ind=indy, pcolorOpts={"norm": LogNorm(), "cmap":"jet_r"}, clim=(1e0, 1e3))
plt.ylim(-800, 250)
plt.xlim(5000, 11000)
plt.gca().set_aspect(2.)
plt.title(("y= %d m")%(mesh.vectorCCy[indy]))
cb = plt.colorbar(out[0], orientation="horizontal")
cb.set_label("Resistivity (Ohm-m)")

def getGeometricFactor(locA, locB, locsM, locsN, eps = 0.01):    
    """
    Geometric factor for a pole-dipole survey
    """
    MA = np.abs(locA[0] - locsM[:, 0]) 
    MB = np.abs(locB[0] - locsM[:, 0]) 
    NA = np.abs(locA[0] - locsN[:, 0]) 
    NB = np.abs(locB[0] - locsN[:, 0])     
    geometric = 1./(2*np.pi) * (1/MA - 1/NA)
    return geometric




A = dcipdata[:,:2]
B = dcipdata[:,2:4]
M = dcipdata[:,4:6]
N = dcipdata[:,6:8]

Elec_locs = np.vstack((A, B, M, N))
uniqElec = Utils.uniqueRows(Elec_locs)
nElec = len(uniqElec[1])
pts = np.c_[uniqElec[0][:,0],  uniqElec[0][:,1]]
elec_topo = EM.Static.Utils.drapeTopotoLoc(mesh, pts[:,:2], actind=actind)
Elec_locsz = np.ones(Elec_locs.shape[0]) * np.nan

for iElec in range (nElec):
    inds = np.argwhere(uniqElec[2] == iElec)
    Elec_locsz[inds] = elec_topo[iElec,2] 
    
Elec_locs = np.c_[Elec_locs, Elec_locsz]
nloc = int(Elec_locs.shape[0]/4)
A = Elec_locs[:nloc]
B = Elec_locs[nloc:2*nloc]
M = Elec_locs[2*nloc:3*nloc]
N = Elec_locs[3*nloc:4*nloc]

uniq = Utils.uniqueRows(np.c_[A, B])
nSrc = len(uniq[1])
mid_AB = A[:,0]
mid_MN = (M[:,0] + N[:,0]) * 0.5
mid_z = -abs(mid_AB - mid_MN) * 0.4
mid_x = abs(mid_AB + mid_MN) * 0.5

srcLists = []
appres = []
geometric = []
voltage = []
inds_data = []

for iSrc in range (nSrc):
    inds = uniq[2] == iSrc
    # TODO: y-location should be assigned ...
    locsM = M[inds,:]
    locsN = N[inds,:]        
    inds_data.append(np.arange(len(inds))[inds])
    rx = EM.Static.DC.Rx.Dipole(locsM, locsN)    
    locA = uniq[0][iSrc,:3]
    locB = uniq[0][iSrc,3:]    
    src = EM.Static.DC.Src.Pole([rx], locA)        
#     src = EM.Static.DC.Src.Dipole([rx], locA, locB)    
    geometric.append(getGeometricFactor(locA, locB, locsM, locsN))    
    appres.append(dcipdata[:,8][inds])
    voltage.append(dcipdata[:,9][inds])
    srcLists.append(src)
inds_data = np.hstack(inds_data)
geometric = np.hstack(geometric)
dobs_appres = np.hstack(appres)
dobs_voltage = np.hstack(voltage) * 1e-3
DCsurvey = EM.Static.DC.Survey(srcLists)
DCsurvey.dobs = dobs_voltage

m0 = np.ones(actind.sum())*np.log(1e-3)
actMap = Maps.InjectActiveCells(mesh, actind, np.log(1e-8))
mapping = Maps.ExpMap(mesh) * actMap
problem = EM.Static.DC.Problem3D_N(mesh, sigmaMap=mapping)
problem.Solver = PardisoSolver
if DCsurvey.ispaired:
    DCsurvey.unpair()
problem.pair(DCsurvey)

f = problem.fields(np.log(sigma)[actind])
dpred = DCsurvey.dpred(np.log(sigma)[actind], f=f)
appres = dpred / geometric
dcdata = Data(DCsurvey, v=dpred)
appresdata = Data(DCsurvey, v=appres)

def vizdata(isrc):
    fig = plt.figure(figsize = (7, 2))
    src = srcLists[isrc]
    rx = src.rxList[0]    
    data_temp = dcdata[src, rx]
    appres_temp = appresdata[src, rx]
    midx = (rx.locs[0][:,0] + rx.locs[1][:,0]) * 0.5
    midz = (rx.locs[0][:,2] + rx.locs[1][:,2]) * 0.5
    ax = plt.subplot(111)
    ax_1 = ax.twinx()
    ax.plot(midx, data_temp, 'k.-')
    ax_1.plot(midx, appres_temp, 'r.-')
    ax.set_xlim(5000, 11000)
    ax.set_ylabel("Voltage")
    ax_1.set_ylabel("$\\rho_a$ (Ohm-m)")
    ax.grid(True)
    plt.show()
interact(vizdata, isrc=(0, DCsurvey.nSrc-1, 1))

fig = plt.figure(figsize = (7, 1.5))
def vizJ(isrc):
    indy = 6
    src = srcLists[isrc]
    rx = src.rxList[0]
    out = mesh.plotSlice(f[src, 'j'], vType="E", normal="Y", view="vec", ind=indy, streamOpts={"color":"k"}, pcolorOpts={"norm": LogNorm(), "cmap":"viridis"}, clim=(1e-10, 1e-4))
    plt.plot(src.loc[0], src.loc[1], 'ro')
    plt.ylim(-800, 250)
    plt.xlim(5000, 11000)
    plt.gca().set_aspect(2.)
    # plt.title(("y= %d m")%(mesh.vectorCCy[indy]))
    plt.title("")
    cb = plt.colorbar(out[0], orientation="horizontal")
    cb.set_label("Current density (A/m$^2$)")
    midx = (rx.locs[0][:,0] + rx.locs[1][:,0]) * 0.5
    midz = (rx.locs[0][:,2] + rx.locs[1][:,2]) * 0.5
    plt.plot(midx, midz, 'g.', ms=4)
    plt.gca().get_xlim()
    plt.show()
interact(vizJ, isrc=(0, DCsurvey.nSrc-1, 1))

vmin, vmax = 1, 1e4
appres = dpred/geometric
temp = appres.copy()
Utils.plot2Ddata(np.c_[mid_x[inds_data], mid_z[inds_data]], temp, ncontour=100, dataloc=True, scale="log", contourOpts={"vmin":np.log10(vmin), "vmax":np.log10(vmax)})
cb = plt.colorbar(out[0], orientation="horizontal", format="1e%.0f", ticks=np.linspace(np.log10(vmin), np.log10(vmax), 3))
cb.set_label("Resistivity (Ohm-m)")
# plt.title("Line 12150N")

vmin, vmax = 1, 1e4
temp = dcipdata[:,8].copy()
temp[dcipdata[:,8]<vmin] = vmin
temp[dcipdata[:,8]>vmax] = vmax
out = Utils.plot2Ddata(np.c_[mid_x[inds_data], mid_z[inds_data]], temp[inds_data], ncontour=100, dataloc=True, scale="log")
cb = plt.colorbar(out[0], orientation="horizontal", format="1e%.0f", ticks=np.linspace(np.log10(vmin), np.log10(vmax), 3))
cb.set_label("Resistivity (Ohm-m)")
# plt.title("Line 12150N")

