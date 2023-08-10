from SimPEG import Mesh
from SimPEG.Utils import mkvc, surface2ind_topo
from SimPEG import Maps
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import Optimization
from SimPEG import InvProblem
from SimPEG import Directives
from SimPEG import Inversion
from SimPEG import PF
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# First we need to define the direction of the inducing field
# As a simple case, we pick a vertical inducing field of magnitude 50,000nT. 
# From old convention, field orientation is given as an azimuth from North 
# (positive clockwise) and dip from the horizontal (positive downward).
H0 = (60000.,90.,0.)


# Create a mesh
dx    = 5.

hxind = [(dx,5,-1.3), (dx, 10), (dx,5,1.3)]
hyind = [(dx,5,-1.3), (dx, 10), (dx,5,1.3)]
hzind = [(dx,5,-1.3),(dx, 10)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

# Get index of the center
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)

# Lets create a simple Gaussian topo and set the active cells
[xx,yy] = np.meshgrid(mesh.vectorNx,mesh.vectorNy)
zz = -np.exp( ( xx**2 + yy**2 )/ 75**2 ) + mesh.vectorNz[-1]

topo = np.c_[mkvc(xx),mkvc(yy),mkvc(zz)] # We would usually load a topofile

actv = surface2ind_topo(mesh,topo,'N') # Go from topo to actv cells
actv = np.asarray([inds for inds, elem in enumerate(actv, 1) if elem], dtype = int) - 1
#nC   = mesh.nC 
#actv = np.asarray(range(mesh.nC))

# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)
nC = len(actv)

# Create and array of observation points
xr = np.linspace(-20., 20., 20)
yr = np.linspace(-20., 20., 20)
X, Y = np.meshgrid(xr, yr)

# Let just put the observation above the topo
Z = -np.exp( ( X**2 + Y**2 )/ 75**2 ) + mesh.vectorNz[-1] + 5.
#Z = np.ones(shape(X)) * mesh.vectorCCz[-1]

# Create a MAGsurvey
rxLoc = np.c_[mkvc(X.T), mkvc(Y.T), mkvc(Z.T)]
rxLoc = PF.BaseMag.RxObs(rxLoc)
srcField = PF.BaseMag.SrcField([rxLoc],param = H0)
survey = PF.BaseMag.LinearSurvey(srcField) 

# We can now create a susceptibility model and generate data
# Lets start with a simple block in half-space
model = np.zeros((mesh.nCx,mesh.nCy,mesh.nCz))
model[(midx-2):(midx+2),(midy-2):(midy+2),-6:-2] = 0.02
model = mkvc(model)
model = model[actv]

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Creat reduced identity map
idenMap = Maps.IdentityMap(nP = nC)

# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv)

# Pair the survey and problem
survey.pair(prob)

# Compute linear forward operator and compute some data
d = prob.fields(model)

# Plot the model
m_true = actvMap * model
m_true[m_true==-100] = np.nan
plt.figure()
ax = plt.subplot(212)
mesh.plotSlice(m_true, ax = ax, normal = 'Y', ind=midy, grid=True, clim = (0., model.max()), pcolorOpts={'cmap':'viridis'})
plt.title('A simple block model.')
plt.xlabel('x'); plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

# We can now generate data
data = d + np.random.randn(len(d)) # We add some random Gaussian noise (1nT)
wd = np.ones(len(data))*1. # Assign flat uncertainties


plt.subplot(221)
plt.imshow(d.reshape(X.shape), extent=[xr.min(), xr.max(), yr.min(), yr.max()])
plt.title('True data.')
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()

plt.subplot(222)
plt.imshow(data.reshape(X.shape), extent=[xr.min(), xr.max(), yr.min(), yr.max()])
plt.title('Data + Noise')
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()

plt.tight_layout()

# Create distance weights from our linera forward operator
wr = np.sum(prob.G**2.,axis=0)**0.5
wr = ( wr/np.max(wr) )


wr_FULL = actvMap * wr
wr_FULL[wr_FULL==-100] = np.nan
plt.figure()
ax = plt.subplot()
mesh.plotSlice(wr_FULL, ax = ax, normal = 'Y', ind=midx, grid=True, clim = (0, wr.max()),pcolorOpts={'cmap':'viridis'})
plt.title('Distance weighting')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

#survey.makeSyntheticData(data, std=0.01)
survey.dobs= data
survey.std = wd
survey.mtrue = model

# Create a regularization
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.cell_weights = wr
reg.norms = [0, 1, 1, 1]
reg.eps_p = 1e-3
reg.eps_1 = 1e-3
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1/wd

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=100 ,lower=0.,upper=1., maxIterLS = 20, maxIterCG= 10, tolCG = 1e-3)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
# Use pick a treshold parameter empirically based on the distribution of model
# parameters (run last cell to see the histogram before and after IRLS)
IRLS = Directives.Update_IRLS(f_min_change = 1e-3, minGNiter=3)
update_Jacobi = Directives.Update_lin_PreCond()
inv = Inversion.BaseInversion(invProb, directiveList=[betaest, IRLS, update_Jacobi])

m0 = np.ones(nC)*1e-4

mrec = inv.run(m0)

# Here is the recovered susceptibility model
ypanel = midx
zpanel = -4
m_l2 = actvMap * IRLS.l2model
m_l2[m_l2==-100] = np.nan

m_lp = actvMap * mrec
m_lp[m_lp==-100] = np.nan

m_true = actvMap * model
m_true[m_true==-100] = np.nan

plt.figure()

#Plot L2 model
ax = plt.subplot(231)
mesh.plotSlice(m_l2, ax = ax, normal = 'Z', ind=zpanel, grid=True, clim = (0., model.max()),pcolorOpts={'cmap':'viridis'})
plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),color='w')
plt.title('Plan l2-model.')
plt.gca().set_aspect('equal')
plt.ylabel('y')
ax.xaxis.set_visible(False)
plt.gca().set_aspect('equal', adjustable='box')

# Vertica section
ax = plt.subplot(234)
mesh.plotSlice(m_l2, ax = ax, normal = 'Y', ind=midx, grid=True, clim = (0., model.max()),pcolorOpts={'cmap':'viridis'})
plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),color='w')
plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([Z.min(),Z.max()]),color='k')
plt.title('E-W l2-model.')
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

#Plot Lp model
ax = plt.subplot(232)
mesh.plotSlice(m_lp, ax = ax, normal = 'Z', ind=zpanel, grid=True, clim = (0., model.max()),pcolorOpts={'cmap':'viridis'})
plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),color='w')
plt.title('Plan lp-model.')
plt.gca().set_aspect('equal')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.gca().set_aspect('equal', adjustable='box')


# Vertical section
ax = plt.subplot(235)
mesh.plotSlice(m_lp, ax = ax, normal = 'Y', ind=midx, grid=True, clim = (0., model.max()),pcolorOpts={'cmap':'viridis'})
plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),color='w')
plt.title('E-W lp-model.')
plt.gca().set_aspect('equal')
ax.yaxis.set_visible(False)
plt.xlabel('x')
plt.gca().set_aspect('equal', adjustable='box')

#Plot True model
ax = plt.subplot(233)
mesh.plotSlice(m_true, ax = ax, normal = 'Z', ind=zpanel, grid=True, clim = (0., model.max()),pcolorOpts={'cmap':'viridis'})
plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),color='w')
plt.title('Plan true model.')
plt.gca().set_aspect('equal')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.gca().set_aspect('equal', adjustable='box')


# Vertical section
ax = plt.subplot(236)
mesh.plotSlice(m_true, ax = ax, normal = 'Y', ind=midx, grid=True, clim = (0., model.max()),pcolorOpts={'cmap':'viridis'})
plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),color='w')
plt.title('E-W true model.')
plt.gca().set_aspect('equal')
plt.xlabel('x')
ax.yaxis.set_visible(False)
plt.gca().set_aspect('equal', adjustable='box')

# Plot predicted data and residual
plt.figure()
pred = prob.fields(mrec)  #: this is matrix multiplication!!

plt.subplot(221)
plt.imshow(data.reshape(X.shape))
plt.title('Observed data.')
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()

plt.subplot(222)
plt.imshow(pred.reshape(X.shape))
plt.title('Predicted data.')
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()

plt.subplot(223)
plt.imshow(data.reshape(X.shape) - pred.reshape(X.shape))
plt.title('Residual data.')
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()

plt.subplot(224)
plt.imshow( (data.reshape(X.shape) - pred.reshape(X.shape)) / wd.reshape(X.shape) )
plt.title('Normalized Residual')
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()

plt.tight_layout()



