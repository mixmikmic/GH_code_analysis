# Start by importing SimPEG, the potential fields package and 
import os
import numpy as np
import matplotlib.pyplot as plt

from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import Optimization
from SimPEG import InvProblem
from SimPEG import Directives
from SimPEG import Inversion
from SimPEG import PF
import SimPEG.PF as PF
from SimPEG.Utils.io_utils import download

get_ipython().magic('matplotlib inline')

# Download files from the remote repository

# This is where the files are stored on the cloud
url = 'https://storage.googleapis.com/simpeg/tkc_synthetic/potential_fields/'
cloudfiles = ['MagData.obs', 'Mesh.msh', 'Initm.sus', 'SimPEG_PF_Input.inp']

downloads = download([url+f for f in cloudfiles], folder='./PF_over_TKC/', overwrite=True)
downloads = dict(zip(cloudfiles, downloads))

driver = PF.MagneticsDriver.MagneticsDriver_Inv(downloads['SimPEG_PF_Input.inp'])

# Objects loaded from the input file are then accessible like this
mesh = driver.mesh
initm = driver.m0.copy()

initm[initm==-100] = np.nan

# Create a figure and plot sections
fig, ax1 = plt.figure(), plt.subplot(1,2,1)
mesh.plotSlice(initm, ax = ax1, normal='Z', ind=18, clim = (0,0.02), pcolorOpts={'cmap':'viridis'})
plt.gca().set_aspect('equal')
plt.title('Z: '+str(mesh.vectorCCz[18]) + " m")

ax2 = plt.subplot(1,2,2)
mesh.plotSlice(initm, ax = ax2, normal='Y', ind=16, clim = (0,0.02), pcolorOpts={'cmap':'viridis'})
plt.gca().set_aspect('equal')
plt.title('Y: '+str(mesh.vectorCCy[16])+' m')

plt.show()

# Get the survey
survey = driver.survey

# Get the active cells (below topography)
actv = driver.activeCells

# Create mapping to come back from the reduce space later
idenMap = Maps.IdentityMap(nP=len(actv))

# Now that we have a model and a survey we can build the linear system ...
# (use the argument forwardOnly=True to avoid storing the dense forward matrix)
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv, rtype = 'tmi')

# Pair the survey and problem (data and model space)
survey.pair(prob)

# Forward operators and data are calculated here (wait for it!)
d = prob.fields(driver.m0)

# Add noise to the data and assign uncertainties
survey.dobs = d + np.random.randn(len(d)) # We add some random Gaussian noise (1 nT)
survey.std = np.ones(len(d))*1. # Assign flat uncertainties (1 nT)

# Then we can plot with the build-in function
PF.Magnetics.plot_obs_2D(survey.srcField.rxList[0].locs,d=survey.dobs ,title='Mag Obs')
plt.show()

# It is potential fields, so we will need to push the inverison down
# Create distance weights from our linera forward operator
wr = np.sum(prob.G**2.,axis=0)**0.5
wr = ( wr/np.max(wr) )

reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.cell_weights = wr
reg.norms=driver.lpnorms
reg.eps_p = 3e-4
reg.eps_q = 3e-4

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1/survey.std

opt = Optimization.ProjectedGNCG(maxIter=100 ,lower=0.,upper=1., maxIterLS = 20, maxIterCG= 10, tolCG = 1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# We add a few directives
# First to guess the initial beta
betaest = Directives.BetaEstimate_ByEig()

# Second, we add a pre-conditioner to speedup the CG solves
update_Jacobi = Directives.Update_lin_PreCond()

# This is the directive that will orchestrate the IRLS steps
IRLS = Directives.Update_IRLS(f_min_change = 1e-4, minGNiter=3)

# We add the directives to the inverse problem
inv = Inversion.BaseInversion(invProb, directiveList=[betaest,IRLS,update_Jacobi])

m0 = np.ones(len(actv))*1e-4
mrec = inv.run(m0)


# Create active map to return to full space after the inversion 
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Map to full space the final model and l2 model
m_lp = actvMap*mrec
m_l2 = actvMap*IRLS.l2model

# # Once it is done, we can save the models (l2 and lp) to a file
# Mesh.TensorMesh.writeModelUBC(mesh, 'SimPEG_MAG_l2l2.sus',m_l2)
# Mesh.TensorMesh.writeModelUBC(mesh, 'SimPEG_MAG_lplq.sus',m_lp)

m_true = Mesh.TensorMesh.readModelUBC(mesh, downloads["Initm.sus"])
m_lp[m_lp==-100] = np.nan
m_l2[m_l2==-100] = np.nan
m_true[m_true==-100] = np.nan

fig = plt.figure()
vmin, vmax = 0.0, 0.015
xmin, xmax = -500 + 557300, 500 + 557300
ymin, ymax = -500 + 7133600, 500 + 7133600
zmin, zmax = -500 + 450, 0 + 450
indz = 17
indx = 17

# Axis label
x = np.linspace(xmin+200, xmax-200,3)
y = np.linspace(zmin+50, zmax-50,3)

ax1 = plt.subplot(1,1,1)
pos =  ax1.get_position()
ax1.set_position([pos.x0-0.1, pos.y0+0.3,  pos.width*0.5, pos.height*0.5])
dat = mesh.plotSlice(m_l2, ax = ax1, normal='Z', ind=indz, clim=np.r_[vmin, vmax],pcolorOpts={'cmap':'viridis'})
#     plt.colorbar(dat[0])
plt.gca().set_aspect('equal')
plt.title('Smooth')
ax1.xaxis.set_visible(False)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel('Northing (m)')
labels = ax1.get_yticklabels()
plt.setp(labels, rotation=90)

# ax2 = plt.subplot(2,2,3)
pos =  ax1.get_position()
ax2 = fig.add_axes([pos.x0+0.0525, pos.y0 - 0.315,  pos.width*0.725, pos.height])
# ax2.yaxis.set_visible(False)
# ax2.set_position([pos.x0 -0.04 , pos.y0,  pos.width, pos.height])

dat = mesh.plotSlice(m_l2, ax = ax2, normal='Y', ind=indx, clim=np.r_[vmin, vmax],pcolorOpts={'cmap':'viridis'})
#     plt.colorbar(dat[0])
plt.gca().set_aspect('equal')
plt.title('')
plt.xlim(xmin, xmax)
plt.ylim(zmin, zmax)
ax2.set_xticks(x)
ax2.set_xticklabels(map(str, map(int, x)),size=12)
plt.xlabel('Easting (m)')
plt.ylabel('Elev. (m)')
ax2.set_yticks(y)
ax2.set_yticklabels(map(str, map(int, y)),size=12)
labels = ax2.get_yticklabels()
plt.setp(labels, rotation=90)

## Add compact model
ax3 = fig.add_axes([pos.x0+0.3, pos.y0,  pos.width, pos.height])
dat = mesh.plotSlice(m_lp, ax = ax3, normal='Z', ind=indz, clim=np.r_[vmin, vmax],pcolorOpts={'cmap':'viridis'})
#     plt.colorbar(dat[0])
plt.gca().set_aspect('equal')
plt.title('Compact')
ax3.xaxis.set_visible(False)
ax3.yaxis.set_visible(False)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

ax4 = fig.add_axes([pos.x0+0.3525, pos.y0 - 0.315,  pos.width*0.725, pos.height])
# ax2.yaxis.set_visible(False)
# ax2.set_position([pos.x0 -0.04 , pos.y0,  pos.width, pos.height])

dat = mesh.plotSlice(m_lp, ax = ax4, normal='Y', ind=indx, clim=np.r_[vmin, vmax],pcolorOpts={'cmap':'viridis'})
#     plt.colorbar(dat[0])
plt.gca().set_aspect('equal')
ax4.yaxis.set_visible(False)
plt.title('')
plt.xlim(xmin, xmax)
plt.ylim(zmin, zmax)
ax4.set_xticks(x)
ax4.set_xticklabels(map(str, map(int, x)),size=12)
plt.xlabel('')
# ylabel('Elev. (m)')

## Add True model
ax5 = fig.add_axes([pos.x0+0.6, pos.y0,  pos.width, pos.height])
dat = mesh.plotSlice(m_true, ax = ax5, normal='Z', ind=indz, clim=np.r_[vmin, vmax],pcolorOpts={'cmap':'viridis'})
#     plt.colorbar(dat[0])
plt.gca().set_aspect('equal')
plt.title('True')
ax5.xaxis.set_visible(False)
ax5.yaxis.set_visible(False)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

ax6 = fig.add_axes([pos.x0+0.6525, pos.y0 - 0.315,  pos.width*0.725, pos.height])
# ax2.yaxis.set_visible(False)
# ax2.set_position([pos.x0 -0.04 , pos.y0,  pos.width, pos.height])

dat = mesh.plotSlice(m_true, ax = ax6, normal='Y', ind=indx, clim=np.r_[vmin, vmax],pcolorOpts={'cmap':'viridis'})
#     plt.colorbar(dat[0])
plt.gca().set_aspect('equal')
ax6.yaxis.set_visible(False)
plt.title('')
plt.xlim(xmin, xmax)
plt.ylim(zmin, zmax)
ax6.set_xticks(x)
ax6.set_xticklabels(map(str, map(int, x)),size=12)
plt.xlabel('')
# ylabel('Elev. (m)')

pos =  ax4.get_position()
cbarax = fig.add_axes([pos.x0 , pos.y0+0.05 ,  pos.width, pos.height*0.1])  ## the parameters are the specified position you set
cb = fig.colorbar(dat[0],cax=cbarax, orientation="horizontal", ax = ax4, ticks=np.linspace(vmin,vmax, 4),format='%.3f')
cbarax.tick_params(labelsize=12)
# cb.ax.xaxis.set_label_position('top')
cb.set_label("Susceptibility (SI)",size=14)

plt.show()

