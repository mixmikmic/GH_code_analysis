## First we need to load all the libraries and set up the path
## for the input files. Same files as used by the online tutorial
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
from SimPEG.Utils.io_utils import download
import matplotlib
import matplotlib.colors as colors
import scipy as sp
import numpy as np
import time as tm
import os
import shutil
from ipywidgets.widgets import interact, IntSlider
get_ipython().run_line_magic('matplotlib', 'inline')

psep = os.path.sep
url = 'https://storage.googleapis.com/simpeg/tkc_synthetic/potential_fields/'
cloudfiles = ['MagData.obs', 'Mesh.msh',
              'Initm.sus', 'SimPEG_PF_Input.inp']

downloads = download([url+f for f in cloudfiles], folder='./MagTKC/', overwrite=True)
downloads = dict(zip(cloudfiles, downloads))
input_file = downloads['SimPEG_PF_Input.inp']

# Read in the input file which included all parameters at once (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(input_file)

# We already have created a mesh, model and survey for this example.
# All the elements are stored in the driver, and can be accessed like this:
mesh = driver.mesh
survey = driver.survey

# We did not include a topography file in this example as the information
# about inactive cells is already captured in our starting model.
# Line 6 of the input file specifies a VALUE to be used as inactive flag.

# Get the active cells
actv = driver.activeCells
nC = len(actv) # Number of active cells
ndv = -100

# Create active map to return to full space after the inversion 
actvMap = Maps.InjectActiveCells(mesh, actv, ndv)

# Create a reduced identity map for the inverse problem
idenMap = Maps.IdentityMap(nP=nC)

# Now that we have a model and a survey we can build the linear system ...
# Create the forward model operator (the argument forwardOnly=False store the forward matrix to memory)
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv, forwardOnly=False, rtype = 'tmi')

# Pair the survey and problem
survey.pair(prob)

# Fist time that we ask for predicted data,
# the dense matrix T is calculated.
# This is generally the bottleneck of the integral formulation in terms of cost
d = prob.fields(driver.m0)

# Add noise to the data and assign uncertainties
data = d + np.random.randn(len(d)) # We add some random Gaussian noise (1nT)
wd = np.ones(len(data))*1. # Assign flat uncertainties

survey.dobs = data
survey.std = wd

# [OPTIONAL] You can write the observations to UBC format here
#PF.Magnetics.writeUBCobs('MAG_Synthetic_data.obs',survey,data)
fig = PF.Magnetics.plot_obs_2D(survey.srcField.rxList[0].locs,d=data ,title='Mag Obs')

# It is potential fields, so we will need to push the inverison down
# Create distance weights from our linera forward operator
wr = np.sum(prob.G**2.,axis=0)**0.5
wr = ( wr/np.max(wr) )
    
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.norms = driver.lpnorms
reg.eps_p = 5e-3
reg.eps_q = 5e-3 
reg.cell_weights = wr

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1/wd

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=100 ,lower=0.,upper=1., maxIterLS = 20, maxIterCG= 10, tolCG = 1e-3)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
# Use pick a treshold parameter empirically based on the distribution of model
# parameters (run last cell to see the histogram before and after IRLS)
IRLS = Directives.Update_IRLS(f_min_change = 1e-4, minGNiter=2)
update_Jacobi = Directives.Update_lin_PreCond()
inv = Inversion.BaseInversion(invProb, directiveList=[betaest,IRLS,update_Jacobi])

m0 = np.ones(idenMap.nP)*1e-4

# Run inversion...
mrec = inv.run(m0)

# Get the final model back to full space and plot!!
m_lp = actvMap*mrec
m_lp[m_lp==ndv] = np.nan

# Get the smooth model aslo
m_l2 = actvMap*IRLS.l2model
m_l2[m_l2==ndv] = np.nan

m_true = actvMap*driver.m0
m_true[m_true==ndv] = np.nan
#[OPTIONAL] Save both models to file
#Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_MAG_l2l2.sus',m_l2)
#Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_MAG_lplq.sus',m_lp)

# Plot the recoverd models 
vmin, vmax = 0., 0.015

mesh = Mesh.TensorMesh([mesh.hx, mesh.hy, mesh.hz],x0="CCN")

def slide(s,normal):
    
    if normal == "Z":
        fig = plt.figure(figsize=(10*1.2, 8))
    else:
        fig = plt.figure(figsize=(10*1.2, 4))
        
    ax1 = plt.subplot(2,2,3)
    dat = mesh.plotSlice(m_lp, ax = ax1, normal=normal, ind=s, clim=np.r_[vmin, vmax],pcolorOpts={'cmap':'viridis'})
#     plt.colorbar(dat[0])
    plt.gca().set_aspect('equal')
    plt.title('Compact model')
    
    if normal == "Z":
        plt.xlim(-600, 600)
        plt.ylim(-600, 600.)    
    else:
        plt.xlim(-600, 600)
        plt.ylim(-500, 0.) 
        
    ax2 = plt.subplot(2,2,1)
    dat = mesh.plotSlice(m_l2, ax = ax2, normal=normal, ind=s, clim=np.r_[vmin, vmax],pcolorOpts={'cmap':'viridis'})
#     plt.colorbar(dat[0])
    plt.gca().set_aspect('equal')
    plt.title('Smooth model')
    
    if normal == "Z":
        plt.xlim(-600, 600)
        plt.ylim(-600, 600.)    
    else:
        plt.xlim(-600, 600)
        plt.ylim(-500, 0.) 
        
    ax2.set_xticklabels([])
        
    ax2 = plt.subplot(1,2,2)
    dat = mesh.plotSlice(m_true, ax = ax2, normal=normal, ind=s, clim=np.r_[vmin, vmax],pcolorOpts={'cmap':'viridis'})
#     plt.colorbar(dat[0])
    plt.gca().set_aspect('equal')
    plt.title('True model')
    
    pos =  ax2.get_position()

    ax2.yaxis.set_visible(False)
    if normal == "Z":
        plt.xlim(-600, 600)
        plt.ylim(-600, 600.) 
        ax2.set_position([pos.x0 -0.04 , pos.y0,  pos.width, pos.height])
    else:
        plt.xlim(-600, 600)
        plt.ylim(-500, 0.) 

    pos =  ax2.get_position()
    cbarax = fig.add_axes([pos.x0 + 0.375 , pos.y0 + 0.05,  pos.width*0.1, pos.height*0.75])  ## the parameters are the specified position you set
    cb = fig.colorbar(dat[0],cax=cbarax, orientation="vertical", ax = ax2, ticks=np.linspace(vmin,vmax, 4))
    cb.set_label("Susceptibility (SI)",size=12)
    
    #{OPTIONAL} Save the figure to png
    #fig.savefig('PF_Compact.png',dpi = 150)
    
interact(slide, s=(0,36), normal=['Y','Z','X'])

# interact(lambda ind: viz(m_l2, ind, normal="Z"), ind=IntSlider(min=0, max=32,step=1, value=28))

# Lets compare the distribution of model parameters and model gradients
plt.figure(figsize=[15,5])
ax = plt.subplot(121)
plt.hist(IRLS.l2model,100)
plt.plot((reg.eps_p,reg.eps_p),(0,mesh.nC),'r--')
plt.yscale('log', nonposy='clip')
plt.title('Hist model - l2-norm')
plt.legend(['$\epsilon_p$'])

ax = plt.subplot(122)
plt.hist(reg.regmesh.cellDiffxStencil*IRLS.l2model,100)
plt.plot((reg.eps_q,reg.eps_q),(0,mesh.nC),'r--')
plt.yscale('log', nonposy='clip')
plt.title('Hist model gradient - l2-norm')
plt.legend(['$\epsilon_q$'])

# Lets look at the distribution of model parameters and model gradients
plt.figure(figsize=[15,5])
ax = plt.subplot(121)
plt.hist(mrec,100)
plt.plot((reg.eps_p,reg.eps_p),(0,mesh.nC),'r--')
plt.yscale('log', nonposy='clip')
plt.title('Hist model - lp-norm')
plt.legend(['$\epsilon_p$'])

ax = plt.subplot(122)
plt.hist(reg.regmesh.cellDiffxStencil*mrec,100)
plt.plot((reg.eps_q,reg.eps_q),(0,mesh.nC),'r--')
plt.yscale('log', nonposy='clip')
plt.title('Hist model gradient - lp-norm')
plt.legend(['$\epsilon_q$'])



