# The usual, we need to load some libraries
from SimPEG import Mesh, Utils, Maps, PF
from SimPEG import mkvc, Regularization, DataMisfit, Optimization, InvProblem, Directives,Inversion
from SimPEG.Utils import mkvc
from SimPEG.Utils.io_utils import download
import numpy as np
import scipy as sp
import os
get_ipython().magic('pylab inline')

# Download data from the cloud 
url = "https://storage.googleapis.com/simpeg/kevitsa_synthetic/"

cloudfiles = [
    'Mesh_global_100m_padded.msh','GravSim.dat',
    'Kevitsa.topo', 'SimPEG_GRAV.inp'
]
keys = ['mesh', 'data', 'topo', 'input']

# Download to ./KevitsaGrav
files = download([url+f for f in cloudfiles], folder='./KevitsaGrav', overwrite=True)
files = dict(zip(keys, files))  # allows us to name the files

# Read in the input file which included all parameters at once (mesh, topo, model, survey, inv param, etc.)
inputFile = files['input'] # input file was the last downloaded
driver = PF.GravityDriver.GravityDriver_Inv()
driver.basePath = './KevitsaGrav'

# All the parameters in the input files can be access via the driver object
# For example, to get the survey:
obs = driver.readGravityObservations(files['data'])
mesh = Mesh.TensorMesh.readUBC(files['mesh'])

# The gridded data holds 20k+ observation points, too large for a quick inversion
# Let's grab a random subset
nD = 500
indx = randint(0,high=obs.dobs.shape[0],size=nD)

# Create a new downsampled survey
locXYZ  = obs.srcField.rxList[0].locs[indx,:]

rxLoc = PF.BaseGrav.RxObs(locXYZ)
srcField = PF.BaseGrav.SrcField([rxLoc])
survey = PF.BaseGrav.LinearSurvey(srcField)
survey.dobs = obs.dobs[indx]
survey.std = obs.std[indx]

ph = PF.Gravity.plot_obs_2D(survey.srcField.rxList[0].locs, survey.dobs,'Observed Data')

# Create a mesh, we will start coarse. Feel free to change the
# the mesh, but make sure you have enough memory and coffee brakes...
dx = 200.
npad = 5
hxind = [(dx, npad, -1.3), (dx, 65), (dx, npad, 1.3)]
hyind = [(dx, npad, -1.3), (dx, 45), (dx, npad, 1.3)]
hzind = [(dx, npad, -1.3), (150, 15), (10, 10, -1.3), (10,5)]

# Create the mesh and move the location to the center of the data
mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CC0')
mesh._x0 += [np.mean(locXYZ[:,0]), np.mean(locXYZ[:,1]), np.max(locXYZ[:,2])-np.sum(mesh.hz)]

ax = mesh.plotGrid()

# We will get the topography from the input file
topo = np.genfromtxt(files['topo'], skip_header=1)

# Find the active cells
actv = Utils.surface2ind_topo(mesh, topo, 'N')

actv = np.asarray(
    [inds for inds, elem in enumerate(actv, 1) if elem], dtype=int
) - 1

nC = len(actv)

print("Number of data points: " + str(nD))
print("Number of model cells: " + str(nC))

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Create reduced identity map
idenMap = Maps.IdentityMap(nP=nC)
mstart = np.ones(nC)*1e-4

# Create gravity problem
prob = PF.Gravity.GravityIntegral(mesh, rhoMap=idenMap, actInd=actv)

survey.pair(prob)

# Make depth weighting, 
# this will also require the calculation of the forward operator ... time for coffee
wr = np.sum(prob.G**2., axis=0)**0.5
wr = (wr/np.max(wr))

# % Create inversion objects
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.cell_weights = wr
reg.norms = [0,2,2,2]


opt = Optimization.ProjectedGNCG(maxIter=100, lower=-.5,upper=0.5, maxIterLS = 20, maxIterCG= 10, tolCG = 1e-3)
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# This is where the misfit function and regularization are put together
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Here are few directives to make the inversion work and apply sparsity.
# After the l2, beta is re-adjusted on the fly to stay near the target misfit
betaest = Directives.BetaEstimate_ByEig()
IRLS = Directives.Update_IRLS(f_min_change=1e-4, minGNiter=3)
update_Jacobi = Directives.Update_lin_PreCond()
inv = Inversion.BaseInversion(invProb, directiveList=[betaest, IRLS,
                                                      update_Jacobi])
# Run the inversion
mrec = inv.run(mstart)

# Here is a quick script to slice through the final model
import ipywidgets as widgets

def ModSlicer(mesh, model):

    
    def plotIt(m, normal, panel, vmin, vmax):

        
        ypanel = int(mesh.vnC[1]/2)
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(211)
        ph = mesh.plotSlice(model[m], ax=ax, normal=normal, ind=int(panel),
                       grid=True,
                       clim=(vmin,vmax), pcolorOpts={'cmap': 'jet', })
        

        # Set default limits
        if normal == 'X':
            Xlim = [mesh.vectorNy.min(), mesh.vectorNy.max()] 
            Ylim = [mesh.vectorNz.min(), mesh.vectorNz.max()] 
        elif normal == 'Y':
            Xlim = [mesh.vectorNx.min(), mesh.vectorNx.max()] 
            Ylim = [mesh.vectorNz.min(), mesh.vectorNz.max()]  
        else:
            Xlim = [mesh.vectorNx.min(), mesh.vectorNx.max()] 
            Ylim = [mesh.vectorNy.min(), mesh.vectorNy.max()]  
            
        ax.set_xlim(Xlim)
        ax.set_ylim(Ylim)
        ax.set_aspect('equal')
        plt.colorbar(ph[0])
        plt.title('Plan lp-model.')
        plt.gca().set_aspect('equal')
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        
    out = widgets.interactive(plotIt,
                              m = widgets.ToggleButtons(
                                        options=['l2', 'lp'],
                                        description='Model:'),
                              normal = widgets.ToggleButtons(
                                        options=['X', 'Y', 'Z'],
                                        description='Normal:',
                                        disabled=False,
                                        button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Description'),
                              panel = widgets.FloatSlider(min=0, max=mesh.vnC.max(), step=1,value=1, continuous_update=False),
                              vmin = widgets.FloatSlider(min=model['l2'][~np.isnan(model['l2'])].min(), max=model['l2'][~np.isnan(model['l2'])].max(), step=0.001,value=model['l2'][~np.isnan(model['l2'])].min(), continuous_update=False),
                              vmax = widgets.FloatSlider(min=model['l2'][~np.isnan(model['l2'])].min(), max=model['l2'][~np.isnan(model['l2'])].max(), step=0.001,value=model['l2'][~np.isnan(model['l2'])].max(), continuous_update=False),
)
    return out

# Plot the result
m_lp = actvMap * mrec
m_lp[m_lp == -100] = np.nan

m_l2 = actvMap*IRLS.l2model
m_l2[m_l2 == -100] = np.nan

model={'l2':m_l2,'lp':m_lp}

# Execute the ploting function
ModSlicer(mesh, model)



