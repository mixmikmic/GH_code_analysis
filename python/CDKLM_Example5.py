#Lets have matplotlib "inline"
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

#Import packages we need
import numpy as np
from matplotlib import animation, rc
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec

import os, pyopencl, datetime, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../')))

# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile

#Finally, import our simulator
from SWESimulators import FBL, CTCS

#Set large figure sizes
rc('figure', figsize=(16.0, 12.0))
rc('animation', html='html5')

#Finally, import our simulator
from SWESimulators import FBL, CTCS, KP07, CDKLM16, RecursiveCDKLM16, SimWriter, PlotHelper, Common
from SWESimulators.BathymetryAndICs import *

#Make sure we get compiler output from OpenCL
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

#Set which CL device to use, and disable kernel caching
if (str.lower(sys.platform).startswith("linux")):
    os.environ["PYOPENCL_CTX"] = "0"
else:
    os.environ["PYOPENCL_CTX"] = "1"
os.environ["CUDA_CACHE_DISABLE"] = "1"
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
os.environ["PYOPENCL_NO_CACHE"] = "1"

#Create OpenCL context
cl_ctx = pyopencl.create_some_context()
print "Using ", cl_ctx.devices[0].name

#Create output directory for images
imgdir='images_' + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
os.makedirs(imgdir)
print "Saving images to " + imgdir 

"""
Class that defines the domain, initial conditions and boundary conditions to generate geostrophic equilibrium steady state,
as defined in the CDKLM 2016 paper, Example 5.
"""
class GeostrophicEquilibrium:
    
    def __init__ (self, cl_ctx, case, newDt=None):
        
        self.cl_ctx = cl_ctx
        
        self.f = 1.0 # s^-1   Coriolis parameter
        self.g = 1.0 # m/s^2   Gravitational acceleration
        self.R = 0.0 # m/s   Bottom friction coefficient
        self.nx = 400
        self.ny = 400
        self.Lx = 20.0 # m   domain length in x-direction
        self.Ly = 20.0 # m   domain length in y-direction
        self.dt = 0.20 # s   Time increment
    
        if newDt is not None:
            self.dt = newDt
        print "dt: " + str(self.dt)
            
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny

        
        assert (len(case) == 1), "Invalid case specification"
        assert (case == "4"), "Only case 4 is valid input to this class"
        self.case = case
        
        self.windStressParams = Common.WindStressParams(type=99)
        
        self.eta0 = None
        self.u0 = None
        self.v0 = None
        self.H = None
        self.Bi = None
        self.sim = None
        self.scheme = None

        self.boundaryConditions = Common.BoundaryConditions() # Wall boundaries
        
        
        self.ghosts = None
        self.validDomain = None
        self.dataShape = None
        
        # Required for using plotting:
        #Calculate radius from center of bump for plotting
        x_center = self.Lx/2.0
        y_center = self.Ly/2.0
        self.y_coords, self.x_coords = np.mgrid[0:self.Ly:self.dy, 0:self.Lx:self.dx]
        self.x_coords = np.subtract(self.x_coords, x_center)
        self.y_coords = np.subtract(self.y_coords, y_center)
        self.radius = np.sqrt(np.multiply(self.x_coords, self.x_coords) + np.multiply(self.y_coords, self.y_coords))
        
        
    def _makeInitialConditions(self):
        self.dataShape = (self.ny + self.ghosts[0]+self.ghosts[2],                         self.nx + self.ghosts[1]+self.ghosts[3])
        
        self.h0 = np.zeros(self.dataShape, dtype=np.float32, order='C');
        self.u0 = np.zeros(self.dataShape, dtype=np.float32, order='C');
        self.v0 = np.zeros(self.dataShape, dtype=np.float32, order='C');
        self.Bi = np.zeros((self.dataShape[0]+1, self.dataShape[1]+1), dtype=np.float32, order='C')
        #print (self.dataShape) 
        
        for j in range(self.dataShape[0]):
            y = (j-self.ghosts[0]+0.5)*self.dy - self.Ly/2.0
            #print(j-self.ghosts[0], y)
            for i in range(self.dataShape[1]):
                x = (i-self.ghosts[1]+0.5)*self.dx - self.Lx/2.0
                #print(i-self.ghosts[1], x)
                
                radius = np.sqrt(2.5*x*x + 0.4*y*y)
                tanh_term = np.tanh(10.0*(radius - 1.0))
                self.h0[j,i] = 1.0 + 0.25*(1.0 - tanh_term)
                self.u0[j,i] =  (1.0/radius)*(      (1.0 - tanh_term*tanh_term)*y)
                self.v0[j,i] = -(1.0/radius)*( 6.25*(1.0 - tanh_term*tanh_term)*x)
        #fig = plt.figure()
        #singlePlotter = PlotHelper.SinglePlot(fig, self.x_coords, self.y_coords, self.v0, interpolation_type="None")
        
        
    def initializeSimulator(self, scheme, rk=2):
        self.scheme = scheme
        assert  ( scheme == "CDKLM16" or scheme == "KP07" or scheme == "RecursiveCDKLM16"),            "Currently only valid for CDKLM16,  KP07 and RecursiveCDKLM16 :)"

        if scheme == "CDKLM16":
            # Setting boundary conditions
            self.ghosts = [2,2,2,2]
            self.validDomain = [-2, -2, 2, 2]

            self._makeInitialConditions()
            
            reload(CDKLM16)
            self.sim = CDKLM16.CDKLM16(self.cl_ctx,                   self.h0, self.u0, self.v0, self.Bi,                   self.nx, self.ny,                   self.dx, self.dy, self.dt,                   self.g, self.f, self.R,                   wind_stress=self.windStressParams,                   boundary_conditions=self.boundaryConditions,                   reportGeostrophicEquilibrium=False,                   rk_order=rk)
            
        elif scheme == "RecursiveCDKLM16":
            print "Running RecursiveCDKLM16!!!"
            # Setting boundary conditions
            self.ghosts = [3,3,3,3]
            self.validDomain = [-3, -3, 3, 3]

            self._makeInitialConditions()
            
            reload(RecursiveCDKLM16)
            self.sim = RecursiveCDKLM16.RecursiveCDKLM16(self.cl_ctx,                   self.h0, self.u0, self.v0, self.Bi,                   self.nx, self.ny,                   self.dx, self.dy, self.dt,                   self.g, self.f, self.R,                   wind_stress=self.windStressParams,                   boundary_conditions=self.boundaryConditions)
            
        elif scheme == "KP07":
            # Setting boundary conditions
            self.ghosts = [2,2,2,2]
            self.validDomain = [-2, -2, 2, 2]
            
            self._makeInitialConditions()
            
            reload(KP07)
            self.sim = KP07.KP07(self.cl_ctx,                   self.h0, self.Bi, self.u0, self.v0,                   self.nx, self.ny,                   self.dx, self.dy, self.dt,                   self.g, self.f, self.R,                   wind_stress=self.windStressParams,                   boundary_conditions=self.boundaryConditions)
    


    
    def runSim(self, T, doPlot=True):
        assert (self.sim is not None), "Simulator not initiated."
        
        simulated_time = self.sim.step(T)
        eta1, u1, v1 = self.sim.download()
        eta1 -= 1.0
        
        if doPlot:
            fig = plt.figure()
            plotter = PlotHelper.PlotHelper(fig, self.x_coords, self.y_coords, self.radius,                     eta1[self.validDomain[2]:self.validDomain[0], self.validDomain[3]:self.validDomain[1]],                     u1[self.validDomain[2]:self.validDomain[0], self.validDomain[3]:self.validDomain[1]],                      v1[self.validDomain[2]:self.validDomain[0], self.validDomain[3]:self.validDomain[1]]);

            print("results for case " + self.case + " from simulator " + self.scheme)
            print("simulated_time: " + str(simulated_time))
        

#case4CDKLM = GeostrophicEquilibrium(cl_ctx, "4", newDt=0.02)
#case4CDKLM.initializeSimulator("CDKLM16")

get_ipython().run_cell_magic('time', '', 'case4CDKLM = GeostrophicEquilibrium(cl_ctx, "4", newDt=0.002)\ncase4CDKLM.initializeSimulator("CDKLM16")\ncase4CDKLM.runSim(8)\nprint("ux+vy, Kx, Ly: ", case4CDKLM.sim.downloadGeoEqNorm())')

get_ipython().run_cell_magic('time', '', 'case4CDKLM_rk3 = GeostrophicEquilibrium(cl_ctx, "4", newDt=0.002)\ncase4CDKLM_rk3.initializeSimulator("CDKLM16", rk=3)\ncase4CDKLM_rk3.runSim(8)\nprint("ux+vy, Kx, Ly: ", case4CDKLM_rk3.sim.downloadGeoEqNorm())')

get_ipython().run_cell_magic('time', '', 'case4KP07 = GeostrophicEquilibrium(cl_ctx, "4", newDt=0.002)\ncase4KP07.initializeSimulator("KP07")\ncase4KP07.runSim(8)')

get_ipython().run_cell_magic('time', '', '# WARNING: This takes some time (6 min)...\ncase4RecCDKLM = GeostrophicEquilibrium(cl_ctx, "4", newDt=0.002)\ncase4RecCDKLM.initializeSimulator("RecursiveCDKLM16")\ncase4RecCDKLM.runSim(8)')

h2, hu2, hv2 = case4CDKLM.sim.download()
h3, hu3, hv3 = case4RecCDKLM.sim.download()
h4, hu4, hv4 = case4KP07.sim.download()

cells = 400

print(np.max(hu2), "Max hu CDKLM")
print(np.max(hu3), "Max hu recCDKLM")
print(np.linalg.norm(h2-h3), "norm h-diff between CDKLM and rec CDKLM")
print(np.linalg.norm(h2[1:cells+1+3+1,1:cells+1+3+1]-h4), "norm h-diff between CDKLM and KP07")

print(case4CDKLM.sim.downloadGeoEqNorm(), "Geostrophic Equilibrium CDKLM")
print(case4CDKLM_rk3.sim.downloadGeoEqNorm(), "Geostrophic Equilibrium CDKLM")

print(np.sum(case4KP07.h0[2:cells+2, 2:cells+2]) - np.sum(case4KP07.h0[2:cells+2, 2:cells+2]), "Conservation h KP07")
print(np.sum(case4CDKLM.h0[3:cells+3, 3:cells+3]) - np.sum(case4CDKLM.h0[3:cells+3, 3:cells+3]), "Conservation h CDKLM")
print(np.sum(case4RecCDKLM.h0[3:cells+3, 3:cells+3]) - np.sum(case4RecCDKLM.h0[3:cells+3, 3:cells+3]), "Conservation h RecCDKLM")

get_ipython().run_cell_magic('time', '', 'case4CDKLMp = GeostrophicEquilibrium(cl_ctx, "4", newDt=0.002)\ncase4CDKLMp.initializeSimulator("CDKLM16")\niterations = 100\nKx = np.zeros(iterations)\nL_y = np.zeros(iterations)\nuxpvy = np.zeros(iterations)\nfor i in range(iterations):\n    case4CDKLMp.runSim(0.08, doPlot=False)\n    uxpvy[i], Kx[i], L_y[i] = case4CDKLMp.sim.downloadGeoEqNorm()\nfig = plt.figure()\nplt.plot(uxpvy, \'b\', label="ux+vy")\nplt.plot(Kx, \'r\', label="Kx")\nplt.plot(L_y, \'g\', label="Ly")\nplt.legend()')

## Example 5 from the CDKLM paper


f = 1.0 # s^-1   Coriolis parameter
g = 1.0 # m/s^2   Gravitational acceleration
R = 0.0 # m/s   Bottom friction coefficient
lengthX = 60.0 # m   domain length in x-direction
lengthY = 60.0 # m   domain length in y-direction

dt = 0.01 # s   Time increment
nx = 1000
ny = 1000


dx = lengthX / nx
dy = lengthY / ny

windStressParams = Common.WindStressParams(type=99)
boundaryConditions = Common.BoundaryConditions() # Wall boundaries

ghosts = [2,2,2,2]
validDomain = [-2, -2, 2, 2]
dataShape = (ny + ghosts[0]+ghosts[2],             nx + ghosts[1]+ghosts[3])
        
h0 = np.zeros(dataShape, dtype=np.float32, order='C');
u0 = np.zeros(dataShape, dtype=np.float32, order='C');
v0 = np.zeros(dataShape, dtype=np.float32, order='C');
Bi = np.zeros((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')
        #print (self.dataShape) 
        
for j in range(dataShape[0]):
    y = (j-ghosts[0]+0.5)*dy - lengthY/2.0
    #print(j-ghosts[0], y)
    for i in range(dataShape[1]):
        x = (i-ghosts[1]+0.5)*dx - lengthX/2.0
        #print(i-self.ghosts[1], x)

        radius = np.sqrt(2.5*x*x + 0.4*y*y)
        tanh_term = np.tanh(10.0*(radius - 1.0))
        h0[j,i] = 1.0 + 0.25*(1.0 - tanh_term)
        u0[j,i] =  (1.0/radius)*(      (1.0 - tanh_term*tanh_term)*y)
        v0[j,i] = -(1.0/radius)*( 6.25*(1.0 - tanh_term*tanh_term)*x)
#fig = plt.figure()
#singlePlotter = PlotHelper.SinglePlot(fig, x_coords, y_coords, v0, interpolation_type="None")

        
# Required for using plotting:
#Calculate radius from center of bump for plotting
x_center = lengthX/2.0
y_center = lengthY/2.0
y_coords, x_coords = np.mgrid[0:lengthY:dy, 0:lengthX:dx]
x_coords = np.subtract(x_coords, x_center)
y_coords = np.subtract(y_coords, y_center)
radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))

reload(CDKLM16)
sim = CDKLM16.CDKLM16(cl_ctx,       h0, u0, v0, Bi,       nx, ny, dx, dy, dt,       g, f, R,       wind_stress=windStressParams,       boundary_conditions=boundaryConditions,       reportGeostrophicEquilibrium=False)
            
fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                h0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]] - 1.0, 
                                u0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]], 
                                v0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]])
T = 100
Kx = np.zeros(T)
Ly = np.zeros(T)
uxpvy = np.zeros(T)

def animate(i):
    if (i>0):
        t = sim.step(0.5)
    else:
        t = 0.0
    h1, u1, v1 = sim.download()
    uxpvy[i], Kx[i], Ly[i] = sim.downloadGeoEqNorm()
    #print uxpvy[i], Kx[i], Ly[i]
    
    brighten = 1
    
    plotter.plot(h1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]] - 1.0, 
                 brighten*u1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]], 
                 brighten*v1[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]]);
    fig.suptitle("CDKLM16 Time = " + "{:04.0f}".format(t) + " s", fontsize=18)

    if (i%10 == 0):
        print "{:03.0f}".format(100*i / T) + " % => t=" + str(t) + "\tMax h: " + str(np.max(h1))
        fig.savefig(imgdir + "/{:010.0f}_cdklm16.png".format(t))
             
anim = animation.FuncAnimation(fig, animate, range(T), interval=100)
plt.close(anim._fig)
anim
    

## Plotting initial conditions
fig = plt.figure()
plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, 
                                h0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]] - 1.0, 
                                u0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]], 
                                v0[validDomain[2]:validDomain[0], validDomain[3]:validDomain[1]])
#h1, u1, v1 = sim.download()
#fig = plt.figure()
#singlePlotter = PlotHelper.SinglePlot(fig, x_coords, y_coords, h1)
#fig = plt.figure()
#singlePlotter = PlotHelper.SinglePlot(fig, x_coords, y_coords, v1)

## Check conservation of total mass

h3, hu3, hv3 = sim.download()
print("Initial water volume: ", sum(sum(h0)))
print("After simulation:     ", sum(sum(h3)))
print(imgdir)
print("min-max h:           ", np.min(h3), np.max(h3))
print("min-max hu:           ", np.min(hu3), np.max(hu3))
print("min-max hv:           ", np.min(hv3), np.max(hv3))

