#Lets have matplotlib "inline"
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

#Import packages we need
import numpy as np
from matplotlib import animation, rc
from matplotlib import pyplot as plt

import os
import pyopencl
import datetime
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../')))

# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile

#Set large figure sizes
rc('figure', figsize=(16.0, 12.0))
rc('animation', html='html5')

#Finally, import our simulator
from SWESimulators import FBL, CTCS, LxF, KP07, CDKLM16, RecursiveCDKLM16, PlotHelper, Common
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

"""
Class that defines the domain, initial conditions and boundary conditions used in the benchmark cases presented
by Roed. 
"""
class BenchmarkParameters:
    
    def __init__ (self, cl_ctx, case):
        
        self.cl_ctx = cl_ctx
        
        self.f = 1.2e-4 # s^-1   Coriolis parameter
        self.rho = 1025.0 # kg/m^3   Density of sea water
        self.g = 9.81 # m/s^2   Gravitational acceleration
        self.R = 2.4e-3 # m/s   Bottom friction coefficient
        self.H0 = 50.0   # m   Equilibrium depth on shelf
        self.H1 = 2500.0 # m   Equilibrium depth offshelf
        self.Lx = 1000.0e3 # m   Alongshore length of domain
        self.Ly = 500.0e3  # m   Width of domain
        self.Ls = 100.0e3  # m   Shelf width
        self.dx = 20.0e3   # m   Alongshore spatial increment
        self.dy = 20.0e3   # m   Cross-shore spatial increment
        self.dt = 90.0   # s   Time increment
        
        # Parameters not defined by Roeed:
        self.A = 1.0   # Eddy viscosity coefficient (O(dx))
        
        # When to stop the simulator:
        self.T = 96.0*60*60  # s   The plots we will compare with is taken from this time
        self.numTimeSteps = self.T/self.dt
        print("Total number of timesteps required:", self.numTimeSteps)
        
        self.nx = int(self.Lx) / int(self.dx)
        self.ny = int(self.Ly) / int(self.dy)
        
        alpha = 5.0e-6 # 1/m   Offshore e-folding length (=1/(10*dx))
        tau0 = 0.1 # Pa   Amplitude of wind stress
        tau1 = 3.0 # Pa   Maximum wind stress moving cyclone
        Rc = 200.0e3 # m   Distance to maximum wind stress from center of cyclone (= 10*dx)
        uC = 15.0 # m/s   Translation speed of cyclone 
        
        assert (len(case) == 2), "Invalid case specification"
        self.windtype = int(case[0])
        self.case = case[1]
        
        self.windStressParams = None
        if (self.windtype == 1):
            self.windStressParams = Common.WindStressParams(type=self.windtype-1,                      tau0=tau0, rho=self.rho, alpha=alpha, Rc=Rc)
        
        self.eta0 = None
        self.u0 = None
        self.v0 = None
        self.Hi = None
        self.sim = None
        self.boundaryConditions = None
        self.scheme = None
        
        self.ghosts = None
        self.validDomain = None
        
        # Required for using plotting:
        #Calculate radius from center of bump for plotting
        x_center = self.Lx/2.0
        y_center = self.Ly/2.0
        self.y_coords, self.x_coords = np.mgrid[0:self.Ly:self.dy, 0:self.Lx:self.dx]
        self.x_coords = np.subtract(self.x_coords, x_center)
        self.y_coords = np.subtract(self.y_coords, y_center)
        self.radius = np.sqrt(np.multiply(self.x_coords, self.x_coords) + np.multiply(self.y_coords, self.y_coords))
        
    def initializeSimulator(self, scheme):
        self.scheme = scheme
        assert  (scheme == "FBL" or scheme == "CTCS" or scheme == "CDKLM16" or scheme == "KP07"),            "Currently only valid for FBL, CTCS, CDKLM16 and KP07 :)"
        
        if (scheme == "FBL"):
            # Setting boundary conditions
            self.ghosts = [0,0,0,0]
            self.validDomain = [None ,None, 0, 0]
            if (self.case == "a"):
                self.boundaryConditions = Common.BoundaryConditions()
            else:
                assert(False), "Open boundary conditions not implemented"

            ghosts = self.ghosts
            self.h0 = np.ones((self.ny+ghosts[0], self.nx+ghosts[1]), dtype=np.float32) * self.H0;
            self.eta0 = np.zeros((self.ny+ghosts[0], self.nx+ghosts[1]), dtype=np.float32);
            self.u0 = np.zeros((self.ny+ghosts[0], self.nx+1), dtype=np.float32);
            self.v0 = np.zeros((self.ny+1, self.nx+ghosts[1]), dtype=np.float32);

            reload(FBL)
            self.sim = FBL.FBL(self.cl_ctx,                   self.h0, self.eta0, self.u0, self.v0,                   self.nx, self.ny,                   self.dx, self.dy, self.dt,                   self.g, self.f, self.R,                   wind_stress=self.windStressParams,                   boundary_conditions=self.boundaryConditions)
        
        elif scheme == "CTCS":
            # Setting boundary conditions
            self.ghosts = [1,1,1,1]
            self.validDomain = [-1, -1, 1, 1]
            if (self.case == "a"):
                self.boundaryConditions = Common.BoundaryConditions()
            else:
                assert(False), "Open boundary conditions not implemented"

            self.h0 = np.ones((self.ny+2, self.nx+2), dtype=np.float32) * self.H0;
            self.eta0 = np.zeros((self.ny+2, self.nx+2), dtype=np.float32);
            self.u0 = np.zeros((self.ny+2, self.nx+1+2), dtype=np.float32);
            self.v0 = np.zeros((self.ny+1+2, self.nx+2), dtype=np.float32);

            reload(CTCS)
            self.sim = CTCS.CTCS(self.cl_ctx,                   self.h0, self.eta0, self.u0, self.v0,                   self.nx, self.ny,                   self.dx, self.dy, self.dt,                   self.g, self.f, self.R, self.A,                    wind_stress=self.windStressParams,                   boundary_conditions=self.boundaryConditions)
            
        elif scheme == "CDKLM16":
            # Setting boundary conditions
            self.ghosts = [2,2,2,2]
            self.validDomain = [-2, -2, 2, 2]
            if (self.case == "a"):
                self.boundaryConditions = Common.BoundaryConditions()
            elif (self.case == "b"):
                self.boundaryConditions = Common.BoundaryConditions(1,2,1,2)
            else:
                assert(False), "Open boundary conditions not implemented"

            dataShape = (self.ny + self.ghosts[0]+self.ghosts[2],                         self.nx + self.ghosts[1]+self.ghosts[3])
            waterHeight = self.H0
            self.eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
            self.u0 = np.zeros(dataShape, dtype=np.float32, order='C');
            self.v0 = np.zeros(dataShape, dtype=np.float32, order='C');
            self.Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * waterHeight;
            
            
            reload(CDKLM16)
            self.sim = CDKLM16.CDKLM16(self.cl_ctx,                   self.eta0, self.u0, self.v0, self.Hi,                   self.nx, self.ny,                   self.dx, self.dy, self.dt,                   self.g, self.f, self.R,                   wind_stress=self.windStressParams,                   boundary_conditions=self.boundaryConditions)

        elif scheme == "KP07":
            # Setting boundary conditions
            self.ghosts = [2,2,2,2]
            self.validDomain = [-2, -2, 2, 2]
            if (self.case == "a"):
                self.boundaryConditions = Common.BoundaryConditions()
            else:
                assert(False), "Open boundary conditions not implemented"

            dataShape = (self.ny + self.ghosts[0]+self.ghosts[2],                         self.nx + self.ghosts[1]+self.ghosts[3])
            waterHeight = self.H0
            self.eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
            self.u0 = np.zeros(dataShape, dtype=np.float32, order='C');
            self.v0 = np.zeros(dataShape, dtype=np.float32, order='C');
            self.Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * waterHeight;

            
            reload(KP07)
            self.sim = KP07.KP07(self.cl_ctx,                   self.eta0, self.Hi, self.u0, self.v0,                   self.nx, self.ny,                   self.dx, self.dy, self.dt,                   self.g, self.f, self.R,                   wind_stress=self.windStressParams,                   boundary_conditions=self.boundaryConditions)
    
    
    def runSim(self):
        assert (self.sim is not None), "Simulator not initiated."
        
        self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        if (self.scheme == "CDKLM16" or self.scheme == "KP07" ):
            print("CDKLM16/KP07 and subtracting waterheigth")
            eta1 = eta1 - self.H0
        fig = plt.figure()
        plotter = PlotHelper.PlotHelper(fig, self.x_coords, self.y_coords, self.radius,                 eta1[self.validDomain[2]:self.validDomain[0], self.validDomain[3]:self.validDomain[1]],                 u1[self.validDomain[2]:self.validDomain[0], self.validDomain[3]:self.validDomain[1]],                  v1[self.validDomain[2]:self.validDomain[0], self.validDomain[3]:self.validDomain[1]]);
    
        print("results for case " + str(self.windtype) + self.case + " from simulator " + self.scheme)
        

get_ipython().run_cell_magic('time', '', 'case1a = BenchmarkParameters(cl_ctx, "1a")\nprint(case1a.windtype)\nprint(case1a.case)\ncase1a.initializeSimulator("FBL")\ncase1a.runSim()\n\nprint(case1a.Lx)\nprint(case1a.nx)\nprint(case1a.dx)')

get_ipython().run_cell_magic('time', '', 'case1aCTCS = BenchmarkParameters(cl_ctx, "1a")\ncase1aCTCS.initializeSimulator("CTCS")\ncase1aCTCS.runSim()')

get_ipython().run_cell_magic('time', '', 'case1aCDKLM16 = BenchmarkParameters(cl_ctx, "1a")\ncase1aCDKLM16.initializeSimulator("CDKLM16")\ncase1aCDKLM16.runSim()')

get_ipython().run_cell_magic('time', '', 'case1aKP07 = BenchmarkParameters(cl_ctx, "1a")\ncase1aKP07.initializeSimulator("KP07")\ncase1aKP07.runSim()')

get_ipython().run_cell_magic('time', '', 'case1aCDKLM16_1b = BenchmarkParameters(cl_ctx, "1b")\ncase1aCDKLM16_1b.initializeSimulator("CDKLM16")\ncase1aCDKLM16_1b.runSim()')



