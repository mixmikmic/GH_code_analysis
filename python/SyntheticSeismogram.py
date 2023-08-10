# Import the necessary packages
get_ipython().magic('pylab inline')
from syntheticSeismogramImport import *  

# Create Interactive Plot for Logs
logs = interact(plotLogsInteract,d2=(0.,100.,5),d3=(100.,200.,5),rho1=(0.,5000.,50.),rho2=(1000.,5000.,50.),rho3=(1000.,5000.,50.),v1=(300.,4000.,50.),v2=(300.,4000.,50.),v3=(300.,4000.,50.))

# Create depth-time interactive plot
interact(plotTimeDepthInteract,d2=(0.,100.,5),d3=(100.,200.,5),v1=(300.,4000.,50.),v2=(300.,4000.,50.),v3=(300.,4000.,50.))

# Interactive seismogram plot for a fixed geologic model
interact(plotSeismogramInteractFixMod,wavf=(5.,100.,5.),wavA=(-2.,2.,0.25))

interact(plotSeismogramInteract,d2=(0.,150.,1),d3=(50.,200.,1),rho1=(2000.,5000.,50.),rho2=(2000.,5000.,50.),rho3=(2000.,5000.,50.),v1=(300.,4000.,50.),v2=(300.,4000.,50.),v3=(300.,4000.,50.),wavf=(5.,100.,2.5),wavA=(-0.5,1.,0.25),addNoise=False,usingT=True) 



