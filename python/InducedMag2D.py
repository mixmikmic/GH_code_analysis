from gpgLabs.Mag import *
from SimPEG import PF, Utils, Mesh
get_ipython().magic('matplotlib inline')

#Input parameters
inp_dir = '../assets/Mag/data/'
fileName = 'DO27_TMI.dat'

xyzd = np.genfromtxt(inp_dir + fileName, skip_header=3)
B = np.r_[60308, 83.8, 25.4]

survey = Mag.createMagSurvey(xyzd, B)
# View the data and chose a profile
param = Simulator.ViewMagSurvey2D(survey)
display(param)

# Define the parametric model interactively
model = Simulator.ViewPrism(param.result)
display(model)

plotwidget = Simulator.PFSimulator(model, param)
display(plotwidget)





