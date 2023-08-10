from gpgLabs.Mag import *
from SimPEG import PF, Utils, Mesh
get_ipython().magic('matplotlib inline')

#Input parameters
inp_dir = '../assets/Mag/data/'
fileName = 'Lab1_Wednesday_TA.csv'

data = np.genfromtxt(inp_dir + fileName, skip_header=1, delimiter=',')
xyzd = np.c_[data[:,0], np.zeros((data.shape[0],2)), data[:,1]]
B = np.r_[60308, 83.8, 25.4]

survey = Mag.createMagSurvey(xyzd, B)
# View the data and chose a profile
# Define the parametric model interactively
model = Simulator.ViewPrism(survey)
display(model)

Q = Simulator.fitline(model,survey)
display(Q)



