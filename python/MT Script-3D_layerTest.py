import SimPEG as simpeg
from scipy.constants import mu_0
def omega(freq):
    """Change frequency to angular frequency, omega"""
    return 2.*np.pi*freq

get_ipython().magic('pylab inline')

np.sum(100*np.cumprod(np.ones(5)*1.6))

   

# M = Mesh.TensorMesh([[(100.,32)],[(100.,34)],[(100.,18)]], x0='CCC')
M = simpeg.Mesh.TensorMesh([[(100,5,-1.5),(100.,10),(100,5,1.5)],[(100,5,-1.5),(100.,10),(100,5,1.5)],[(100,5,1.6),(100.,10),(100,3,2)]], x0=['C','C',-3529.5360])

print M.vectorNz

# Setup the model
conds = [1e-2,1]
sig = simpeg.Utils.ModelBuilder.defineBlock(M.gridCC,[-200,-200,-400],[200,200,-200],conds)
sig[M.gridCC[:,2]>0] = 1e-8
sig[M.gridCC[:,2]<-600] = 1e-1
sigBG = np.zeros(M.nC) + conds[0]
sigBG[M.gridCC[:,2]>0] = 1e-8
colorbar(M.plotImage(log10(sig)))

# Receivers 
# 3D impedance at the surface (elevation 0)
rxList = []
for rxType in ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi',]:
    rxList.append(simpegmt.SurveyMT.RxMT(simpeg.mkvc(np.array([0.0]),2).T,rxType))
# Source list
srcList =[]
tD = False
if tD:
    for freq in freqs:
        srcList.append(simpegmt.SurveyMT.srcMT_polxy_1DhomotD(rxList,freq))
else:
    for freq in freqs:
        srcList.append(simpegmt.SurveyMT.srcMT_polxy_1Dprimary(rxList,freq,sigma_0))
# Make the survey
survey = simpegmt.SurveyMT.SurveyMT(srcList)
survey.mtrue = m_true
# Set the problem
problem = simpegmt.ProblemMT1D.eForm_psField(m1d,mapping=mappingExpAct)
from pymatsolver import MumpsSolver
problem.solver = MumpsSolver
problem.pair(survey)

# Make the data 

# Setup y (north) polarization (_y)
ex_y = np.zeros(M.nEx, dtype='complex128')
ey_y = np.zeros((M.vnEy), dtype='complex128')
ez_y = np.zeros(M.nEz, dtype='complex128')
# Assign the source to ex_x
for i in arange(M.vnEy[0]):
    for j in arange(M.vnEy[1]):
        ey_y[i,j,:] = e0_1d        
# eBG_y = np.vstack((ex_y,Utils.mkvc(M.r(ey_y,'Ey','Ey','V'),2),ez_y))
# rhs_y = ABG.dot(eBG_y)

eBG_y = np.r_[ex_y,simpeg.Utils.mkvc(ey_y),ez_y]
rhs_y = ABG.dot(eBG_y)

get_ipython().run_cell_magic('time', '', '# Solve the systems for each polarization\nAinv = simpeg.SolverLU(A)')

get_ipython().run_cell_magic('time', '', 'e_x = Ainv*rhs_x\ne_y = Ainv*rhs_y')

Meinv = M.getEdgeInnerProduct(np.ones_like(sig), invMat=True)

j_x = Meinv*Msig*e_x
j_y = Meinv*Msig*e_x

e_x_CC = M.aveE2CCV*e_x
e_y_CC = M.aveE2CCV*e_y
j_x_CC = M.aveE2CCV*j_x
j_y_CC = M.aveE2CCV*j_y
# j_x_CC = Utils.sdiag(np.r_[sig, sig, sig])*e_x_CC

fig, ax = plt.subplots(1,2, figsize = (12, 5))
dat0 = M.plotSlice(abs(e_x_CC), vType='CCv', view='vec', streamOpts={'color': 'k'}, normal='X', ax = ax[0])
cb0 = plt.colorbar(dat0[0], ax = ax[0])
dat1 = M.plotSlice(abs(j_x_CC), vType='CCv', view='vec', streamOpts={'color': 'k'}, normal='X', ax = ax[1])
cb1 = plt.colorbar(dat1[0], ax = ax[1])

# Calculate the data
rx_x, rx_y = np.meshgrid(np.arange(-500,501,50),np.arange(-500,501,50))
rx_loc = np.hstack((simpeg.Utils.mkvc(rx_x,2),simpeg.Utils.mkvc(rx_y,2),np.zeros((np.prod(rx_x.shape),1))))
# Get the projection matrices
Qex = M.getInterpolationMat(rx_loc,'Ex')
Qey = M.getInterpolationMat(rx_loc,'Ey')
Qez = M.getInterpolationMat(rx_loc,'Ez')
Qfx = M.getInterpolationMat(rx_loc,'Fx')
Qfy = M.getInterpolationMat(rx_loc,'Fy')
Qfz = M.getInterpolationMat(rx_loc,'Fz')

e_x_loc = np.hstack([simpeg.Utils.mkvc(Qex*e_x,2),simpeg.Utils.mkvc(Qey*e_x,2),simpeg.Utils.mkvc(Qez*e_x,2)])
e_y_loc = np.hstack([simpeg.Utils.mkvc(Qex*e_y,2),simpeg.Utils.mkvc(Qey*e_y,2),simpeg.Utils.mkvc(Qez*e_y,2)])
h_x_loc = np.hstack([simpeg.Utils.mkvc(Qfx*C*e_x,2),simpeg.Utils.mkvc(Qfy*C*e_x,2),simpeg.Utils.mkvc(Qfz*C*e_x,2)])
h_y_loc = np.hstack([simpeg.Utils.mkvc(Qfx*C*e_y,2),simpeg.Utils.mkvc(Qfy*C*e_y,2),simpeg.Utils.mkvc(Qfz*C*e_y,2)])

# Make a combined matrix
dt = np.dtype([('ex1',complex),('ey1',complex),('ez1',complex),('hx1',complex),('hy1',complex),('hz1',complex),('ex2',complex),('ey2',complex),('ez2',complex),('hx2',complex),('hy2',complex),('hz2',complex)])
combMat = np.empty((len(e_x_loc)),dtype=dt)
combMat['ex1'] = e_x_loc[:,0]
combMat['ey1'] = e_x_loc[:,1]
combMat['ez1'] = e_x_loc[:,2]
combMat['ex2'] = e_y_loc[:,0]
combMat['ey2'] = e_y_loc[:,1]
combMat['ez2'] = e_y_loc[:,2]
combMat['hx1'] = h_x_loc[:,0]
combMat['hy1'] = h_x_loc[:,1]
combMat['hz1'] = h_x_loc[:,2]
combMat['hx2'] = h_y_loc[:,0]
combMat['hy2'] = h_y_loc[:,1]
combMat['hz2'] = h_y_loc[:,2]

def calculateImpedance(fieldsData):
    ''' 
    Function that calculates MT impedance data from a rec array with E and H field data from both polarizations
    '''
    zxx = (fieldsData['ex1']*fieldsData['hy2'] - fieldsData['ex2']*fieldsData['hy1'])/(fieldsData['hx1']*fieldsData['hy2'] - fieldsData['hx2']*fieldsData['hy1'])
    zxy = (-fieldsData['ex1']*fieldsData['hx2'] + fieldsData['ex2']*fieldsData['hx1'])/(fieldsData['hx1']*fieldsData['hy2'] - fieldsData['hx2']*fieldsData['hy1'])
    zyx = (fieldsData['ey1']*fieldsData['hy2'] - fieldsData['ey2']*fieldsData['hy1'])/(fieldsData['hx1']*fieldsData['hy2'] - fieldsData['hx2']*fieldsData['hy1'])
    zyy = (-fieldsData['ey1']*fieldsData['hx2'] + fieldsData['ey2']*fieldsData['hx1'])/(fieldsData['hx1']*fieldsData['hy2'] - fieldsData['hx2']*fieldsData['hy1'])
    return zxx, zxy, zyx, zyy

zxx, zxy, zyx, zyy = calculateImpedance(combMat)

zxy

zyx

zxy + zyx



