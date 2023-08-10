import SimPEG as simpeg
from scipy.constants import mu_0
def omega(freq):
    """Change frequency to angular frequency, omega"""
    return 2.*np.pi*freq

get_ipython().magic('pylab inline')

np.sum(100*np.cumprod(np.ones(5)*1.6))

   

# M = Mesh.TensorMesh([[(100.,32)],[(100.,34)],[(100.,18)]], x0='CCC')
# M = simpeg.Mesh.TensorMesh([[(100,5,-1.5),(100.,5),(100,5,1.5)],[(100,5,-1.5),(100.,5),(100,5,1.5)],[(100,5,-1.5),(100.,10),(100,5,1.5)]], x0=['C','C','C'])
M = simpeg.Mesh.TensorMesh([[(1000,6,-1.5),(1000.,6),(1000,6,1.5)],[(1000,6,-1.5),(1000.,2),(1000,6,1.5)],[(1000,10,-1.3),(1000.,2),(1000,10,1.3)]], x0=['C','C','C'])# Setup the model

print M.vectorNz

# Setup the model
# conds = [1,1e-2]
# elev = 300
# sig = simpeg.Utils.ModelBuilder.defineBlock(M.gridCC,[-100000,-100000,-200],[100000,100000,0],conds)
# sig[M.gridCC[:,2]>elev] = 1e-8
# sig[M.gridCC[:,2]<-500] = 1e-1
# sig[M.gridCC[:,2]<-900] = 1e-2
elev=0
conds = [1,1e-2]
sig = np.ones(M.nC)*conds[0]
sig[M.gridCC[:,0]>0] = conds[1]
sig[M.gridCC[:,2]>elev] = 1e-8
# sigBG = np.zeros(M.nC) + conds[0]
# sigBG[M.gridCC[:,2]>0] = 1e-8
sigBG = np.ones(M.nC)*conds[0]
sigBG[M.gridCC[:,2]>elev] = 1e-8
colorbar(M.plotImage(log10(sig)))
colorbar(M.plotImage(log10(sigBG)))

# Get the mass matrix 
# The model
Msig = M.getEdgeInnerProduct(sig)
MsigBG = M.getEdgeInnerProduct(sigBG)
Mmu = M.getFaceInnerProduct(mu_0, invProp=True)

freq = 1.0
C = M.edgeCurl
A = C.T*Mmu*C - 1j*omega(freq)*Msig
ARH = -(C.T*Mmu*C - 1j*omega(freq)*MsigBG)

get_ipython().run_cell_magic('time', '', "# Solve the systems for each polarization\nsys.path.append('/media/gudni/ExtraDrive1/Codes/python/pymatsolver/')\nfrom pymatsolver import MumpsSolver\nAinv = MumpsSolver(A)")

# Need to solve x and y polarizations of the source.
from simpegMT.Utils import get1DEfields
# Get a 1d solution for a halfspace background
mesh1d = simpeg.Mesh.TensorMesh([M.hz],np.array([M.x0[2]]))
e0_1d = get1DEfields(mesh1d,M.r(sigBG,'CC','CC','M')[0,0,:],freq,sourceAmp=None).conj()
# Setup the primary field (p) for the x (east) polarization (_px)
ex_px = np.zeros((M.vnEx),dtype=complex)
ey_px = np.zeros(M.nEy,dtype=complex)
ez_px = np.zeros(M.nEz,dtype=complex)
# Assign the source to ex_x
for i in arange(M.vnEx[0]):
    for j in arange(M.vnEx[1]):
        ex_px[i,j,:] = -e0_1d
ep_px = np.r_[simpeg.Utils.mkvc(ex_px),ey_px,ez_px]
rhs_px = ARH.dot(ep_px)

# Setup y (north) polarization (_y)
ex_py = np.zeros(M.nEx, dtype='complex128')
ey_py = np.zeros((M.vnEy), dtype='complex128')
ez_py = np.zeros(M.nEz, dtype='complex128')
# Assign the source to ey_y
for i in arange(M.vnEy[0]):
    for j in arange(M.vnEy[1]):
        ey_py[i,j,:] = e0_1d  
        
ep_py = np.r_[ex_py,simpeg.Utils.mkvc(ey_py),ez_py]
rhs_py = ARH.dot(ep_py)

get_ipython().run_cell_magic('time', '', 'es_px = Ainv*rhs_px\nes_py = Ainv*rhs_py')

# Need to sum the ep and es to get the total field.
e_x = es_px #+ ep_px
e_y = es_py #+ ep_py

Meinv = M.getEdgeInnerProduct(np.ones_like(sig), invMat=True)

j_x = Meinv*Msig*e_x
j_y = Meinv*Msig*e_x

e_x.shape

e_x_CC = M.aveE2CCV*e_x
e_y_CC = M.aveE2CCV*e_y
j_x_CC = M.aveE2CCV*j_x
j_y_CC = M.aveE2CCV*j_y
# j_x_CC = Utils.sdiag(np.r_[sig, sig, sig])*e_x_CC

fig, ax = plt.subplots(1,2, figsize = (12, 5))
dat0 = M.plotSlice(np.log10(np.abs(e_x_CC)), vType='CCv', view='vec', streamOpts={'color': 'k'}, normal='Y', ax = ax[0])
cb0 = plt.colorbar(dat0[0], ax = ax[0])
dat1 = M.plotSlice(np.log10(np.abs(e_y_CC)), vType='CCv', view='vec', streamOpts={'color': 'k'}, normal='Y', ax = ax[1])
cb1 = plt.colorbar(dat1[0], ax = ax[1])

# Calculate the data
rx_x, rx_y = np.meshgrid(np.arange(-3000,3001,500),np.arange(-1000,1001,500))
rx_loc = np.hstack((simpeg.Utils.mkvc(rx_x,2),simpeg.Utils.mkvc(rx_y,2),elev+np.zeros((np.prod(rx_x.shape),1))))
# Get the projection matrices
Qex = M.getInterpolationMat(rx_loc,'Ex')
Qey = M.getInterpolationMat(rx_loc,'Ey')
Qez = M.getInterpolationMat(rx_loc,'Ez')
Qfx = M.getInterpolationMat(rx_loc,'Fx')
Qfy = M.getInterpolationMat(rx_loc,'Fy')
Qfz = M.getInterpolationMat(rx_loc,'Fz')

e_x_loc = np.hstack([simpeg.Utils.mkvc(Qex*e_x,2),simpeg.Utils.mkvc(Qey*e_x,2),simpeg.Utils.mkvc(Qez*e_x,2)])
e_y_loc = np.hstack([simpeg.Utils.mkvc(Qex*e_y,2),simpeg.Utils.mkvc(Qey*e_y,2),simpeg.Utils.mkvc(Qez*e_y,2)])
Ciw = -C/(1j*omega(freq)*mu_0)
h_x_loc = np.hstack([simpeg.Utils.mkvc(Qfx*Ciw*e_x,2),simpeg.Utils.mkvc(Qfy*Ciw*e_x,2),simpeg.Utils.mkvc(Qfz*Ciw*e_x,2)])
h_y_loc = np.hstack([simpeg.Utils.mkvc(Qfx*Ciw*e_y,2),simpeg.Utils.mkvc(Qfy*Ciw*e_y,2),simpeg.Utils.mkvc(Qfz*Ciw*e_y,2)])

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

# rx_loc

ind = np.where(np.sum(np.power(rx_loc - np.array([-3000,0,elev]),2),axis=1)< 5)
m
print appResPhs(freq,zyx[ind])
print appResPhs(freq,zxy[ind])



e0_1d = e0_1d.conj()
Qex = mesh1d.getInterpolationMat(np.array([elev]),'Ex')
Qfx = mesh1d.getInterpolationMat(np.array([elev]),'Fx')
h0_1dC = -(mesh1d.nodalGrad*e0_1d)/(1j*omega(freq)*mu_0)
h0_1d = mesh1d.getInterpolationMat(mesh1d.vectorNx,'Ex')*h0_1dC
indSur = np.where(mesh1d.vectorNx==elev)

print (Qfx*e0_1d),(Qex*h0_1dC)#e0_1d, h0_1d
print appResPhs(freq,(Qfx*e0_1d)/(Qex*h0_1dC).conj())

import simpegMT as simpegmt
sig1D = M.r(sig,'CC','CC','M')[0,0,:]
anaEd, anaEu, anaHd, anaHu = simpegmt.Utils.MT1Danalytic.getEHfields(mesh1d,sig1D,freq,mesh1d.vectorNx)
anaEtemp = anaEd+anaEu
anaHtemp = anaHd+anaHu
# Scale the solution
anaE = (anaEtemp/anaEtemp[-1])#.conj()
anaH = (anaHtemp/anaEtemp[-1])#.conj()

anaZ = anaE/anaH
indSur = np.where(mesh1d.vectorNx==elev)
print anaZ
print appResPhs(freq,anaZ[indSur])
print appResPhs(freq,-anaZ[indSur])

mesh1d.vectorNx





