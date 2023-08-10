from SimPEG import *
import simpegDCIP as DC
get_ipython().magic('pylab inline')

cs = 25.
hx = [(cs,7, -1.3),(cs,21),(cs,7, 1.3)]
hy = [(cs,7, -1.3),(cs,21),(cs,7, 1.3)]
hz = [(cs,7, -1.3),(cs,20)]

mesh = Mesh.TensorMesh([hx, hy, hz], 'CCN')

blk1 = Utils.ModelBuilder.getIndicesBlock(np.r_[-50, 75, -50], np.r_[75, -50, -150], mesh.gridCC)
sighalf = 1e-3
sigma = np.ones(mesh.nC)*sighalf
sigma[blk1] = 1e-1
sigmahomo = np.ones(mesh.nC)*sighalf

mesh.plotSlice(sigma, normal='X', grid=True)

xtemp = np.linspace(-150, 150, 21)
ytemp = np.linspace(-150, 150, 21)
xyz_rxM = Utils.ndgrid(xtemp-10., ytemp, np.r_[0.])
xyz_rxN = Utils.ndgrid(xtemp+10., ytemp, np.r_[0.])
# xyz_rxM = Utils.ndgrid(xtemp, ytemp, np.r_[0.])

plt.plot(xyz_rxP[:,0], xyz_rxP[:,1], 'k.')
plt.plot(xyz_rxM[:,0], xyz_rxM[:,1], 'r.')
plt.plot(xyz_rxN[:,0], xyz_rxN[:,1], 'g.')

fig, ax = plt.subplots(1,1, figsize = (5,5))
mesh.plotSlice(sigma, grid=True, ax = ax)
ax.plot(xyz_rxP[:,0],xyz_rxP[:,1], 'w.')
ax.plot(xyz_rxN[:,0],xyz_rxN[:,1], 'r.', ms = 3)

rx

rx = DC.RxDipole(xyz_rxM, xyz_rxN)
tx = DC.SrcDipole([rx], [-200, 0, -12.5],[+200, 0, -12.5])

txList = []
txList.append(DC.SrcDipole([rx], [-200, 0, -12.5],[+200, 0, -12.5]))
txList.append(DC.SrcDipole([rx], [-200, 0, -12.5],[+100, 0, -12.5]))

txList

# survey = DC.SurveyDC([tx])
survey = DC.SurveyDC(txList)

# survey.unpair()
problem = DC.ProblemDC_CC(mesh)
problem.pair(survey)

try:
    from pymatsolver import MumpsSolver
    problem.Solver = MumpsSolver
except Exception, e:
    problem.Solver = SolverLU

get_ipython().run_cell_magic('time', '', 'dataP = survey.dpred(sigmahomo)')

data = survey.dpred(sigma)

Data = (data/dataP).reshape((21,21,2), order='F')



plt.pcolor(Data[:,:,0].T)

u1 = problem.fields(sigma)
u2 = problem.fields(sigmahomo)

# Msig1 = Utils.sdiag(1./(mesh.aveF2CC.T*(1./sigma)))
# Msig2 = Utils.sdiag(1./(mesh.aveF2CC.T*(1./sigmahomo)))

j1 = Msig1*mesh.cellGrad*u1[tx, 'phi_sol']
j2 = Msig2*mesh.cellGrad*u2[tx, 'phi_sol']

# us = u1-u2
# js = j1-j2

mesh.plotSlice(mesh.aveF2CCV*j1, vType='CCv', normal='Y', view='vec', streamOpts={"density":3, "color":'w'})
xlim(-300, 300)
ylim(-300, 0)

mesh.plotSlice(mesh.aveF2CCV*js, vType='CCv', normal='Y', view='vec', streamOpts={"density":3, "color":'w'})
xlim(-300, 300)
ylim(-300, 0)

a = np.random.randn(3)

print (a.reshape([1,-1])).repeat(3, axis = 0)
print (a.reshape([1,-1])).repeat(3, axis = 0).sum(axis=1)

def DChalf(txlocP, txlocN, rxloc, sigma, I=1.):
    rp = (txlocP.reshape([1,-1])).repeat(rxloc.shape[0], axis = 0)
    rn = (txlocN.reshape([1,-1])).repeat(rxloc.shape[0], axis = 0)
    rP = np.sqrt(((rxloc-rp)**2).sum(axis=1))
    rN = np.sqrt(((rxloc-rn)**2).sum(axis=1))
    return I/(sigma*2.*np.pi)*(1/rP-1/rN)

data_analP = DChalf(np.r_[-200, 0, 0.],np.r_[+200, 0, 0.], xyz_rxP, sighalf)
data_analN = DChalf(np.r_[-200, 0, 0.],np.r_[+200, 0, 0.], xyz_rxN, sighalf)
data_anal = data_analP-data_analN

Data_anal = data_anal.reshape((21, 21), order = 'F')
Data = data.reshape((21, 21), order = 'F')
X = xyz_rxM[:,0].reshape((21, 21), order = 'F')
Y = xyz_rxM[:,1].reshape((21, 21), order = 'F')

fig, ax = plt.subplots(1,2, figsize = (12, 5))
vmin = np.r_[data, data_anal].min()
vmax = np.r_[data, data_anal].max()
dat0 = ax[0].contourf(X, Y, Data, 60, vmin = vmin, vmax = vmax)
dat1 = ax[1].contourf(X, Y, Data_anal, 60, vmin = vmin, vmax = vmax)
cb0 = plt.colorbar(dat1, orientation = 'horizontal', ax = ax[0])
cb1 = plt.colorbar(dat1, orientation = 'horizontal', ax = ax[1])













