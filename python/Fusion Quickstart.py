n = 5

A = [ [12., 20.,  0., 25., 15.],      [ 2.,  8., 16.,  0.,  0.],      [20., 20., 20., 20., 20.]]
     
b = [288.,192.,384.]

lx= [0. for i in range(n)]
ux= [1000. for i in range(n)]

from mosek.fusion import *

M = Model()

x = M.variable('x', n, Domain.inRange(lx,ux))
t = M.variable('t', 1, Domain.unbounded())

M.constraint(Var.vstack(t, x), Domain.inQCone())

le = M.constraint( Expr.mul(A,x), Domain.equalsTo(b))

M.objective(ObjectiveSense.Minimize, t)

M.solve()

print 'primal solution status = ',M.getPrimalSolutionStatus()
print 'primal solution        = \n', x.level()

print 'dual solution status = ', M.getDualSolutionStatus()
print 'Ax-b dual mult.      = ', le.dual()



