def rosen(x):

    f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    c = []

    return f, c

from pyoptsparse import NSGA2

# choose optimizer and define options
optimizer = NSGA2()
optimizer.setOption('maxGen', 200)
optimizer.setOption('PopSize', 40)
optimizer.setOption('pMut_real', 0.01)
optimizer.setOption('pCross_real', 1.0)

from pyoptwrapper import optimize

x0 = [4.0, 4.0]
lb = [-5.0, -5.0]
ub = [5.0, 5.0]


xopt, fopt, info = optimize(rosen, x0, lb, ub, optimizer)
print 'results:', xopt, fopt, info

from pyoptsparse import SNOPT

optimizer = SNOPT()

xopt, fopt, info = optimize(rosen, x0, lb, ub, optimizer)
print 'results:', xopt, fopt, info



