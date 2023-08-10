from timml import *
from pylab import *
get_ipython().magic('matplotlib notebook')

ml = Model(k=[10, 20, 5],
           zb=[-20, -80, -140], 
           zt=[0, -40, -90], 
           c=[4000, 10000] )
w = Well(ml, xw=0, yw=0, Qw=10000, rw=0.2, layers=1 )
Constant(ml, xr=10000, yr=0, head=20, layer=0 )
Uflow(ml, grad=0.002, angle=0 )
ml.solve()

print 'The head at the well is:'
print ml.headVector(w.xw, w.yw)

timcontour(ml, -3000, 3000, 50, -3000, 3000, 50, layers=3, levels=10, size=(6,6))

timcontour(ml, -3000, 3000, 50, -3000, 3000,50, layers=0, levels=20, xsec=1, labels=1, size=(6,6))
timtracelines(ml, -2000 * ones(3), -1000 * ones(3), [-120, -60, -10], 50, xsec=1)
timtracelines(ml, 0 * ones(3), 1000 * ones(3), [-120, -50, -10], 50, xsec=1)

