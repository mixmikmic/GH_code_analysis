from timml import *
from pylab import *
get_ipython().magic('matplotlib notebook')

ml = Model(k=[10], zb=[0], zt=[10])

rf = Constant(ml, xr=-1000, yr=0, head=41)

uf = Uflow(ml, grad=0.001, angle=0)

ml.solve()

timcontour(ml, -1000, 100, 50, -500, 500, 50, levels=[39, 42, 0.1], labels=True, size=(6,6))

w = Well(ml, xw=-400, yw=0, Qw=50, rw=0.2)

ml.solve()
timcontour(ml, -1000, 100, 50, -500, 500, 50, levels=[39, 42, 0.1], labels=True, size=(6,6))
timtracelines(ml, xlist=10 * [-800], ylist=linspace(-500, 500, 10), zlist=10*[0], step=20)

ml = Model(k=[10], zb=[0], zt=[10])
rf = Constant(ml, xr=-1000, yr=0, head=41)
uf = Uflow(ml, grad=0.001, angle=0)
w = Well(ml, xw=-400, yw=0, Qw=200, rw=0.2)
yls = linspace(-800, 800, 21)
for i in range(20):
    HeadLineSink(ml, 0, yls[i], 0, yls[i+1], 40)
ml.solve()
timcontour(ml, -1000, 100, 50, -500, 500, 50, levels=[39, 42, 0.1], labels=True, size=(6,6))
timtracelines(ml, xlist=10 * [-800], ylist=linspace(-500, 500, 10), zlist=10*[0], step=20)
timtracelines(ml, xlist=5 * [-0.01], ylist=linspace(-150, 150, 5), zlist=5*[0], step=20, color='r')

timcontour(ml, -1000, 100, 50, -500, 500, 50, levels=[39, 42, 0.1], labels=True, size=(6,6))
capturezone(ml, w=w, N=20, z=0, tmax=5 * 365.25)

