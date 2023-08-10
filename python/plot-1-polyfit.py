get_ipython().magic('matplotlib inline')

import pygimli as pg
import numpy as np
import matplotlib.pyplot as plt

print(pg.__version__)

class FunctionModelling(pg.ModellingBase):
    def __init__(self, nc, xvec, verbose=False):
        pg.ModellingBase.__init__(self, verbose)
        self.x_ = xvec
        self.nc_ = nc
        self.regionManager().setParameterCount(nc)

    def response(self, par):
        y = pg.RVector(len(self.x_), par[0])

        for i in range(1, self.nc_):
            y += pg.pow(self.x_, i) * par[i]

        return y

    def startModel(self):
        return pg.RVector(self.nc_, 0.5)

x = np.arange(0., 10., 1)
y = 1.1 + 2.1 * x

nP = 3

# two coefficients and x-vector (first data column)
fop = FunctionModelling(nP, x)

# initialize inversion with data and forward operator and set options
inv = pg.RInversion(y, fop)

# constant absolute error of 0.01 is 1% (not necessary, only for chi^2)
inv.setAbsoluteError(0.01)

# the problem is well-posed and does not need regularization
inv.setLambda(0)

# actual inversion run yielding coefficient model
coeff = inv.run()

print(coeff)

plt.plot(x, y, 'rx', x, inv.response(), 'b-')

plt.show()

