get_ipython().magic('matplotlib inline')
import os
import math
import IPython
import numexpr
import time as t
import numpy as np
import pandas as pd
from sys import version
import matplotlib.pyplot as plt
import matplotlib as mpl 
#os.chdir('C:\\Users\\Suso')
print '='*100
print 'Python version:     ' + version
print 'Numpy version:      ' + np.__version__
print 'Pandas version:     ' + pd.__version__
print 'Matplotlib ver:     ' + mpl.__version__
print 'IPython version:    ' + IPython.__version__
# numexpr.print_versions()
direct = get_ipython().magic('pwd')
print 'Working directory:  ' + direct
print '='*100
now = t.asctime()
print 'Today is ' + now + ' ... AND WE ARE READY TO GO HOME!! '
print '='*100

from peak.events import trellis

class TempConverter(trellis.Component):
     F = trellis.maintain(
         lambda self: self.C * 1.8 + 32,
         initially = 32
     )
     C = trellis.maintain(
         lambda self: (self.F - 32)/1.8,
         initially = 0
     )
     @trellis.perform
     def show_values(self):
         print "Celsius......", self.C
         print "Fahrenheit...", self.F


tc = TempConverter(C=100)

tc.F = 32

tc.C = 40

tc.C = 40

class Rectangle(trellis.Component):
    
    trellis.attrs(
        bottom = 0,
        width = 20,
        left = 0,
        height = 30
    )

    trellis.compute.attrs(
        top = lambda self: self.bottom + self.height,
        right  = lambda self: self.left + self.width,
    )

    @trellis.perform
    def show(self):
        print self

    def __repr__(self):
        return "Rectangle"+repr(
            ((self.left,self.bottom), 
             (self.width,self.height),
             (self.right,self.top))
        )

r1 = Rectangle()

r2 = Rectangle(width=40, height=10)

r2.width = 17

r2.left = 25

from peak.events import trellis
class NoiseFilter(trellis.Component):
    trellis.attrs( value = 0, threshhold = 5, )
    
    @trellis.maintain(initially =0)
    def filtered(self):
        if abs(self.value - self.filtered)> self.threshhold:
            return self.value
        return self.filtered

nf = NoiseFilter()
nf.filtered

nf.value = 1
nf.filtered

nf.value = 10
nf.filtered

nf.value = 7
nf.filtered

import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
clf = linear_model.LinearRegression()
data_x = [[0,0],[1,1],[2,2]]
data_y = [0,1,2]
clf.fit(data_x, data_y)

print "Coefficients : ", clf.coef_
print "Residual sum of squares: %.2f" % np.mean((clf.predict(data_x) - data_y)**2)
print data_x
len(data_x)
#plt.scatter(data_x, data_y, color = 'black')


import pylab as pl
import numpy as np
from sklearn.linear_model import LinearRegression

linear_reg = linear_model.LinearRegression()
data_x = [[0,0],[1,1],[2,2],[3,2]]
data_y = [0,1,2,3]
linear_reg.fit(data_x,data_y)

print type(data_x)

coeff = linear_reg.fit(data_x, data_y)

print coeff.coef_[0]
print coeff.coef_[1]

import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from peak.events import trellis

linear_reg = linear_model.LinearRegression()
data_x = [[0,0],[1,1],[2,2]]
data_y = [0,1,2]
linear_reg.fit(data_x,data_y)
print type(data_x)

coeff = linear_reg.fit(data_x, data_y)

print "Coefficients : ", coeff.coef_

class ReactRegression(trellis.Component):
    
    trellis.attrs(
        x = len(data_x) ,
        y = len(data_y) ,
        a1 = coeff.coef_[0] ,
        a2 = coeff.coef_[1] ,
    )
    
    @trellis.maintain (initially = len(data_x))
    def reset_fit(self):
        if (self.x - self.reset_fit) > 0:
            reset_fit = 1 
            return reset_fit
        return self.reset_fit

            # coeff = linear_reg.fit(data_x, data_y)
           # a1 = coeff.coef_[0]
           # a2 = coeff.coef_[1]
            
    @trellis.perform
    def show_values(self):
        print "a1......", self.a1
        print "a2......", self.a2
        
coeff2 = ReactRegression()
print coeff2.x

print (coeff2.x - coeff2.reset_fit)
print coeff2.reset_fit
data_x = [[0,0],[1,1],[2,2],[3,2]]
data_y = [0,1,2,3]
coeff2 = ReactRegression()

import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import time 
from sklearn.linear_model import LinearRegression

from peak.events import trellis

linear_reg2 = linear_model.LinearRegression()
data_x = [[0,0],[1,1],[2,2]]
data_y = [0,1,2]

type(data_x)
coeff = linear_reg2.fit(data_x, data_y)
print coeff

class ReactRegression(trellis.Component):
    coef_lreg = trellis.maintain(coeff)
    
@trellis.perform
def show(self):
    print self
    
def __repr__(self):
    return "Coefficients : " + repr(self.coef_)
    
    
reg1 = ReactRegression()

import time
get_ipython().magic('pinfo time.sleep')

reg1 = ReactRegression()

import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from peak.events import trellis

linear_reg = linear_model.LinearRegression()

print linear_reg

class ReactRegression(trellis.Component):
    coef_lreg = trellis.maintain(
        linear_reg.fit([[0,0],[1,1],[2,2]],[0,1,2])
    )
    
@trellis.perform
def show_coef(self):
    print "Coefficients : " + self.coef_
            
Reg1 = ReactRegression()
#[[0,0],[1,1],[2,2]],[0,1,2]


import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

digits = datasets.load_digits()

X, y = digits.data, digits.target

print X.shape

print y.shape

print y.size

get_ipython().magic('pinfo datasets.load_digits')

get_ipython().magic('pylab inline')
import numpy as np
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

digits = datasets.load_digits()

X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)

# classify small against large digits
y = (y > 4).astype(np.int)


# Set regularization parameter
for i, C in enumerate(10. ** np.arange(1, 4)):
    # turn down tolerance for short training time
    clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
    clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
    clf_l1_LR.fit(X, y)
    clf_l2_LR.fit(X, y)

    coef_l1_LR = clf_l1_LR.coef_.ravel()
    coef_l2_LR = clf_l2_LR.coef_.ravel()

    # coef_l1_LR contains zeros due to the
    # L1 sparsity inducing norm

    sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
    sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100

    print("C=%d" % C)
    print("Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR)
    print("score with L1 penalty: %.4f" % clf_l1_LR.score(X, y))
    print("Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR)
    print("score with L2 penalty: %.4f" % clf_l2_LR.score(X, y))

    l1_plot = plt.subplot(3, 2, 2 * i + 1)
    l2_plot = plt.subplot(3, 2, 2 * (i + 1))
    if i == 0:
        l1_plot.set_title("L1 penalty")
        l2_plot.set_title("L2 penalty")

    l1_plot.imshow(np.abs(coef_l1_LR.reshape(8, 8)), interpolation='nearest',
                   cmap='binary', vmax=1, vmin=0)
    l2_plot.imshow(np.abs(coef_l2_LR.reshape(8, 8)), interpolation='nearest',
                   cmap='binary', vmax=1, vmin=0)
    plt.text(-8, 3, "C = %d" % C)

    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l2_plot.set_xticks(())
    l2_plot.set_yticks(())

plt.show()

get_ipython().magic('pinfo make_classification')

X, y = make_classification(n_samples=m, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

y.size

len(datasets)



