get_ipython().system('pip install pysal')

import pysal
import numpy as np
from pysal.spreg import ols
from pysal.spreg import ml_error
from pysal.spreg import ml_lag

f = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
y = np.array(f.by_col['HOVAL'])
y.shape = (len(y),1)
X= []
X.append(f.by_col['INC'])
X.append(f.by_col['CRIME'])
X = np.array(X).T

ls = ols.OLS(y, X, name_y = 'home val', name_x = ['Income', 'Crime'], name_ds = 'Columbus')
print(ls.summary)

w = pysal.open(pysal.examples.get_path("columbus.gal")).read()

mi = pysal.Moran(ls.u, w, two_tailed=False)
print('Observed I:', mi.I, '\nExpected I:', mi.EI, '\n   p-value:', mi.p_norm)

spat_err = ml_error.ML_Error(y, X, w, 
                             name_y='home value', name_x=['income','crime'], 
                             name_w='columbus.gal', name_ds='columbus')
print(spat_err.summary)



