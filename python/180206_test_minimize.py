import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
# import MSM_util
from MSM_util import *
import sys, scipy, numpy

# print(scipy.__version__, numpy.__version__, sys.version_info)

GLD = pd.read_excel('data_GVZ_GLD.xlsx')
date_GLD = GLD.iloc[:,3]
GLD = GLD.loc[:,'GLD']
GLD_d = GLD - np.mean(GLD)
# plt.plot(date_GLD,GLD_d)
data = GLD_d

kbar = 5
startingvals = []
LB = [1, 1, 0.001, 0.0001]
UB = [50, 1.99, 0.99999, 5]

# set up A_template for a transition matrix
A_template = T_mat_temp(kbar)


# Grid search for starating values
# TODO: (try sklearn.GridsearchCV) or map lambda instead of double for-loop
# input_param, LLS = MSM_starting_values(data, startingvals, kbar, A_template)
input_param, LLS = MSM_starting_values2(data, startingvals, kbar)
input_param[0] = sum(input_param[0])
bound = list(map(lambda x,y: (x,y), LB,UB))

# input_param = np.asarray(input_param)


param_op = minimize(MSM_likelihood_new2, input_param, args=(kbar, data),
                    method ='tnc', bounds=bound, tol= 1e-5)

input_param2 = np.asarray(input_param)
minimize(MSM_likelihood_new2, input_param2, args=(kbar, data),method ='L-BFGS-B', bounds=bound, tol= 1e-5)

from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import numpy as np



# create a set of Parameters
params = Parameters()
params.add('m0',   value= input_param[0],  min=LB[0], max = UB[0])
params.add('b', value= input_param[1],  min=LB[1], max = UB[1])
params.add('gamma_k', input_param[2],  min=LB[2], max = UB[2])
params.add('sigma', value= input_param[3],  min=LB[3], max = UB[3])


# minimize(MSM_likelihood_new2, input_param2, args=(kbar, data),method ='L-BFGS-B', bounds=bound, tol= 1e-5)
# do fit, here with leastsq model
minner = Minimizer(MSM_likelihood_new2V2, params, fcn_args=(kbar, data))
result = minner.minimize(method ='L-BFGS-B')

# # calculate final result
# final = data + result.residual

# # write error report
# report_fit(result)

# # try to plot results
# try:
#     import pylab
#     pylab.plot(x, data, 'k+')
#     pylab.plot(x, final, 'r')
#     pylab.show()
# except:
#     pass

input_param[0]

params['m0'].value

from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import numpy as np

# create a set of Parameters
params = Parameters()
params.add('m0',   value= input_param[0],  min=LB[0], max = UB[0])
params.add('b', value= input_param[1],  min=LB[1], max = UB[1])
params.add('gamma_k', input_param[2],  min=LB[2], max = UB[2])
params.add('sigma', value= input_param[3],  min=LB[3], max = UB[3])

minner = Minimizer(MSM_likelihood_new2V2, params, fcn_args=(kbar, data))
result = minner.minimize(method='slsqp')




