get_ipython().run_cell_magic('capture', '', '%matplotlib notebook \n\n\nimport numpy as np\nimport math\nimport matplotlib.pyplot as plot\nfrom matplotlib import ticker\n\nfrom sklearn.metrics import r2_score\n\n###############################################################################\n# Lasso and Linear\nfrom sklearn.linear_model import Lasso,LinearRegression\n\nimport matplotlib\nfrom mpl_toolkits.mplot3d import Axes3D\n\nimport scipy.optimize as sciop\nfrom scipy.optimize import basinhopping\n\n!make fortran;\nimport irreverisble #importing the fortran mechanics routine\n\nglobal exp\nexp = []                           # ***** target \nexp = np.loadtxt(\'ref/HSRS/22\')\n\ndef error_evaluation_rms(errors):\n    \n    sum_of_squares = 0\n    \n    for error in errors:\n        sum_of_squares = sum_of_squares + error**2\n        \n    return ((sum_of_squares/len(errors))**(1./2.)) #incorporated division by n, which is the proper rms \n\ndef mcfunc(model_parameters):\n\n    # -------------- number samples, =1 in this case\n    no_samples = 1\n    T_service = 22. + 273.\n    prec_stress = 0\n    SS_stress = 750\n    \n    strain_stress, WTN = irreverisble.mechanics(prec_stress,SS_stress,T_service,model_parameters,no_samples)\n    strain_stress = np.array(np.trim_zeros(strain_stress)).reshape(-1,2)\n    #print strain_stress\n    \n    #----------------------------\n    cal_val = []\n    errors = []\n    \n    #traverses experimental data points\n    for iexp, data in enumerate(exp[:,0]):\n        \n        #finding nearest neighbors that surround the data points, and using them to determine the error\n        for ical, data in enumerate(strain_stress[:,0]):\n            \n            ical = ical-1 # May or may not be advantageous to keep this instead of the range attribute for mem save\n            \n            left_strainpoint = strain_stress[ical,0]\n            right_strainpoint = strain_stress[ical+1,0]\n            \n            exp_datapoint = exp[iexp,0]\n            \n            if(exp_datapoint>left_strainpoint and exp_datapoint<right_strainpoint):\n                                \n                # stores the differences between the successive approximations so we interpolate\n                left_difference = exp_datapoint-left_strainpoint\n                right_difference = right_strainpoint-exp_datapoint\n                \n                total_difference = left_difference+right_difference\n                \n                left_weight = left_difference/total_difference\n                right_weight = right_difference/total_difference\n                  \n                # interpolate stress based on strain?\n                interpolated_strain = left_weight*left_strainpoint + right_weight*right_strainpoint\n                interpolated_stress = left_weight*strain_stress[ical,1] + right_weight*strain_stress[ical+1,1]\n                    \n                stress_error = interpolated_stress - exp[iexp,1]    \n                #print stress_error\n                \n                #adds value, we want to find difference between these approximated data points and the real results\n                cal_val.append([interpolated_strain,interpolated_stress])                 \n                errors.append(stress_error)\n                \n                break\n    \n    #print errors\n    error_rms = error_evaluation_rms(errors)    \n    cal_val = np.asarray(cal_val)\n    \n    #print cal_val\n    #----------------------------\n    \n    # return error as well as the results of stress-strain curve?\n    #return strain_stress, error_rms\n    return error_rms\n\n"""\ndef IntervalPlot3D(xlabel="",ylabel="",zlabel="",title="",fontsize=14):\n\n\n    fig = plt.figure()\n    ax = fig.gca(projection=\'3d\')\n    plt.title(title)\n    matplotlib.rcParams.update({\'font.size\': fontsize})\n    \n    interval = 1.\n\n    x_domain = np.arange(-102.,-94.,interval)\n    y_domain = np.arange(5.,15.,interval)\n\n    x = np.zeros(0)\n    y = np.zeros(0)\n\n    for y_val in y_domain:\n\n        x = np.append(x,x_domain)\n\n        for x_val in x_domain:\n\n            y = np.append(y,y_val)\n\n    z = np.zeros(0)\n\n    for index, value in enumerate(x):\n\n        model_params = (x[index],y[index])\n        z = np.append(z,mcfunc(model_params))\n\n    ax.plot(x,y,z,"p")\n\n    ax.set_xlabel(xlabel)\n    ax.set_ylabel(ylabel)\n    ax.set_zlabel(zlabel)\n\n    plt.show()"""\n    \n#IntervalPlot3D(xlabel="Param 1",ylabel="Param 2",zlabel="Error from experimental results",title="Error", fontsize=16)')

"""import numpy as np
import scipy as sp
import matplotlib.pyplot as plot

from scipy.optimize import minimize

import timeit
from memory_profiler import memory_usage

#all methods to minimize
methods = ['Powell','CG','SLSQP']

start = np.zeros(0)
stop = np.zeros(0)
num_iters = np.zeros(0)

most_mem = np.zeros(0)
result = []

#runtime code goes here

function = mcfunc

#testing every minimization method
for method in methods:
    
    mem_use = memory_usage(-1,interval=0.1)
    start = np.append(start,timeit.default_timer())
    
    guess = [-5.,10.] # guess for correct minimum
    
    # Possibly was finding the iterations in the wrong order
    current_result = minimize(function, x0 = guess, method = method,tol=1e-6)
    result.append(current_result)
    
    keys = current_result.keys() # contains all traits of result
    iterations = -1
    
    if 'nit' in keys:    
        iterations = current_result.get('nit')
        
  ,'BFGS','L-BFGS-B','TNC','COBYLA','SLSQP'  num_iters = np.append(num_iters,iterations)
    stop = np.append(stop,timeit.default_timer())
    
    # tracks amount of memory used
    most_mem = np.append(most_mem,max(mem_use)) 

exec_time = stop-start

# If an algorithm took (-1) iterations, the number of iterations was not returned
for counter, method in enumerate(methods):
    
    print '{0} took {1} seconds. The result, {4} was found at ({2}, {3})'.format(method,exec_time[counter],result[counter].x[0],result[counter].x[1],result[counter].fun)
    print '{0} used {1} megabytes and took {2} iterations'.format(method,most_mem[counter],num_iters[counter])
    print
    """

get_ipython().run_cell_magic('capture', '', '"""import numpy as np\nimport scipy as sp\nimport matplotlib.pyplot as plot\n\nfrom scipy.optimize import minimize\n\nimport timeit\nfrom memory_profiler import memory_usage\n\nstart = np.zeros(0)\nstop = np.zeros(0)\nnum_iters = np.zeros(0)\n\nmost_mem = np.zeros(0)\n\n#runtime code goes here\n\nfunction = mcfunc\n\n#testing every minimization method\nfor method in methods:\n    \n    mem_use = memory_usage(-1,interval=0.1)\n    start = np.append(start,timeit.default_timer())\n    \n    guess = [-5.,10.] # guess for correct minimum\n    \n    # Possibly was finding the iterations in the wrong order\n    result = basinhopping(function, x0 = guess)\n    \n    keys = result.keys() # contains all traits of result\n    iterations = -1\n    \n    if \'nit\' in keys:    \n        iterations = result.get(\'nit\')\n        \n    num_iters = np.append(num_iters,iterations)\n    stop = np.append(stop,timeit.default_timer())\n    \n    # tracks amount of memory used\n    most_mem = np.append(most_mem,max(mem_use)) \n\nexec_time = stop-start\n\n# If an algorithm took (-1) iterations, the number of iterations was not returned\nfor counter, method in enumerate(methods):\n    \n    print \'{0} took {1} seconds. The result, {4} was found at ({2}, {3})\'.format(method,exec_time[counter],result.x[0],result.x[1],result.fun)\n    print \'{0} used {1} megabytes and took {2} iterations\'.format(method,most_mem[counter],num_iters[counter])\n    print"""')

#import test_suite

#test_suite.minimize_suite(mcfunc,['Powell','SLSQP'],[-500.,1.])

#import test_suite

#test_suite.minimize_suite(mcfunc,['Nelder-Mead','SLSQP'],[-500.,1.])

import test_suite

test_suite.minimize_suite(mcfunc,['Nelder-Mead','SLSQP'],[-500.,1.])

import test_suite

test_suite.minimize_suite(mcfunc,['Powell'],[-100.,1.])

import test_suite

test_suite.minimize_suite(mcfunc,['L-BFGS-B'],[-500.,1.])

guess = [-500.,1.]
#test_suite.minimize_suite(mcfunc,['Powell'],guess)

test_suite.minimize_suite(mcfunc,['Nelder-Mead'],guess)

test_suite.minimize_suite(mcfunc,['SLSQP'],guess)

test_suite.minimize_suite(mcfunc,['BFGS'],guess)


test_suite.minimize_suite(mcfunc,['TNC'],[guess])

test_suite.minimize_suite(mcfunc,['TNC'],[-100.,1.])

print mcfunc((-93.16876,0.8780065))

# we want to compare the results of the strain_stress with the experimental data to minimize error
import test_suite
strain_stress, error = mcfunc((-93.16876,0.8780065))
test_suite.plotSingle2D(strain_stress,'strain','stress','linear','linear')

"""def IntervalPlot3D(function, x_domain, y_domain, xlabel="",ylabel="",zlabel="",title="",fontsize=14):

    fig = plot.figure()
    ax = fig.gca(projection='3d')
    plt.title(title)
    matplotlib.rcParams.update({'font.size': fontsize})

    x = np.zeros(0)
    y = np.zeros(0)

    for y_val in y_domain:

        x = np.append(x,x_domain)

        for x_val in x_domain:

            y = np.append(y,y_val)

    z = np.zeros(0)

    for index, value in enumerate(x):

        z = np.append(z,function(x[index],y[index])

    ax.plot(x,y,z)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    plot.show()
    
IntervalPlot3D((lambda x,y : x+y**2), x_domain=np.arange(0,10,1.),y_domain=np.arange(0,10,1.))"""

"""
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 14

fig = plt.figure()
ax = fig.gca(projection='3d')

interval = 10.

x_domain = np.arange(-100.,0.,interval)
y_domain = np.arange(-10.,20.,interval)

x = np.zeros(0)
y = np.zeros(0)

for y_val in y_domain:
    
    x = np.append(x,x_domain)
    
    for x_val in x_domain:
       
        y = np.append(y,y_val)

z = np.zeros(0)
        
for index, value in enumerate(x):
    
    model_params = (x[index],y[index])
    print mcfunc(model_params)
    #z = np.append(z,mcfunc(model_params))
    
ax.plot(x,y,z,"p")

ax.set_xlabel("Parameter 1", fontsize = 16)
ax.set_ylabel("Parameter 2", fontsize = 16)
ax.set_zlabel("Error from experimental results", fontsize = 16)

plt.show()
"""

import test_suite
test_suite.IntervalPlot3D(mcfunc,np.arange(-100.,-90.,1.),np.arange(0.01,5.01,0.2),xlabel="Parameter 1",
                          ylabel="Parameter 2", title = "Error Minimization in a Basin")

