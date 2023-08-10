import pandas as pd
import itertools
import numpy as np
import seaborn as sns
import pylab

import scipy.stats as stats
import statsmodels.api as sm

column_labs = ['x%d'%(i+1) for i in range(6)]
encoded_inputs = list( itertools.product([-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]) )
doe = pd.DataFrame(encoded_inputs,columns=column_labs)
print(len(doe))

doe['x1-x2-x3-x4'] = doe.apply( lambda z : z['x1']*z['x2']*z['x3']*z['x4'] , axis=1)
doe['x4-x5-x6']    = doe.apply( lambda z : z['x4']*z['x5']*z['x6'] , axis=1)
doe['x2-x4-x5']    = doe.apply( lambda z : z['x2']*z['x4']*z['x5'] , axis=1)

doe[0:10]

print(len( doe[doe['x1-x2-x3-x4']==1] ))

# Defining multiple DOE matrices:

# DOE 1 based on identity I = x1 x2 x3 x4 
doe1 = doe[doe['x1-x2-x3-x4']==1]

# DOE 2 based on identity I = x4 x5 x6
doe2 = doe[doe['x4-x5-x6']==-1]

# DOE 3 based on identity I = x2 x4 x5
doe3 = doe[doe['x2-x4-x5']==-1]

doe1[column_labs].T

doe2[column_labs].T

doe3[column_labs].T

quarter_fractional_doe = doe[ np.logical_and( doe['x1-x2-x3-x4']==1, doe['x4-x5-x6']==1 ) ]
print("Number of experiments: %d"%(len(quarter_fractional_doe[column_labs])))
quarter_fractional_doe[column_labs].T

