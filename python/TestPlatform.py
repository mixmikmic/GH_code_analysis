import os
os.environ['INPUTFILE'] = 'inputData/pricedata_LMP_100_short.csv'

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import pandas as pd
import numpy as np
import scipy.sparse as sps1
from copy import deepcopy
from datetime import *
import dateutil
from dateutil import parser, relativedelta
import pytz
import sys, os, pickle


from simulationFunctions import *

import sys  # This is necessary for printing updates within a code block, via sys.stdout.flush()
import time # Use time.sleep(secs) to sleep a process if needed

print("Running Storage Efficiency sweep")
print("Current environment directory:" + sys.prefix)
print("System version: "+sys.version)

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)


try:
    fname = os.environ['INPUTFILE']
except KeyError:
    fname = "inputData/pricedata_LMP_100.csv" # Only 100 nodes    

print("Running Storage Efficiency sweep with input file "+fname)
APNode_Prices = pd.read_csv( fname, header=0,index_col=0)#,nrows=10)
APNode_Prices.columns = pd.DatetimeIndex(APNode_Prices.columns,tz=dateutil.tz.tzutc())  # Note: This will be in UTC time. Use .tz_localize(pytz.timezone('America/Los_Angeles')) if a local time zone is desired- but note that this will 

## Deal with NaN prices
# Drop nodes which are above a cutoff
goodNodes = (APNode_Prices.isnull().sum(axis=1) < (0.02 * APNode_Prices.shape[1])) # True if node is less than x% NaN values
APNode_Prices = APNode_Prices[goodNodes]
# Interpolate remaining NaNs
APNode_Prices.interpolate(method='linear',axis=1)
print("Finished Loading Data")
sys.stdout.flush()

negativeNodes = pickle.load(open('nodeList.pkl','rb'))

startDate = parser.parse('01/01/12 00:00')
endDate   = parser.parse('12/31/16 23:00')
# startDate = pytz.utc.localize(startDate)
# endDate   = pytz.utc.localize(endDate)

thisSlice = APNode_Prices.loc[negativeNodes[0:1],startDate:endDate]
thisSlice.shape

reservoirSize=1
E_min = 0
E_max = 1
# The 1-hour system will be SOC constrained, rather than power constrained. We accordingly don't worry about P_max and P_min

# pid = multiprocessing.current_process().pid

timestep = relativedelta.relativedelta(thisSlice.columns[2],thisSlice.columns[1])
delta_T = timestep.hours  # Time-step in hours

startDate = thisSlice.columns.values[ 0].astype('M8[m]').astype('O') # Convert to datetime, not timestamp
endDate   = thisSlice.columns.values[-1].astype('M8[m]').astype('O')
startDate = pytz.utc.localize(startDate)
endDate   = pytz.utc.localize(endDate)
timespan = relativedelta.relativedelta(endDate +timestep, startDate)
simulationYears = timespan.years + timespan.months/12. + timespan.days/365. + timespan.hours/8760.  # Leap years will be slightly more than a year, and that's ok.

print("Simulation is %.3f years long"%simulationYears)

myEfficiencies = np.arange(0.7,1.01,0.1)
lastEfficiency = 0  # This is used to track whether the efficiency has switched
storagePrice = 0. * simulationYears # Amortized cost of storage
myLength = thisSlice.shape[1]

model = CyClpSimplex()
model.logLevel = 0
x_var = model.addVariable('x',myLength*3+2)
h_constant = sps.hstack( [1, sps.coo_matrix((1, myLength*3+1))] ) # Force h to be a specific size:         
(A,b) = createABineq_noPowerConstraint(myLength, E_min, E_max)

# (A,b) = createABineq_noPowerConstraint(myLength, E_min, E_max, P_min, P_max)
# (A,b, A_eq, B_eq) = createABMatrices(myLength, delta_T, eff_in, eff_out, P_min, P_max, E_min, E_max) 

model.addConstraint(h_constant * x_var == reservoirSize,'fixedSize')
model.addConstraint(         A * x_var <= b.toarray(),  'inequalities')

# Result dataframe: Size, kwhPassed, and profits for each node, at each efficiency (columns)
resultIndex = pd.MultiIndex.from_product([thisSlice.index,['cycleCount','storageProfit']])
results = pd.DataFrame(index = resultIndex, columns=myEfficiencies)
powerOut = pd.DataFrame(index = thisSlice.index, columns = thisSlice.columns)

#### LOOP THROUGH nodes
for i in range(thisSlice.shape[0]):
    print("Working on node %s"%thisSlice.index[i])
    # Define cost function
    myNodeName = thisSlice.index[i]
    energyPrice = thisSlice.loc[myNodeName,:] / 1000.0 # Price $/kWh as array

    c = np.concatenate([[storagePrice],[0]*(myLength+1),energyPrice,energyPrice],axis=0)  # No cost for storage state; charged for what we consume, get paid for what we discharge
    c_clp = CyLPArray(c)
    model.objective = c_clp * x_var

    for eff_round in myEfficiencies:

        if eff_round != lastEfficiency:  # If we just switched nodes (and not efficiencies) don't bother updating efficiencies
            try:
                model.removeConstraint('equalities')
            except:
                pass
            try:
                model.removeConstraint('powermax')
            except:
                pass
            (eff_in, eff_out) = [np.sqrt(eff_round)] *2  # Properly account for the round trip efficiency of storage
            P_max = E_max/eff_in # Max charge power, e.g max limit for C_i
            P_min = -1*E_max*eff_out # Max discharge power, e.g. min D_i
            (A_eq, b_eq) = createABeq(myLength, delta_T, eff_in, eff_out)
            (A_P, b_p)   = createPowerConstraint(myLength, P_min, P_max)
            model.addConstraint(A_eq * x_var == b_eq.toarray(),'equalities')
            model.addConstraint(A_P  * x_var <= b_p,'powermax')

        model.primal()  # Solve

        x = model.primalVariableSolution['x']
        results.loc[(myNodeName,'storageProfit'),eff_round] = np.dot(-c, x)   #Calculate profits at optimal storage level
        c_grid = x[2+myLength : 2+myLength*2]
        results.loc[(myNodeName,'cycleCount'),   eff_round] = sum(c_grid)*eff_in # net kWh traveled
        print("Profits of %.3f for efficiency %s" %(np.dot(-c,x), eff_round))
        
        if round(eff_round,2) == 0.9:
            powerOut.loc[myNodeName,:] = x[2+myLength : 2+myLength*2] + x[2+myLength*2 : 2+myLength*3]

        lastEfficiency = eff_round

    # Done with the loop; reverse the efficiency set and move on to the next node
    myEfficiencies = myEfficiencies[::-1]



