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
import sys

import matplotlib.pyplot as plt
#import matplotlib.colors as mpl_colors

from simulationFunctions import *

from IPython.display import clear_output, display
import sys  # This is necessary for printing updates within a code block, via sys.stdout.flush()
import time # Use time.sleep(secs) to sleep a process if needed
get_ipython().magic('matplotlib inline')

print("Current environment directory:" + sys.prefix)
print("System version: "+sys.version)

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

fs = 14
plt.rc('font',family='Times New Roman')
fn = 'Times New Roman'

# import price data as a dataframe: columns are times, rows are nodes.  Size is nodes x 8760
# fname = "/Users/emunsing/GoogleDrive/CE 290 Project/Data Collection/Prices/R code/All_PNodes_LMP_Aggregated_2013short.csv" #OUTDATED
# fname = "inputData/pricedata_LMP.csv" # FULL
fname = "inputData/pricedata_LMP_100.csv" # Only 100 nodes
# fname = "inputData/pricedata_LMP_5.csv" # Only 5 nodes

APNode_Prices = pd.read_csv( fname, header=0,index_col=0)#,nrows=10)
APNode_Prices.columns = pd.DatetimeIndex(APNode_Prices.columns,tz=dateutil.tz.tzutc())  # Note: This will be in UTC time. Use .tz_localize(pytz.timezone('America/Los_Angeles')) if a local time zone is desired- but note that this will 
timestep = relativedelta.relativedelta(APNode_Prices.columns[2],APNode_Prices.columns[1])
delta_T = timestep.hours  # Time-step in hours

## Deal with NaN prices
# Drop nodes which are above a cutoff
goodNodes = (APNode_Prices.isnull().sum(axis=1) < (0.02 * APNode_Prices.shape[1])) # True if node is less than x% NaN values
APNode_Prices = APNode_Prices[goodNodes]
# Interpolate remaining NaNs
APNode_Prices.interpolate(method='linear',axis=1)

print(APNode_Prices.shape)
APNode_Prices.head()

startDate = parser.parse('08/01/13 00:00')  # year starts at 2013-01-01 00:00:00
endDate = parser.parse('08/03/13 06:00')  # year ends at 2013-12-31 23:00:00

timespan =  endDate + timestep - startDate  # Length of time series (integer)
simulationYears = (timespan.days*24 + timespan.seconds/3600) / 8760.  # Leap years will be slightly more than a year, and that's ok.
storagePrice = 0 * simulationYears # Amortized cost of storage

myEfficiencies = [0.6,0.8,0.9]
reservoirSize=1

E_min = 0
E_max = 1

nodeName = 'BARRY_6_N001'
myEnergyPrices = APNode_Prices.loc[nodeName,startDate:endDate].values / 1000.0
myLength = len(myEnergyPrices)
nans, x = nan_helper(myEnergyPrices)
myEnergyPrices[nans] = np.interp(x(nans), x(~nans), myEnergyPrices[~nans])

# Define cost function
c = np.concatenate([[storagePrice],[0]*(myLength+1),myEnergyPrices,myEnergyPrices],axis=0)  # No cost for storage state; charged for what we consume, get paid for what we discharge
c_clp = CyLPArray(c)
# Use the following section to force h to be a specific size:
h_constant = sps.hstack( [1, sps.coo_matrix((1, myLength*3+1))] );               
               
# Define model
model = CyClpSimplex()
x_var = model.addVariable('x',len(c))
model.objective = c_clp * x_var

print("Done with prep")

# Run loop through each Efficiency

resultDf = pd.DataFrame(columns=['storageSize','storageProfit','cycleCount','eState'])
results = {}

solverStartTime = time.time()

for eff_round in myEfficiencies:
    (eff_in, eff_out) = [np.sqrt(eff_round)] *2  # Properly account for the round trip efficiency of storage

    P_max = E_max/eff_in # Max discharge power, e.g max limit for C_i
    P_min = -1*E_max/eff_out # Max charge power, e.g. min D_i

    # Set up CyLP model
    (A, b, A_eq, b_eq) = createABMatrices(myLength, delta_T, eff_in, eff_out, P_min, P_max, E_min, E_max)

    # Force h to be a specific size:
    A_eq = sps.vstack( [h_constant, A_eq ] )
    b_eq = sps.vstack( [reservoirSize, b_eq] )

    # Define model
    model = CyClpSimplex()
    x_var = model.addVariable('x',len(c))
    model.objective = c_clp * x_var
    model += A * x_var <= b.toarray()
    model += A_eq * x_var == b_eq.toarray()
    
    model.primal()  # Solve

    x = model.primalVariableSolution['x']
    results['storageSize'] = x[0]
    results['storageProfit'] = np.dot(-c, x)   #Calculate profits at optimal storage level
    c_grid = x[2+myLength : 2+myLength*2]
    results['cycleCount'] = sum(c_grid)*eff_in # net kWh traveled
    results['eState'] = x[2 : myLength+2]
    
    resultDf.loc[str(eff_round),:] = results
    
solverEndTime = time.time()
clear_output()

print('Total function call time: %.3f seconds' % (solverEndTime - solverStartTime))

resultDf.to_csv('Data/SweepStorageEfficiency_BARRY_6_N001_ForPlotting.csv')

## Plotting and saving data for export

eStates = np.ndarray((resultDf.shape[0],myLength))
j = 0

for eff_round in resultDf.index: # ['1','4','8']: #
    print eff_round
    eStates[j,:] = resultDf.loc[eff_round,'eState']#/maxSize
    j = j+1
#    plt.plot(range(myLength),resultDf.loc[thisSize,'eState']/maxSize)

plt.plot(range(myLength),eStates.transpose())
    
np.savetxt('Data/chargeStates_efficiency.csv',eStates,delimiter=',')

## THIS IS USED FOR THE PAPER
startDate = parser.parse('08/01/13 00:00')  # year starts at 2013-01-01 00:00:00
endDate = parser.parse('08/03/13 06:00')  # year ends at 2013-12-31 23:00:00

# This is for arbitrary length
startDate = parser.parse('01/01/13 00:00')  # year starts at 2013-01-01 00:00:00
endDate = parser.parse('06/30/13 23:00')  # year ends at 2013-12-31 23:00:00

# This is the full dataset
startDate = APNode_Prices.columns.values[ 0].astype('M8[m]').astype('O')
endDate   = APNode_Prices.columns.values[-1].astype('M8[m]').astype('O')

timespan =  endDate + timestep - startDate  # Length of time series (integer)
simulationYears = (timespan.days*24 + timespan.seconds/3600) / 8760.  # Leap years will be slightly more than a year, and that's ok.

storagePriceSet = [5]
storagePrice = storagePriceSet[0] * simulationYears # Amortized cost of storage

E_min = 0
E_max = 1

saveEState = False  # Switch this to 'True' if we want to save the charge history for plotting validation figures

## Market Price for desired APNode
numberOfNodes = APNode_Prices.shape[0] # number of nodes 

## Variable Set-up

# Manage efficiencies
myefficiencies = [0.9]

k = 0
eff_round = myefficiencies[k]  # Round-trip efficiency
(eff_in, eff_out) = [np.sqrt(eff_round)] *2  # Properly account for the round trip efficiency of storage

P_max = E_max/eff_in # Max discharge power, e.g max limit for C_i
P_min = -1*E_max/eff_out # Max charge power, e.g. min D_i

myNodeName = 'BARRY_6_N001' # Use this if we're just interested in a specific node

myEnergyPrices = APNode_Prices.loc[myNodeName,startDate:endDate] / 1000. # Price $/kWh as array
myLength = len(myEnergyPrices)

# Deal with NaN entries in price series
nans, x = nan_helper(myEnergyPrices)
myEnergyPrices[nans] = np.interp(x(nans), x(~nans), myEnergyPrices[~nans])

# Set up CyLP model

# Create A-matrices once efficiency is specified
(A, b, A_eq, b_eq) = createABMatrices(myLength, delta_T, eff_in, eff_out, P_min, P_max, E_min, E_max)

# Define cost function
c = np.concatenate([[storagePrice],[0]*(myLength+1),myEnergyPrices,myEnergyPrices],axis=0)  # No cost for storage state; charged for what we consume, get paid for what we discharge

# Define model
model = CyClpSimplex()
x_var = model.addVariable('x',len(c))
c_clp = CyLPArray(c)
model.objective = c_clp * x_var

# Use the following section to force h to be a specific size:
h_constant = sps.hstack( [1, sps.coo_matrix((1, myLength*3+1))] );
A_eq = sps.vstack( [h_constant, A_eq ] )
b_eq = sps.vstack( [1, b_eq] )  # the 1 here will be overwritten by the reservoir size later

# Add constraints to model
model += A * x_var <= b.toarray()
model += A_eq * x_var == b_eq.toarray()

# Run loop through each storage size

storageSizes = np.arange(0,10,0.1)
# storageSizes = [2,6,12,18]
resultDf = pd.DataFrame(columns=['storageSize','storageProfit','cycleCount','eState'])
results = {}

solverStartTime = time.time()

for j in storageSizes:
    # Change mandated size of system: want to set the upper and lower bounds of the first inequality constraint (index is len(a.toarray()))
    model.setRowUpper(b.shape[0],j)
    model.setRowLower(b.shape[0],j)

    model.primal()  # Solve

    x = model.primalVariableSolution['x']
    results['storageSize'] = x[0]
    results['storageProfit'] = np.dot(-c, x)   #Calculate profits at optimal storage level
    e_state = x[2 : myLength+2]
    c_grid = x[2+myLength : 2+myLength*2]
    #d_grid = x[2+myLength*2 : ]
    #p_batt = c_grid + d_grid
    results['cycleCount'] = sum(c_grid)*eff_in # net kWh traveled
    
    if saveEState:
        results['eState'] = e_state
    
    resultDf.loc[str(j),:] = results
    #resultDf.append(results.values(), columns = ['storageSize','storageProfit','cycleCount'], ignore_index=True)
    #print('Size: \t %.4f \t Profits: \t %.4f \t Cycle Count: %.4f' % tuple(results.values()))
    
resultDf['annualStorageProfit'] = resultDf['storageProfit'] / simulationYears

solverEndTime = time.time()
clear_output()

#print('Storage size: %.4f' % storageSize)
print('Total function call time: %.3f seconds' % (solverEndTime - solverStartTime))
#print('Profits: %4f' % storageProfit)

resultDf.to_csv('Data/SweepStorageSize_ForPlotting.csv')

## Plot Results ## 
fs = 12

plt.plot(resultDf['storageSize'],resultDf['annualStorageProfit'])
plt.xlabel('Reservoir Size (h)', fontsize=fs+1)
plt.ylabel('Long-run annual profits ($/kWh/yr)', fontsize=fs+1)
plt.suptitle('Profits with Varying Reservoir Sizes', fontsize= fs+2)

plt.savefig('Plots/Profit_VaryingSize.pdf',bbox_inches='tight')

ax = plt.subplot()
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #label.set_fontname('Arial')
    label.set_fontsize(fs)

resultSample = [1,2,6]

eStates = np.ndarray([myLength,len(resultSample)])
j = 0

for thisSize in resultSample: #
    maxSize = float(thisSize)
    print maxSize
    eStates[:,j] = resultDf.loc[str(thisSize),'eState']/maxSize
    j = j+1
#    plt.plot(range(myLength),resultDf.loc[thisSize,'eState']/maxSize)

plt.plot(range(myLength),eStates)
    
np.savetxt('Data/chargeStates_varySize.csv',eStates,delimiter=',')

def computePriceSweep(thisSlice):
    ###### BEGINNING OF MULTIPROCESSING FUNCTION ###########
    myLength = thisSlice.shape[1]

    # Simulation parameters - set these!
    storagePriceSet = np.arange(1.0,20.1,0.1)
    storagePriceSet = np.arange(1.0,3.1,1)
    eff_round = 0.9  # Round-trip efficiency
    E_min = 0
    E_max = 1

    # Endogenous parameters; calculated automatically
    (eff_in, eff_out) = [np.sqrt(eff_round)] *2  # Properly account for the round trip efficiency of storage
    P_max = E_max/eff_in # Max discharge power at grid intertie, e.g max limit for C_i
    P_min = -1*E_max/eff_out # Max charge power at grid intertie, e.g. min D_i

    # Create a result dataframe
    resultIndex = pd.MultiIndex.from_product([thisSlice.index,['size','kwhPassed','profit']])
    results = pd.DataFrame(index = resultIndex, columns=storagePriceSet)

    # Create clean problem matrices - needs efficiency and length!
    (A, b, A_eq, b_eq) = createABMatrices(myLength, delta_T, eff_in, eff_out, P_min, P_max, E_min, E_max)
    # Define model:
    coolStart = CyClpSimplex()
    coolStart.logLevel = 0
    x_var = coolStart.addVariable('x',myLength*3+2)
    # Add constraints to model:
    coolStart += A * x_var <= b.toarray()
    coolStart += A_eq * x_var == b_eq.toarray()
    # print("Finished with problem setup")
    # sys.stdout.flush()

    # everythingStarts = time.time()
    # startTime = time.time()

    for myNodeName in thisSlice.index:
        ## Set up prices
    #     myNodeName = thisSlice.index[i]
        energyPrice = thisSlice.loc[myNodeName,:] / 1000.0 # Price $/kWh as array

        c = np.concatenate([[0.0],[0.0]*(myLength+1),energyPrice,energyPrice],axis=0)  #placeholder; No cost for storage state; charged for what we consume, get paid for what we discharge
        #[[storagePricePlaceholder],[0]*(myLength+1),myPrice,myPrice],axis=1)
        c_clp = CyLPArray(c)

        sweepStartTime = time.time()

        for myStoragePrice in storagePriceSet:
            c_clp[0] = myStoragePrice * simulationYears
            coolStart.objective = c_clp * x_var

            # Run the model
            coolStart.primal()

            # Results- Rows are Nodes, indexed by name. Columns are Storage price, indexed by price
            x_out = coolStart.primalVariableSolution['x']
            results.loc[(myNodeName,'size'),     myStoragePrice] = x_out[0]
            results.loc[(myNodeName,'profit'),   myStoragePrice] = np.dot(-c, x_out)
            results.loc[(myNodeName,'kwhPassed'),myStoragePrice] = sum(x_out[2+myLength : 2+myLength*2]) * eff_in # Note: this is the net power pulled from the grid, not the number of cycles when the system is unconstrained

        storagePriceSet = storagePriceSet[::-1] # Reverse the price set so that we start from the same price for the next node to make warm-start more effective

    #     if ((i+1) % reportFrequency == 0): # Save our progress along the way
    #         elapsedTime = time.time()-startTime
    #         print("Finished node %s; \t%s computations in \t%.4f s \t(%.4f s/solve)" 
    #               % (i, scenariosPerReport,elapsedTime,elapsedTime/scenariosPerReport))
    #         sys.stdout.flush()
    #         sizeDf.to_csv('Data/VaryingPrices_StorageSizing_v2.csv')
    #         profitDf.to_csv('Data/VaryingPrices_StorageProfits_v2.csv')
    #         cycleDf.to_csv('Data/VaryingPrices_StorageCycles_v2.csv')

    # print("Done in %.3f s"%(time.time()-everythingStarts))
    return results


## CUSTOM START/END DATE
startDate = parser.parse('01/01/12 00:00')  # year starts at 2013-01-01 00:00:00
endDate = parser.parse('01/31/12 23:00')  # year ends at 2013-12-31 23:00:00
startDate = pytz.timezone('America/Los_Angeles').localize(startDate).astimezone(pytz.utc)
endDate = pytz.timezone('America/Los_Angeles').localize(endDate).astimezone(pytz.utc)

# ## FULL DATASET
# startDate = APNode_Prices.columns.values[ 0].astype('M8[m]').astype('O') # Convert to datetime, not timestamp
# endDate   = APNode_Prices.columns.values[-1].astype('M8[m]').astype('O')
# startDate = pytz.utc.localize(startDate)
# endDate   = pytz.utc.localize(endDate)

timespan = relativedelta.relativedelta(endDate +timestep, startDate)
simulationYears = timespan.years + timespan.months/12. + timespan.days/365. + timespan.hours/8760.  # Leap years will be slightly more than a year, and that's ok.

startNode = 0
stopNode  = 0 # if set to zero, then will loop through all nodes
if ((stopNode == 0)|(stopNode > APNode_Prices.shape[0])): stopNode = APNode_Prices.shape[0]

    
thisSlice = APNode_Prices.ix[startNode:stopNode,startDate:endDate]
# someResults = computePriceSweep(APNode_Prices)


import multiprocessing

# Split dataset into roughly even chunks
j = min(multiprocessing.cpu_count(),10)
# chunksize = (APNode_Prices.shape[0]/j)+1
# splitFrames = [df for g,df in APNode_Prices.groupby(np.arange(APNode_Prices.shape[0])//chunksize)]
chunksize = (thisSlice.shape[0]/j)+1
splitFrames = [df for g,df in thisSlice.groupby(np.arange(thisSlice.shape[0])//chunksize)]

pool = multiprocessing.Pool(processes = j)
resultList = pool.map(computePriceSweep,splitFrames)
joinedResults = pd.concat(resultList)

joinedResults.sort_index(inplace=True)
sizeDf   = joinedResults.loc[(slice(None),'size'),:].reset_index(level=1,drop=True)
profitDf = joinedResults.loc[(slice(None),'profit'),:].reset_index(level=1,drop=True)
cycleDf  = joinedResults.loc[(slice(None),'kwhPassed'),:].reset_index(level=1,drop=True)

sizeDf.to_csv('Data/VaryingPrices_StorageSizing_v2.csv')
profitDf.to_csv('Data/VaryingPrices_StorageProfits_v2.csv')
cycleDf.to_csv('Data/VaryingPrices_StorageCycles_v2.csv')

## Model Set-up

# Note: the following is a list of 200 nodes with node codes that begin
# with unique 4-letter prefixes. This should help to create a random sample of prices and geographies
#uniqueNodes = [1,2,3,4,5,6,20,22,23,24,25,27,33,34,36,37,38,39,42,43,44,45,46,47,48,49,50,51,54,56,57,58,59,60,62,65,68,69,71,73,74,76,78,82,83,86,87,88,89,91,92,93,94,95,101,104,106,107,110,114,116,117,118,122,124,125,131,134,135,137,142,143,144,150,151,152,153,154,157,158,160,166,167,169,172,173,174,176,177,178,179,180,182,187,188,189,190,193,196,199,200,202,205,206,208,211,213,224,229,230,233,237,240,242,243,244,245,246,248,249,250,252,254,256,257,260,261,264,266,272,274,276,278,279,281,282,285,287,288,290,292,293,297,300,302,305,307,308,309,310,314,315,316,317,318,320,324,328,329,332,333,335,338,340,341,342,344,346,347,352,357,359,360,364,368,369,371,375,376,377,380,382,386,387,388,393,395,396,405,406,408,410,413,416,418,420,421,423,426,432]

# Computational burden is superlinear, so break into 1-year chunks. 1 year computes in about 0.25s per node after cold start; 5-year dataset takes >4min for solve

startDate = parser.parse('01/01/13 00:00')  # year starts at 2013-01-01 00:00:00
endDate = parser.parse('12/31/13 23:00')  # year ends at 2013-12-31 23:00:00

# Full dataset
# startDate = APNode_Prices.columns.values[ 0].astype('M8[m]').astype('O')
# endDate   = APNode_Prices.columns.values[-1].astype('M8[m]').astype('O')

timespan =  endDate + timestep - startDate  # Length of time series (integer)
simulationYears = (timespan.days*24 + timespan.seconds/3600) / 8760.  # Leap years will be slightly more than a year, and that's ok.
myLength = APNode_Prices.columns.get_loc(endDate) - APNode_Prices.columns.get_loc(startDate)+1

eff_round = 0.9  # Round-trip efficiency
(eff_in, eff_out) = [np.sqrt(eff_round)] *2  # Properly account for the round trip efficiency of storage

E_min = 0
E_max = 1
P_max = E_max/eff_in # Max discharge power at grid intertie, e.g max limit for C_i
P_min = -1*E_max/eff_out # Max charge power at grid intertie, e.g. min D_i

# Create clean problem matrices - first specify efficiency!
(A, b, A_eq, b_eq) = createABMatrices(myLength, delta_T, eff_in, eff_out, P_min, P_max, E_min, E_max)
# Define model:
coolStart = CyClpSimplex()
x_var = coolStart.addVariable('x',myLength*3+2)
# Add constraints to model:
coolStart += A * x_var <= b.toarray()
coolStart += A_eq * x_var == b_eq.toarray()

## LOOP BLOCK- loops through nodes, and then for each node loops through each price

## Sweep storage price
#storagePriceSet = [5,5.1,19.9, 20]
storagePriceSet = np.arange(1.0,20.1,0.1) #np.arange(1,20,0.1)

startNode = 0
stopNode = 0 # if set to zero, then will loop through all nodes
if ((stopNode == 0)|(stopNode > APNode_Prices.shape[0])): stopNode = APNode_Prices.shape[0]

# Initializing variables for loop
myStoragePrice = storagePriceSet[0]
badNodes = []
sizeDf = pd.DataFrame()
profitDf = pd.DataFrame()
cycleDf = pd.DataFrame()
                          
for i in range(startNode,stopNode):
    ## Set up prices
    myNodeName = APNode_Prices.index[i]
    energyPrice = APNode_Prices.loc[myNodeName,startDate:endDate] / 1000.0 # Price $/kWh as array

    # Deal with NaN prices
    nodePriceNanCount = sum(np.isnan(energyPrice))
    if (0.05 < (nodePriceNanCount/myLength)):  # Set cutoff threshold here
        badNodes.push(i) # Add this to the list of bad nodes
        continue # Jump to the next node
    nans, x = nan_helper(energyPrice)
    energyPrice[nans] = np.interp(x(nans), x(~nans), energyPrice[~nans])

    c = np.concatenate([[0.0],[0.0]*(myLength+1),energyPrice,energyPrice],axis=0)  #placeholder; No cost for storage state; charged for what we consume, get paid for what we discharge
    #[[storagePricePlaceholder],[0]*(myLength+1),myPrice,myPrice],axis=1)
    c_clp = CyLPArray(c)

    sweepStartTime = time.time()

    for myStoragePrice in storagePriceSet:
        c_clp[0] = myStoragePrice * simulationYears
        coolStart.objective = c_clp * x_var

        # Run the model
        startTime = time.time()
        coolStart.primal()
        endTime = time.time()

        # Results- Rows are Nodes, indexed by name. Columns are Storage price, indexed by price
        x_out = coolStart.primalVariableSolution['x']
        sizeDf.loc[myNodeName,str(myStoragePrice)]   = x_out[0]
        profitDf.loc[myNodeName,str(myStoragePrice)] = np.dot(-c, x_out)
        cycleDf.loc[myNodeName,str(myStoragePrice)]  = sum(x_out[2+myLength : 2+myLength*2]) * eff_in
#         print("Finished with initial solve in \t%.4f s for node %s" % ((endTime-startTime),myNodeName))
        sys.stdout.flush()
    
    sweepTime = time.time()-sweepStartTime
    print("Finished %s computations in \t%.4f s for node %s \t(%.4f s/solve)" % (len(storagePriceSet), sweepTime, myNodeName, sweepTime/(len(storagePriceSet)-1) ))
    sys.stdout.flush()
    storagePriceSet = storagePriceSet[::-1] # Reverse the price set so that we start from the same price for the next node to make warm-start more effective

    if (i % 10 == 0): # Save our progress along the way
        sizeDf.to_csv('Data/VaryingPrices_StorageSizing_v2.csv')
        profitDf.to_csv('Data/VaryingPrices_StorageProfits_v2.csv')
        cycleDf.to_csv('Data/VaryingPrices_StorageCycles_v2.csv')
    
#clear_output()
print("Finished; writing all to file...")
sizeDf.to_csv('Data/VaryingPrices_StorageSizing_v2.csv')
profitDf.to_csv('Data/VaryingPrices_StorageProfits_v2.csv')
cycleDf.to_csv('Data/VaryingPrices_StorageCycles_v2.csv')

sizeDf = pd.read_csv('Data/VaryingPrices_StorageSizing_v2.csv',header=0,index_col=0)
profitDf = pd.read_csv('Data/VaryingPrices_StorageProfits_v2.csv',header=0,index_col=0)
cycleDf = pd.read_csv('Data/VaryingPrices_StorageCycles_v2.csv',header=0,index_col=0)

myDf = sizeDf # assume that columns are different simulations and rows are observations
x = myDf.columns.astype(float)
fs = 12

# Need to save the handles for the items that we want in the legend
# Because plotted main data with tiny opacity, will need to re-plot with black and save data
ax = plt.subplot()
plt.plot(x,myDf.transpose(),alpha=0.01, color='black')
dataHdl, = plt.plot(x,myDf.min(), color='black') # This will be covered up; we just use this for the legend
minHdl, = plt.plot(x,myDf.min(), color='red')
maxHdl, = plt.plot(x,myDf.max(), color='green')
medHdl, = plt.plot(x,myDf.median(), color='cyan',linewidth=2)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #label.set_fontname('Arial')
    label.set_fontsize(fs)

plt.xlabel('Cost of Reservoir Capacity ($/kWh/yr)',fontsize=fs+1)
plt.ylabel('Reservoir Size at Optimum (hours)',fontsize=fs+1)
plt.suptitle('Optimal Storage Size with Varying Battery Cost',fontsize=fs+2)
plt.legend([dataHdl,minHdl,medHdl, maxHdl],['Nodal data','Min value','Median value','Max value'],loc='upper right',fontsize=fs)
plt.savefig('Plots/VaryingPrices_storageCapacity.pdf',bbox_inches='tight')
#plt.savefig('kwhValue.png', dpi=300, bbox_inches='tight')

myDf = profitDf # assume that columns are different simulations and rows are observations
x = myDf.columns.astype(float)
fs = 12

# Need to save the handles for the items that we want in the legend
# Because plotted main data with tiny opacity, will need to re-plot with black and save data
ax = plt.subplot()
plt.plot(x,myDf.transpose(),alpha=0.01, color='black')
dataHdl, = plt.plot(x,myDf.min(), color='black') # This will be covered up; we just use this for the legend
minHdl, = plt.plot(x,myDf.min(), color='red')
maxHdl, = plt.plot(x,myDf.max(), color='green')
medHdl, = plt.plot(x,myDf.median(), color='cyan',linewidth=2)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #label.set_fontname('Arial')
    label.set_fontsize(fs)

plt.xlabel('Cost of Reservoir Capacity $/kWh/yr',fontsize=fs+1)
plt.ylabel('Profits after construction costs',fontsize=fs+1)
plt.suptitle('Operator Profits with Varying Battery Cost',fontsize=fs+2)
plt.legend([dataHdl,minHdl,medHdl, maxHdl],['Nodal data','Min value','Median value','Max value'],loc='upper right',fontsize=fs)
plt.savefig('Plots/VaryingPrices_storageProfit.pdf',bbox_inches='tight')
#plt.savefig('kwhValue.png', dpi=300, bbox_inches='tight')

## Plotting histogram of price at which storage system will no longer operate
# i.e. the cost at which no storage is built
# This should match our results from the earlier optimization (dispatch without sizing)

myDf = sizeDf
#myDf = sizeDf.iloc[0:9,:]

maxCost = pd.Series()

for thisRowName in myDf.index:
    thisRow = myDf.loc[thisRowName,:]
    maxCost[thisRowName] = float(thisRow[thisRow < 0.01].index[0])

#plt.hist(maxCost,40,histtype='bar',rwidth=0.75)
n, bins, patches = plt.hist(maxCost,40,normed=1,histtype='bar',rwidth=0.75, label= 'Nodal profits $/kWh/year, \n90% efficiency',linewidth=0, facecolor= (114/256.,147/256.,203/256.),)

plt.xlabel('Cost at which no storage is built ($/kWh)',fontsize=fs+1)
plt.ylabel('Density',fontsize=fs+1)
plt.suptitle('Distribution of market-exit prices for storage',fontsize=fs+2)
                            
## This matches the plot that we made from our previous research, to within rounding error introduced by our limited step size

## Plotting histogram of price at which 4 hours of storage is built
# this is relevant because many storage systems are targeting a 4hr duration

myDf = sizeDf
#myDf = sizeDf.iloc[0:9,:]

maxCost = pd.Series()

for thisRowName in myDf.index:
    thisRow = myDf.loc[thisRowName,:]
    maxCost[thisRowName] = float(thisRow[thisRow < 4].index[0])

n, bins, patches = plt.hist(maxCost,40,normed=1,histtype='bar',rwidth=0.75, label= 'Nodal profits $/kWh/year, \n90% efficiency',linewidth=0, facecolor= (114/256.,147/256.,203/256.),)

plt.xlabel('Storage cost at which 4hrs is optimal size',fontsize=fs+1)
plt.ylabel('Density',fontsize=fs+1)
plt.suptitle('Distribution of prices for 4hr storage',fontsize=fs+2)
                            
print("Min price at which 4hr storage is optimal: \t%s " % maxCost.min())
print("Avg price at which 4hr storage is optimal: \t%s " % maxCost.mean())
print("Median price at which 4hr storage is optimal: \t%s " % maxCost.median())
print("Max price at which 4hr storage is optimal: \t%s " % maxCost.max())

def efficiencySweep(thisSlice):
    # Simulation parameters
    myEfficiencies = [0.6,0.8,0.9]
    # myEfficiencies = np.arange(0.4,1.01,0.02)
    reservoirSize=1
    E_min = 0
    E_max = 1
    # The 1-hour system will be SOC constrained, rather than power constrained. We accordingly don't worry about P_max and P_min

    lastEfficiency = 0  # This is used to track whether the efficiency has switched
    storagePrice = 0 * simulationYears # Amortized cost of storage
    myLength = thisSlice.shape[1]


    # Result dataframe: Size, kwhPassed, and profits for each node, at each efficiency (columns)
    resultIndex = pd.MultiIndex.from_product([thisSlice.index,['storageSize','cycleCount','storageProfit']])
    results = pd.DataFrame(index = resultIndex, columns=myEfficiencies)
    powerOut = pd.DataFrame(index = thisSlice.index, columns = thisSlice.columns)

    # Build basic model, with everything except the state transition constraints
    # For each node,
    #  Set the cost function to be the prices for that period

    # For each efficiency,
    #  if the new efficiency is not the old efficiency:
    #    Add the state transition constraints with name 'chargeCons'
    #  Run the simulation
    #  Remove the state transition constraint

    model = CyClpSimplex()
    model.logLevel = 1
    x_var = model.addVariable('x',myLength*3+2)
    h_constant = sps.hstack( [1, sps.coo_matrix((1, myLength*3+1))] ) # Force h to be a specific size:         
    (A,b) = createABineq_noPowerConstraint(myLength, E_min, E_max)
    model.addConstraint(h_constant * x_var == reservoirSize,'fixedSize')
    model.addConstraint(         A * x_var <= b.toarray(),  'inequalities')


    #### LOOP THROUGH nodes
    for myNodeName in thisSlice.index:
        # Define cost function
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
                (eff_in, eff_out) = [np.sqrt(eff_round)] *2  # Properly account for the round trip efficiency of storage
                (A_eq, b_eq) = createABeq(myLength, delta_T, eff_in, eff_out)
                model.addConstraint(A_eq * x_var == b_eq.toarray(),'equalities')

            model.primal()  # Solve

            x = model.primalVariableSolution['x']
            results.loc[(myNodeName,'storageProfit'),eff_round] = np.dot(-c, x)   #Calculate profits at optimal storage level
            c_grid = x[2+myLength : 2+myLength*2]
            results.loc[(myNodeName,'cycleCount'),   eff_round] = sum(c_grid)*eff_in # net kWh traveled
            if eff_round == 0.9:
                powerOut.loc[myNodeName,:] = x[2+myLength : 2+myLength*2] + x[2+myLength*2 : 2+myLength*3]

            lastEfficiency = eff_round

        # Done with the loop; reverse the efficiency set and move on to the next node
        myEfficiencies = myEfficiencies[::-1]

    return (results,powerOut)

## CUSTOM START/END DATE
startDate = parser.parse('01/01/12 00:00')  # year starts at 2013-01-01 00:00:00
endDate = parser.parse('12/31/12 23:00')  # year ends at 2013-12-31 23:00:00
startDate = pytz.timezone('America/Los_Angeles').localize(startDate).astimezone(pytz.utc)
endDate = pytz.timezone('America/Los_Angeles').localize(endDate).astimezone(pytz.utc)

# ## FULL DATASET
# startDate = APNode_Prices.columns.values[ 0].astype('M8[m]').astype('O') # Convert to datetime, not timestamp
# endDate   = APNode_Prices.columns.values[-1].astype('M8[m]').astype('O')
# startDate = pytz.utc.localize(startDate)
# endDate   = pytz.utc.localize(endDate)

timespan = relativedelta.relativedelta(endDate +timestep, startDate)
simulationYears = timespan.years + timespan.months/12. + timespan.days/365. + timespan.hours/8760.  # Leap years will be slightly more than a year, and that's ok.

startNode = 0
stopNode  = 0 # if set to zero, then will loop through all nodes
if ((stopNode == 0)|(stopNode > APNode_Prices.shape[0])): stopNode = APNode_Prices.shape[0]
thisSlice = APNode_Prices.ix[startNode:stopNode,startDate:endDate]

import multiprocessing

# Split dataset into roughly even chunks
j = min(multiprocessing.cpu_count(),10)
# chunksize = (APNode_Prices.shape[0]/j)+1
# splitFrames = [df for g,df in APNode_Prices.groupby(np.arange(APNode_Prices.shape[0])//chunksize)]
chunksize = (thisSlice.shape[0]/j)+1
splitFrames = [df for g,df in thisSlice.groupby(np.arange(thisSlice.shape[0])//chunksize)]

print("Entering the pool... bye-bye!")
solverStartTime = time.time()

pool = multiprocessing.Pool(processes = j)
resultList = pool.map(efficiencySweep,splitFrames) # Each worker returns a tuple of (result,PowerOut)

(resultFrames, powerOutputs) = zip(*resultList)

results = pd.concat(resultFrames).sort_index()
powerResults = pd.concat(powerOutputs).sort_index()

profitDf = results.loc[(slice(None),'storageProfit'),:].reset_index(level=1,drop=True)
cycleDf  = results.loc[(slice(None),'cycleCount'),:].reset_index(level=1,drop=True)

profitDf.to_csv('Data/kwhValue_step_02.csv')
cycleDf.to_csv('Data/cycleCount_step_02.csv')
powerResults.to_csv('Data/powerOutput_90pct.csv')

print('Total function call time: %.3f seconds' % (time.time() - solverStartTime))

###### SWEEP EFFICIENCY, HOLD NODE CONSTANT ######


###### SWEEP NODES, HOLD EFFICIENCY CONSTANT #####

# Approach:
# Set a multiplier alpha
# This modifies the cost matrix
# Loop through all nodes
#  Loop through the alpha values

startDate = parser.parse('01/01/13 00:00')  # year starts at 2013-01-01 00:00:00
endDate = parser.parse('12/31/13 23:00')  # year ends at 2013-12-31 23:00:00

timespan =  endDate + timestep - startDate  # Length of time series (integer)
simulationYears = (timespan.days*24 + timespan.seconds/3600) / 8760.  # Leap years will be slightly more than a year, and that's ok.
myLength = APNode_Prices.columns.get_loc(endDate) - APNode_Prices.columns.get_loc(startDate)+1

eff_round = 0.9  # Round-trip efficiency
(eff_in, eff_out) = [np.sqrt(eff_round)] *2  # Properly account for the round trip efficiency of storage

E_min = 0
E_max = 1
P_max = E_max/eff_in # Max discharge power at grid intertie, e.g max limit for C_i
P_min = -1*E_max/eff_out # Max charge power at grid intertie, e.g. min D_i

# Create clean problem matrices - first specify efficiency!
(A, b, A_eq, b_eq) = createABMatrices(myLength, delta_T, eff_in, eff_out, P_min, P_max, E_min, E_max)
# Define model:
coolStart = CyClpSimplex()
x_var = coolStart.addVariable('x',myLength*3+2)
# Add constraints to model:

# Fix reservoir size
reservoirSize=1
h_constant = sps.hstack( [1, sps.coo_matrix((1, myLength*3+1))] )
A_eq = sps.vstack( [h_constant, A_eq ] )
b_eq = sps.vstack( [reservoirSize, b_eq] )

coolStart += A * x_var <= b.toarray()
coolStart += A_eq * x_var == b_eq.toarray()

## LOOP BLOCK- loops through nodes, and then for each node loops through each price

alphaSet = [0,0.01,0.02,0.05,0.1,0.15,0.2]

startNode = 0
stopNode = 0 # if set to zero, then will loop through all nodes
if ((stopNode == 0)|(stopNode > APNode_Prices.shape[0])): stopNode = APNode_Prices.shape[0]

# Initializing variables for loop
myStoragePrice = 0
badNodes = []
sizeDf = pd.DataFrame()
profitDf = pd.DataFrame()
cycleDf = pd.DataFrame()

for i in range(startNode,stopNode):
    ## Set up prices
    myNodeName = APNode_Prices.index[i]
    energyPrice = APNode_Prices.loc[myNodeName,startDate:endDate] / 1000.0 # Price $/kWh as array
    # Deal with NaN prices
    nodePriceNanCount = sum(np.isnan(energyPrice))
    if (0.05 < (nodePriceNanCount/myLength)):  # Set cutoff threshold here
        badNodes.push(i) # Add this to the list of bad nodes
        continue # Jump to the next node
    nans, x = nan_helper(energyPrice)
    energyPrice[nans] = np.interp(x(nans), x(~nans), energyPrice[~nans])

    sweepStartTime = time.time()

    for alpha in alphaSet:
        c = np.concatenate([[myStoragePrice],[0.0]*(myLength+1),(1+alpha)*energyPrice,(1-alpha)*energyPrice],axis=0)  #placeholder; No cost for storage state; charged for what we consume, get paid for what we discharge
        c_clp = CyLPArray(c)
        coolStart.objective = c_clp * x_var

        # Run the model
        startTime = time.time()
        coolStart.primal()
        endTime = time.time()

        # Results- Rows are Nodes, indexed by name. Columns are Storage price, indexed by price
        x_out = coolStart.primalVariableSolution['x']
        sizeDf.loc[myNodeName,str(alpha)]   = x_out[0]
        profitDf.loc[myNodeName,str(alpha)] = np.dot(-c, x_out)
        cycleDf.loc[myNodeName,str(alpha)]  = sum(x_out[2+myLength : 2+myLength*2]) * eff_in
#         print("Finished with initial solve in \t%.4f s for node %s" % ((endTime-startTime),myNodeName))
        sys.stdout.flush()
    
    sweepTime = time.time()-sweepStartTime
    print("Finished %s computations in \t%.4f s for node %s \t(%.4f s/solve)" % (len(alphaSet), sweepTime, myNodeName, sweepTime/(len(storagePriceSet)-1) ))
    sys.stdout.flush()
    storagePriceSet = storagePriceSet[::-1] # Reverse the price set so that we start from the same price for the next node to make warm-start more effective

    if (i % 10 == 0): # Save our progress along the way
        sizeDf.to_csv('Data/VaryingAlpha_StorageSizing_v2.csv')
        profitDf.to_csv('Data/VaryingAlpha_StorageProfits_v2.csv')
        cycleDf.to_csv('Data/VaryingAlpha_StorageCycles_v2.csv')
    
#clear_output()
print("Finished; writing all to file...")
sizeDf.to_csv('Data/VaryingAlpha_StorageSizing_v2.csv')
profitDf.to_csv('Data/VaryingAlpha_StorageProfits_v2.csv')
cycleDf.to_csv('Data/VaryingAlpha_StorageCycles_v2.csv')

myDf = profitDf
x = alphaSet

ax = plt.subplot()
# plt.plot(x,myDf.transpose(),alpha=0.01, color='black')
dataHdl, = plt.plot(x,myDf.min(), color='black') # This will be covered up; we just use this for the legend
minHdl, = plt.plot(x,myDf.min(), color='red')
maxHdl, = plt.plot(x,myDf.max(), color='green')
medHdl, = plt.plot(x,myDf.median(), color='cyan',linewidth=2)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #label.set_fontname('Arial')
    label.set_fontsize(fs)

plt.xlabel('Price sensitivity alpha',fontsize=fs+1)
plt.ylabel('Profits after construction costs',fontsize=fs+1)
plt.suptitle('Operator Profits with Varying Price Sensitivity',fontsize=fs+2)
plt.legend([dataHdl,minHdl,medHdl, maxHdl],['Nodal data','Min value','Median value','Max value'],loc='upper right',fontsize=fs)
# plt.savefig('Plots/VaryingAlpha_storageProfit.pdf',bbox_inches='tight')
#plt.savefig('kwhValue.png', dpi=300, bbox_inches='tight')



