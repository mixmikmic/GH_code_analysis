import pandas as pd
import os, sys, time, glob, pickle

print("Current environment directory:" + sys.prefix)
print("System version: "+sys.version)

# Use this if we want to first set the input file to be used
os.environ['INPUTFILE'] = "inputData/pricedata_LMP.csv"

get_ipython().system(' echo $INPUTFILE')

try:
    fname = os.environ['INPUTFILE']
except KeyError:
    fname = "inputData/pricedata_LMP_100.csv" # Only 100 nodes    

APNode_Prices = pd.read_csv( fname, header=0,index_col=0)#,nrows=10)
goodNodes = (APNode_Prices.isnull().sum(axis=1) < (0.02 * APNode_Prices.shape[1])) # True if node is less than x% NaN values
APNode_Prices = APNode_Prices[goodNodes]
    
allNodeNames = APNode_Prices.index.values

# Read all the dataframes, and return a dataframe from all the dataframes together
allFiles = glob.glob('Data/efficiencyResults*temp.csv')
dfList = []
for fname in allFiles:
    df = pd.read_csv(fname, index_col=0, header=0, usecols=[0,1,2,3]) # Just hold on to enough data to check whether they are completed
    dfList.append(df)
allDfs = pd.concat(dfList)

completed = allDfs.iloc[:,-1].dropna()
uniqueCompleteNodes = completed.index.unique().values
remainingNodes = list(set(allNodeNames) - set(uniqueCompleteNodes))
remainingNodes.sort()

with open('nodeList.pkl','wb') as f:
    f.write(pickle.dumps(remainingNodes))
os.environ['NODELIST'] = "TRUE"  # This will lead efficiencySweep to work from this update node list
    
print("%s nodes remaining uncompleted" %len(remainingNodes))

# End conditions: 
#  - List of Uncompleted node name strings are stored in nodeList.pkl
#  - Environment nodeList flag is set (efficiencySweep will use this next time)

#  - 

# Just check the list of unprocessed nodes
with open('nodeList.pkl','rb')as f:
    a = pickle.loads(f.read())

print("%s nodes remaining uncompleted" %len(remainingNodes))

allFiles = glob.glob('Data/efficiencyResults*temp.csv')
dfList = []
for fname in allFiles:
    df = pd.read_csv(fname, index_col=[0,1], header=0) # Just hold on to enough data to check whether they are completed
    dfList.append(df)
allDfs = pd.concat(dfList)
allDfs.sort_index(inplace=True)

results = allDfs.dropna().drop_duplicates()  #Empirically, dropping duplicate rows should produce unique entries for each node.

profitDf = results.loc[(slice(None),'storageProfit'),:].reset_index(level=1,drop=True)
cycleDf  = results.loc[(slice(None),'cycleCount'),:].reset_index(level=1,drop=True)

profitDf.to_csv('Data/kwhValueAggregated_step_02.csv')
cycleDf.to_csv('Data/cycleCountAggregated_step_02.csv')

powerResults.to_csv('Data/powerOutput_90pct.csv')

temp = pd.read_csv('Data/kwhValue_step_02.csv')
temp.head()

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.plot(profitDf.values)

complete = allDfs.dropna()
complete = complete.drop_duplicates()

complete.shape

uniqueCompleteNodes.values.shape

duplicateNodes = list(set(completeNotUnique) - set(uniqueCompleteNodes))

len(set(completeNotUnique))

allDfs.loc[('ADCC_2_N001',slice(None)),:]

tempDf = pd.DataFrame(zip(*completeNotUnique)).transpose()
tempDf.head()

justProfit = completed.loc[:,:]

uniqueCompleteNodes.shape

temp = allDfs.loc[uniqueCompleteNodes,:]
temp.shape

completed.head()

uniqueCompleteNodes[0:10]

temp = "Data/powerOut_90pct.csv"
df = pd.read_csv(fname, index_col=[0,1],header=0)
df.head()

fname = 'Data/efficiencyResults_pid32418temp.csv'
df = pd.read_csv(fname, index_col=[0,1], header=0)
df.head()

fname = 'Data/efficiencyPower_pid32418temp.csv'
df = pd.read_csv(fname, index_col=[0,1], header=0)
df.head()

allFiles = glob.glob('Data/efficiencyResults*temp.csv')
dfList = []
for fname in allFiles:
    df = pd.read_csv(fname, index_col=0, header=0, usecols=[0,1,2,3]) # Just hold on to enough data to check whether they are completed
    dfList.append(df)
allDfs = pd.concat(dfList)

completed = allDfs.iloc[:,-1].dropna()
uniqueCompleteNodes = completed.index.unique().values



