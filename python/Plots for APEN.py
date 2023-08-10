import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
import numpy as np
import os, sys
from datetime import *
from dateutil import parser
#import statsmodels.api as sm
#import statsmodels.formula.api as smf

get_ipython().magic('matplotlib inline')
from IPython.display import clear_output

print("Current environment directory:" + sys.prefix)
print("System version: "+sys.version)

fs = 14
plt.rc('font',family='Times New Roman')
fn = 'Times New Roman'

# Load price data - this will be one row of data with headers
priceDf = pd.read_csv('Data/inputData/BARRY_6_N001_only_2013.csv',index_col=0,parse_dates=True)
priceDf.columns = pd.DatetimeIndex(priceDf.columns) # convert to an indexable time
startDate = parser.parse('08/01/13 00:00')
endDate = parser.parse('08/03/13 06:00')

# These data run off data created by 'Implementing CyLP'.Expecting a file with raw data, no headers or index names
chargeDf = pd.read_csv('Data/chargeStates_efficiency.csv', header=None) # this is currently just for the range 08-01-13 00:00 to 08/03/13 06:00
chargeDf.columns = pd.date_range(startDate, endDate, freq='H')
chargeDf.index = [0.6,0.8,0.9]
mergeDf = pd.concat([priceDf,chargeDf],axis=0)
mergeDf = mergeDf.loc[:,startDate:endDate]
mergeDf = mergeDf.transpose()

plotDf = mergeDf
fig, ax1 = plt.subplots(figsize=(8,4))
energyPriceColor = (0.3,0.3,0.3)
myChargeColor = (57/256.0, 106/256.0, 177/256.0)

hdl1 = ax1.plot(plotDf.index,plotDf['BARRY_6_N001'], color=energyPriceColor, linewidth=2, label='Nodal LMP')    
# Charge behavior- second axis
ax2 = ax1.twinx()
hdl2 = ax2.plot(plotDf.index, plotDf[0.6], '--', color=myChargeColor, linewidth=2, label='60% Efficient')
hdl3 = ax2.plot(plotDf.index, plotDf[0.8], ':', color=myChargeColor,  linewidth=2, label='80% Efficient')

# Titles
plt.suptitle('Charge/Discharge Behavior: Varying Reservoir Size', fontsize=fs+2)
ax1.set_ylabel('Energy Price ($/MWh)', fontsize=fs+1) #, color='k')
ax2.set_ylabel('Charge Level (% of Capacity)', fontsize=fs+1, color=myChargeColor)

# Datetime ticks
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
myLocator = DayLocator()
ax1.xaxis.set_major_locator(myLocator)
ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))

# Labels and limits
ax1.set_xlabel('Time', fontsize=fs+1)
plotStartDate = parser.parse('07/31/13 18:00')
plotEndDate = parser.parse('08/03/13 14:00')
ax1.set_xlim([plotStartDate, plotEndDate])
ax1.set_ylim([20,70])
ax2.set_ylim([-0.1,1.1])

# # Plot arrows
oneHour = timedelta(hours=1)
# Arrow for energy price
startPltDate = pltdates.date2num(startDate-oneHour)
headLength = 0.05
ax1.arrow(startPltDate, plotDf.loc[startDate,'BARRY_6_N001'], 
          pltdates.date2num(plotStartDate) - startPltDate + headLength*1.3, 0,
          head_width=2, head_length=headLength, fc=energyPriceColor, ec=energyPriceColor, linewidth=1.5)
# Arrow for charge status
startPltDate = pltdates.date2num(endDate+oneHour)
headLength = 0.05
ax2.arrow(startPltDate, plotDf.loc[endDate,0.6], 
          pltdates.date2num(plotEndDate) - startPltDate - headLength*1.3, 0,
          head_width=0.04, head_length=headLength, fc=myChargeColor, ec=myChargeColor, linewidth=1.5)



# Legend
handles = hdl1+hdl2+hdl3
labels = [l.get_label() for l in handles]
plt.legend(handles, labels,loc='upper right')

# Set colors for the axis ticks
ax1.spines['right'].set_color(myChargeColor)
ax2.spines['right'].set_color(myChargeColor)
for tl in ax2.get_yticklabels():
    tl.set_color(myChargeColor)

plt.savefig('Plots/chargeValidation_varyingEfficiency.pdf',bbox_inches='tight')

# These data run off data created by 'Implementing CyLP'.Expecting a file with raw data, no headers or index names
chargeDf = pd.read_csv('Data/chargeStates_varySize.csv', header=None).transpose() # this is currently just for the range 08-01-13 00:00 to 08/03/13 06:00
chargeDf.columns = pd.date_range(startDate, endDate, freq='H')
chargeDf.index = [1,2,6] # Check the block under 'Validation: plotting charge status for different reservoir sizes' in Implementation...ipynb to get the corresponding set
mergeDf = pd.concat([priceDf,chargeDf],axis=0)
mergeDf = mergeDf.loc[:,startDate:endDate]
mergeDf = mergeDf.transpose()

plotDf = mergeDf
fig, ax1 = plt.subplots(figsize=(8,4))
myChargeColor = (57/256.0, 106/256.0, 177/256.0)

hdl1 = ax1.plot(plotDf.index,plotDf['BARRY_6_N001'], color=(0.3,0.3,0.3), linewidth=2, label='Nodal LMP')    
# Charge behavior- second axis
ax2 = ax1.twinx()
hdl2 = ax2.plot(plotDf.index, plotDf[2], '--', color=myChargeColor, linewidth=2, label='2-hour Reservoir')
hdl3 = ax2.plot(plotDf.index, plotDf[6], ':', color=myChargeColor,  linewidth=2, label='6-hour Reservoir')

# Titles
plt.suptitle('Charge/Discharge Behavior: Varying Reservoir Size', fontsize=fs+2)
ax1.set_ylabel('Energy Price ($/MWh)', fontsize=fs+1) #, color='k')
ax2.set_ylabel('Charge Level (% of Capacity)', fontsize=fs+1, color=myChargeColor)

# Datetime ticks
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
myLocator = DayLocator()
ax1.xaxis.set_major_locator(myLocator)
ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))

# Labels and limits
plotStartDate = parser.parse('07/31/13 18:00')
plotEndDate = parser.parse('08/03/13 14:00')
ax1.set_xlim([plotStartDate, plotEndDate])
startPltDate = pltdates.date2num(startDate-oneHour)

ax1.set_xlabel('Time', fontsize=fs+1)
# ax1.set_xlim(['07/31/13 18:00','08/03/13 13:00'])
ax1.set_ylim([20,70])
ax2.set_ylim([-0.1,1.1])

# # Plot arrows
oneHour = timedelta(hours=1)
# Arrow for energy price
startPltDate = pltdates.date2num(startDate-oneHour)
headLength = 0.05
ax1.arrow(startPltDate, plotDf.loc[startDate,'BARRY_6_N001'], 
          pltdates.date2num(plotStartDate) - startPltDate + headLength*1.3, 0,
          head_width=2, head_length=headLength, fc=energyPriceColor, ec=energyPriceColor, linewidth=1.5)
# Arrow for charge status
startPltDate = pltdates.date2num(endDate+oneHour)
headLength = 0.05
ax2.arrow(startPltDate, plotDf.loc[endDate,2], 
          pltdates.date2num(plotEndDate) - startPltDate - headLength*2, 0,
          head_width=0.04, head_length=headLength, fc=myChargeColor, ec=myChargeColor, linewidth=1.5)
# Legend
handles = hdl1+hdl2+hdl3
labels = [l.get_label() for l in handles]
plt.legend(handles, labels,loc=(0.7,0.38))

# Set colors for the axis ticks
ax1.spines['right'].set_color(myChargeColor)
ax2.spines['right'].set_color(myChargeColor)
for tl in ax2.get_yticklabels():
    tl.set_color(myChargeColor)

plt.savefig('Plots/chargeValidation_varyingSize.pdf',bbox_inches='tight')

# Nodes in rows, scenarios in columns
sizeDf = pd.read_csv('Data/VaryingPrices_StorageSizing_v2.csv',header=0,index_col=0)
profitDf = pd.read_csv('Data/VaryingPrices_StorageProfits_v2.csv',header=0,index_col=0)
cycleDf = pd.read_csv('Data/VaryingPrices_StorageCycles_v2.csv',header=0,index_col=0)
print("%s nodes and %s scenarios"%(sizeDf.shape[0], sizeDf.shape[1]))

myDf = sizeDf # assume that columns are different simulations and rows are observations
x = myDf.columns.astype(float)
# fs = 12

# Need to save the handles for the items that we want in the legend
# Because plotted main data with tiny opacity, will need to re-plot with black and save data
ax = plt.subplot()
plt.plot(x,myDf.transpose(),alpha=0.01, color='black')  # Plots all the data with low opacity
dataHdl, = plt.plot(x,myDf.min(), color='black') # This will be covered up; we just use this for the legend
minHdl, = plt.plot(x,myDf.min(), color='red')
maxHdl, = plt.plot(x,myDf.max(), color='green')
medHdl, = plt.plot(x,myDf.median(), color='cyan',linewidth=2)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(fs)

ax.set_ylim([-1,13])
plt.xlabel('Cost of Reservoir Capacity ($/kWh/yr)',fontsize=fs+1)
plt.ylabel('Reservoir Size at Optimum (hours)',fontsize=fs+1)
plt.suptitle('Optimal Storage Size with Varying Battery Cost',fontsize=fs+2)
plt.legend([dataHdl,minHdl,medHdl, maxHdl],['Nodal data','Min value','Median value','Max value'],loc='upper right',fontsize=fs)
# plt.savefig('Plots/VaryingPrices_storageCapacity.eps', format='eps',bbox_inches='tight')
plt.savefig('Plots/VaryingPrices_storageCapacity.pdf',bbox_inches='tight')
#plt.savefig('kwhValue.png', dpi=300, bbox_inches='tight')

resultDf.head()

resultDf = pd.read_csv('Data/SweepStorageSize_ForPlotting.csv')
## Plot Results ## 
fs = 12

plt.rc('text',usetex='true')

plt.plot(resultDf['storageSize'],resultDf['storageProfit'])
plt.xlabel('Reservoir Size ($h$)', fontsize=fs+1)
plt.ylabel('Long-run annual profits ($/kWh/yr)', fontsize=fs+1)
plt.suptitle('Profits with Varying Reservoir Sizes', fontsize= fs+2)

plt.rc('text',usetex='false')
plt.savefig('Plots/Profit_VaryingSize.pdf',bbox_inches='tight')

ax = plt.subplot()
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #label.set_fontname('Arial')
    label.set_fontsize(fs)

def plotEfficiencyGraph(myDf, eff):
    minVal = myDf.min()
    maxVal = myDf.max()
    #meanVal = myDf.mean()
    medVal = myDf.median()

    # Need to save the handles for the items that we want in the legend
    # Because plotted main data with tiny opacity, will need to re-plot with black and save data
    ax = plt.subplot()
    plt.plot(eff,myDf.transpose(),alpha=0.01, color='black')
    dataHdl, = plt.plot(eff,minVal, color='black', label='Nodal data') # This will be covered up; we just use this for the legend
    minHdl, = plt.plot(eff,minVal, color='red', label='Min value')
    maxHdl, = plt.plot(eff,maxVal, color='green', label='Max value')
    medHdl, = plt.plot(eff,medVal, color='cyan',linewidth=2, label='Median value')

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        #label.set_fontname('Arial')
        label.set_fontsize(fs)
        
    plt.legend([dataHdl,minHdl,medHdl, maxHdl],['Nodal data','Min value','Median value','Max value'],fontsize=fs+1)

    return (ax, plt)


profitFileName = 'Data/kwhValueAggregated_step_02.csv' 

kwhValue = pd.read_csv(profitFileName, header=None)
if np.isnan(kwhValue.iloc[0,0]):
    kwhValue = pd.read_csv(profitFileName,header=0,index_col=0)
    
eff = range(40, 102, 2)

ax,plt = plotEfficiencyGraph(kwhValue, eff)

#meanHdl, = plt.plot(eff,meanVal, color = 'cyan')
plt.xlabel('Round-trip efficiency (%)',fontsize=fs+1)
plt.ylabel('Annual trading profits ($/kWh/year)',fontsize=fs+1)
plt.suptitle('Annual short-run trading profits per kWh',fontsize=fs+2)
plt.legend(loc=(.16,.63))
plt.savefig('Plots/varyefficiency_kwhValue.pdf',bbox_inches='tight')
#plt.savefig('kwhValue.png', dpi=300, bbox_inches='tight')

kwhValue.head()

cycleCount = pd.read_csv('Data/cycleCount_step_02.csv', header=None)

ax,plt = plotEfficiencyGraph(cycleCount, eff)

d365, = plt.plot(eff,365*np.ones((31,1)),color="gray", linestyle='--')
d730, = plt.plot(eff,730*np.ones((31,1)),color="gray", linestyle=':')

plt.xlabel('Round-trip efficiency (%)',fontsize=fs+1)
plt.ylabel('Annual cycle count (Cycles/Year)',fontsize=fs+1)
plt.suptitle('   Number of cycles per year with varying efficiencies',fontsize=fs+2)

handles, labels = ax.get_legend_handles_labels()
handles.extend([d365, d730])
labels.extend(['1x daily','2x daily'])
legend = plt.legend(handles,labels,loc='upper left',fontsize=fs+1)

plt.savefig('Plots/varyefficiency_cycleCount.pdf',bbox_inches='tight')
#plt.savefig('cycleCount.pdf', bbox_inches='tight')

cycleValue = pd.read_csv('Data/cycleValue_step_02.csv', header=None)

ax,plt = plotEfficiencyGraph(cycleValue, eff)

plt.xlabel('Round-trip efficiency (%)',fontsize=fs+1)
plt.ylabel('Average short-run trading profits \nper cycle ($/kWh/1000 Cycles)',fontsize=fs+1)
plt.suptitle('Profit per cycle with varying efficiency',fontsize=fs+2)
plt.legend(loc='upper right')

plt.savefig('Plots/varyefficiency_cycleProfit.pdf', bbox_inches='tight')

kwhValue = pd.read_csv('Data/kwhValue_step_02.csv', header=None)
n, bins, patches = plt.hist(kwhValue.iloc[:,25],40,normed=1,histtype='bar',rwidth=0.75, label= 'Profits at LMP nodes, \n90% efficiency', linewidth=0, facecolor= (114/256.,147/256.,203/256.),)

import pylab as P

x = np.linspace(6,18,200)

# y = P.normpdf(x,9.18083, 1.68882)
y = plt.mlab.normpdf(x,9.18083, 1.68882)


ax = plt.subplot()
normHdl, = plt.plot(x, y, linewidth=2, color=(211/256.,94/256.,96/256.),label='Best fit, normal distribution')

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #label.set_fontname('Arial')
    label.set_fontsize(fs)

## Following code chunk was taken from http://stackoverflow.com/questions/1289681/drawing-braces-with-pyx    
def half_brace(x, beta):
    x0, x1 = x[0], x[-1]
    y = 1/(1.+np.exp(-1*beta*(x-x0))) + 1/(1.+np.exp(-1*beta*(x-x1)))
    return y

xmax, xstep = 4, .01
xaxis = np.arange(0, xmax/2, xstep)
y0 = half_brace(xaxis, 20.)
y = np.concatenate((y0, y0[::-1]))

plt.plot(np.arange(0,xmax,xstep)+14,y*.05+0.1,'black')
plt.figtext(.77,0.42,'Highest-profit \narbitrage nodes',fontsize = fs+1,horizontalalignment='center')
## end of code chunk

plt.xlabel('Annual short-run trading profits ($/kWh/year)',fontsize=fs+1)
plt.ylabel('Density',fontsize=fs+1)
plt.suptitle('Distribution of nodal short-run trading profits',fontsize=fs+2)
plt.legend(loc='upper right',fontsize=fs)
plt.savefig('Plots/ProfitDistribution.eps', type='eps', bbox_inches='tight')

#plt.setp(patches,'facecolor','g','alpha',0.2)
plt.savefig('Plots/ProfitDistribution.pdf', bbox_inches='tight')

# sizeDf = pd.read_csv('Data/VaryingAlpha_StorageSizing_v2.csv')
profitDf = pd.read_csv('Data/VaryingAlpha_StorageProfits_v2.csv', header=0, index_col=0)
# cycleDf = pd.read_csv('Data/VaryingAlpha_StorageCycles_v2.csv')

myDf = profitDf
alphaSet = myDf.columns.values.astype('float64')

ax,plt = plotEfficiencyGraph(myDf, alphaSet)

ax.set_xlim([alphaSet[0],alphaSet[-1]])
ax.set_ylim([0,20])

plt.legend(loc="upper right", ncol=2)  # Legend location, either: "upper right", "best", or tuple for positioning of the bottom-left corner in (x,y)

#meanHdl, = plt.plot(eff,meanVal, color = 'cyan')
plt.xlabel(r'Price Sensitivity $\alpha$',fontsize=fs+1)
plt.ylabel('Annual trading profits ($/kWh/year)',fontsize=fs+1)
plt.suptitle('Sensitivity of Results to Price-Taker Assumption',fontsize=fs+2)
plt.savefig('Plots/Profits_VaryingAlpha.pdf',bbox_inches='tight')
# #plt.savefig('kwhValue.png', dpi=300, bbox_inches='tight')

profitDf.describe().loc[['min','50%','max'],:]

from dateutil import parser, relativedelta

# import price data as a dataframe: columns are times, rows are nodes.  Size is nodes x 8760
APNode_Prices = pd.read_csv("/Users/emunsing/GoogleDrive/CE 290 Project/Data Collection/Prices/R code/All_PNodes_MCC_Aggregated_2013short.csv",
                      header=0,index_col=0)#,nrows=10)
APNode_Prices.columns = pd.DatetimeIndex(APNode_Prices.columns)
timestep = relativedelta.relativedelta(APNode_Prices.columns[2],APNode_Prices.columns[1])
delta_T = timestep.hours  # Time-step in hours

APNode_Prices.head()

avg_mcc = APNode_Prices.mean(axis=1)
avg_mcc.name = 'mcc'

nodes = pd.read_csv('Data/LMP_PointMap_Nodes.csv',index_col=0)
nodes.head()
export = nodes.join(avg_mcc,how='inner')
export.index.name = 'name'
export['mcc'].to_csv('Data/avg_mcc.csv')

export.head()

