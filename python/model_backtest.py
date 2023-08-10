import pandas as pd
import numpy as np
import datetime as dt
import os

# Set Start & Finish dates for Backtest period
startDate = dt.datetime(2001, 6, 8)
finishDate = dt.datetime(2015, 5, 4)

# Load daily closing prices for SPY, using'Date' column as DataFrame index & parsing trade dates to datetime objects 
data = pd.read_csv(r'price_data\SPY.csv',index_col="Date",parse_dates=True)

# define a specic dataframe
df_SP = pd.DataFrame(data['SPY'][(data.index >= startDate) & (finishDate >= data.index)])
df_SP.head()

# Calculate daily % returns of SPY and show the cumulative (compounded) returns over time 
df_SP['% Change'] = df_SP['SPY'].pct_change()
df_SP['cum_rets'] = (1 + df_SP['SPY'].pct_change()).cumprod()

df_SP.head()

# Calculate Annualized SPY returns & volatility from "Buy & Hold" daily % returns 
SPY_mean = df_SP['% Change'].mean()*252
SPY_vol = df_SP['% Change'].std()*np.sqrt(252)

# Calculate Annualized Sharpe ratio for "Buy & Hold" SPY strategy
SPY_sharpe = SPY_mean / SPY_vol

# Output Annualized returns & Sharpe ratio
print('SPY(Buy & Hold) mean: ', SPY_mean, '\nSPY(Buy & Hold) sharpe: ', SPY_sharpe)

df_SP['cum_rets'].describe()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 12, 12

ax1 = plt.subplot(211)
df_SP['cum_rets'].plot(ax=ax1)
ax1.set_ylabel('portfolio value')
plt.title('SPY Buy & Hold')

import pandas_datareader as web

#get the risk free rate (3 Month T-Bills)
rf_yearly = pd.DataFrame(web.DataReader("TB3MS", "fred", start=startDate, end=finishDate)['TB3MS'])

rf_yearly['TB3MS'] = rf_yearly['TB3MS'] - 0.30
rf_d = (rf_yearly.asfreq('D').fillna(method='pad'))/(365*100)

rf = rf_d.iloc[2:].resample('B').ffill()

rf.head()

# Calculate daily % returns of risk-free rate (3 Month T-Bills - 30 bps) & show cumulative (compounded) returns over time 
rf['% Change'] = (1 + rf['TB3MS'].iloc[1:]).cumprod()

rf.head()

ax1 = plt.subplot(211)
df_SP['cum_rets'].plot(ax=ax1, label='SPY')
ax1.set_ylabel('portfolio value')
plt.title('SPY (Buy & Hold) vs Cash')

ax2 = plt.subplot(211)
rf['% Change'].plot(ax=ax2, label='cash')

plt.legend(loc=0)

def calculate_mean_sharpe(mycol):
    mult = 8
    mypos = (mycol * mult).round(1).clip(-0.5,1.5)
    prctgChange = df_SP['% Change'] * mypos.shift(1)
    cumChange = (1 + prctgChange).cumprod()
    cumChange.head()
    simp_mean = prctgChange.mean()*252
    simp_vol = prctgChange.std()*np.sqrt(252)
    simp_sharpe = simp_mean / simp_vol
    return pd.Series({'mean': simp_mean, 'sharpe': simp_sharpe})

path = 'Predicted_Return\\'
folders = os.listdir(path)
folderDict = {}
for folder in folders:
    filenames = os.listdir(os.path.join(path, folder))
    alldf = pd.DataFrame()
    for filename in filenames:
        file = os.path.join(path, folder, filename)
        method = filename.replace('PredReturn_', '').replace('.csv', '')
        if method in ['simpLR', 'LR_Trans']:
            mydf = pd.read_csv(file, names=['Date',method],index_col='Date',parse_dates=True)
        else:
            mydf = pd.read_csv(file)
            mydf = mydf.set_index(mydf.columns.values[0])
            mydf.columns = [method + '_' + x for x in mydf.columns]
        alldf = pd.concat((alldf, mydf), axis=1)
        alldf = alldf[(alldf >= startDate) & (alldf <= finishDate)]
    folderResults = alldf.apply(calculate_mean_sharpe, 0).transpose()
    folderDict[folder] = folderResults
    
finaldf = pd.concat( list(folderDict.values()), axis = 1, keys=list(folderDict.keys())) 
finaldf.head()

finaldf

def plot_return(folderIndx, modelIndx, versionIndx):
    folders = os.listdir(path)
    folder = folders[folderIndx]
    filenames = os.listdir(os.path.join(path,folder))
    filename = filenames[modelIndx]
    
    file = os.path.join(path, folder, filename)
    method = filename.replace('PredReturn_', '').replace('.csv', '')
    if method in ['simpLR', 'LR_Trans']:
        mydf = pd.read_csv(file, names=['Date',method],index_col='Date',parse_dates=True)
    else:
        mydf = pd.read_csv(file)
        mydf = mydf.set_index(mydf.columns.values[0])
        mydf.columns = [method + '_' + x for x in mydf.columns]
    mydf = mydf[(mydf >= startDate) & (mydf <= finishDate)]
    version = mydf.columns[versionIndx]
    mycol = mydf.iloc[:,versionIndx]
    mult = 8
    mypos = (mycol * mult).round(1).clip(-0.5,1.5)
    prctgChange = df_SP['% Change'] * mypos.shift(1)
    
    # OUTPUT TIME SERIES OF DAILY RETURNS (Additional check for data & calculation accuracy)
    prctgChange.to_csv(r'price_data\test.csv')    
    
    cumChange = (1 + prctgChange).cumprod()
    cumChange.head()
    simp_mean = prctgChange.mean()*252
    simp_vol = prctgChange.std()*np.sqrt(252)
    simp_sharpe = simp_mean / simp_vol
    
    ax1 = plt.subplot(211)
    df_SP['cum_rets'].plot(ax=ax1, label='SPY')
    ax1.set_ylabel('portfolio value')
    plt.title('SPY (Buy & Hold) vs {0} ({1})'.format(version,folder))

    ax2 = plt.subplot(211)
    rf['% Change'].plot(ax=ax2, label='cash')

    ax3 = plt.subplot(211)
    cumChange.plot(ax=ax3, label=method)
    plt.legend(loc=0)
    plt.xlim([df_SP.index.min(), df_SP.index.max()])
    
    print('backtest mean: ', simp_mean, '\nbacktest sharpe: ', simp_sharpe)

# BE SURE TO SPECIFY: (Predicted Fwd. Return/Lookback Period, Model, Model Version)
plot_return(3, 3, 3)



