import datetime as dt
import matplotlib.pyplot as plt 
import numpy as np 
import numpy.linalg as la 
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from pandas.io.data import DataReader
from scipy.stats import mode 
from scipy.stats.stats import pearsonr
from IPython.display import Image
get_ipython().magic('matplotlib inline')

# first we need a list of the tickers on the S&P 500 so that we can query Yahoo for that data
tickerdf = pd.read_csv('constituents.csv')
tickers = list(tickerdf['Symbol'].values)
start_date = dt.datetime(2012, 1, 1)

print(tickers[:5])

ticker_df_list = []
for ticker in tickers: 
    try:
        r = DataReader(ticker, "yahoo", start=start_date)   # download price data from yahoo 
        r['Ticker'] = ticker 
        ticker_df_list.append(r) 
    except:
        print "couldn't find data for {}".format(ticker)

df = pd.concat(ticker_df_list)
cell= df[['Ticker','Adj Close']]
cell.reset_index().sort(['Ticker', 'Date'], ascending=[1,0]).set_index('Ticker')
cell.to_pickle('close_price.pkl')

# generate 500 stocks, each with 2000 sampled changes (let's assume they all start at 0)
rpaths = np.random.standard_t(5, size=(2000,500))  # generate paths

# the cumulative sums plotted (i.e. the stock over time)
plt.figure(figsize=(16,5))                                         
plt.plot(rpaths.cumsum(axis=0),alpha=0.5)
plt.show()

# the actual values sampled from the t-distribution
plt.figure(figsize=(16,5))
plt.hist(rpaths.T[1],bins=50)

evalplot = la.eig(pd.DataFrame(rpaths).corr())[0]
plt.figure(figsize=(16,5))
plt.hist(evalplot,bins=50)

# density distribution
def eigden(lam,n,r):
    q = float(n)/r
    lplus = 1+1/q+2*np.sqrt(1/q)
    lminus = 1+1/q-2*np.sqrt(1/q)
    return q/(2*np.pi*lam)*np.sqrt((lplus-lam)*(lam-lminus))

lamvls = np.linspace(0.001,3,1000) 
plt.figure(figsize=(14,5))
plt.hist(evalplot,bins=75,normed=True)
plt.plot(lamvls,[eigden(lam,2000,500) for lam in lamvls])
plt.show()

from nearest_correlation import nearcorr

def lamplus(n,r): 
    '''
    Upper eigenvalue limit of the Marchenko Pastur density 
    '''
    q = float(n)/float(r)
    return (1+1/q+2*np.sqrt(1/q))

# Read in closing price data if it is not already a local variable
if 'cell' not in locals():
    df = pd.read_pickle('close_price.pkl')
else: 
    df = cell 
    
dte1 = '2010-07-01'
dte2 = '2015-07-01'
tickers = sorted(list(set(df['Ticker'].values)))                   # sorted list of unique tickers  
tkrlens = [len(df[df.Ticker==tkr][dte1:dte2]) for tkr in tickers]  # find lengths of times series for each ticker 
tkrmode = mode(tkrlens)[0][0]                                      # find mode of time series lengths 

# idenfity tickers whose lenghts equal the mode and call these good tickers
good_tickers = [tickers[i] for i,tkr in enumerate(tkrlens) if tkrlens[i]==tkrmode]  

rtndf = pd.DataFrame()  # initialize a return dataframe

# Normalize all time series to have mean zero and variance one and compute their returns 
for tkr in good_tickers: 
    tmpdf = df[df.Ticker==tkr]['Adj Close'][dte1:dte2]
    tmprtndf = ((tmpdf-tmpdf.shift(1))/tmpdf).dropna()
    rsdf = (tmprtndf-tmprtndf.mean())/tmprtndf.std()
    rtndf = pd.concat([rtndf, rsdf],axis=1)

rtndf = rtndf.dropna()
rtndf.columns = good_tickers
t,m = rtndf.shape
cmat = rtndf.corr()                                           # compute correlation matrix 
evls, evcs = la.eig(cmat)                                     # compute eigenvalue/vector decomposition of matrix 
evallst = map(abs,evls)                                       # take abs of evals (very small imag parts)
filtvals = [val for val in evallst if val < lamplus(t,m)]     # filter eigenvalues 
sevlist = [np.mean(filtvals)]*len(filtvals)                 
feval = evallst[:(len(evallst)-len(sevlist))] + sevlist       # build list of new eigenvalues
rcmat = abs(np.dot(np.dot(evcs,np.diag(feval)),la.inv(evcs))) # reconstruct candidate correlation matrix 
rcmat = (rcmat + rcmat.T)/2                                   # symmetrize the candidate matrix 
ncorr = nearcorr(rcmat, max_iterations=1000)                  # find nearest correlation matrix 
ncorrdf = pd.DataFrame(ncorr,columns=good_tickers,index=good_tickers)



