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
from nearest_correlation import nearcorr
from IPython.display import Image
get_ipython().magic('matplotlib inline')

# Gather Data 
#
# A list of S&P 500 tickers is available at: 
# http://data.okfn.org/data/core/s-and-p-500-companies/r/constituents.csv
#
# One can also build up a potential list of tickers for 
# stocks traded on the NASDAQ, NYSE, and AMEX exchanges from: 
# http://www.nasdaq.com/screening/company-list.aspx

tickerdf = pd.read_csv('SandP500.csv')    # read in S&P 500 tickers 
tickers = list(tickerdf['Symbol'].values) # extract list of tickers from dataframe
verbose_flag = False                      # flag that turns on ticker print statements
start_date = dt.datetime(2010, 1, 1)      # date when to start downloading data
ticker_df_list = []                       # initialize list of dataframes for each ticker                          

for ticker in tickers: 
    try:
        r = DataReader(ticker, "yahoo", start=start_date)   # download price data from yahoo 
        r['Ticker'] = ticker 
        ticker_df_list.append(r)
        if verbose_flag:
            print "Obtained data for ticker %s" % ticker  
    except:
        if verbose_flag:
            print "No data for ticker %s" % ticker  

df = pd.concat(ticker_df_list)            # build single df of all data
cell= df[['Ticker','Adj Close']]          # extract ticker and close price information 
cell.reset_index().sort(['Ticker', 'Date'], ascending=[1,0]).set_index('Ticker')
cell.to_pickle('close_price.pkl')         # pickle data

deg_freedom = 5  # Student's t distribution degree of freedom parameter 
num_srs = 500    # number of time series to generate  
num_pts = 2000   # number of points per time series 
rpaths = np.random.standard_t(deg_freedom,size=(num_pts,num_srs))  # generate paths
evalplot = la.eig(pd.DataFrame(rpaths).corr())[0]                  # compute eigenvalues
plt.figure(figsize=(16,5))                                           
plt.plot(rpaths.cumsum(axis=0),alpha=0.5)
plt.show()
plt.figure(figsize=(16,5))
plt.hist(rpaths.T[1],bins=50);
plt.figure(figsize=(16,5))
plt.hist(evalplot,bins=50);

def eigden(lam,n,r):
    '''
    Definition of the Marchenko Pastur density 
    '''
    q = float(n)/r
    lplus = 1+1/q+2*np.sqrt(1/q)
    lminus = 1+1/q-2*np.sqrt(1/q)
    return q/(2*np.pi*lam)*np.sqrt((lplus-lam)*(lam-lminus))

lamvls = np.linspace(0.001,3,1000) 
plt.figure(figsize=(14,5))
plt.hist(evalplot,bins=75,normed=True)
plt.plot(lamvls,[eigden(lam,num_pts,num_srs) for lam in lamvls])
plt.show()

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

print evls[0:10]
plt.figure(figsize=(14,5))
plt.hist([val for val in evls if val < 3] ,bins=100,normed=True);
plt.plot(lamvls,[eigden(lam,len(rtndf),len(rtndf.columns)) for lam in lamvls]);

