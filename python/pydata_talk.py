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
from cvxpy import Variable, Minimize, quad_form, Problem, sum_entries, norm
get_ipython().magic('matplotlib inline')

# Gather Data 
#
# A list of S&P 500 tickers is available at: 
# http://data.okfn.org/data/core/s-and-p-500-companies/r/constituents.csv
#
# One can also build up a potential list of tickers for 
# stocks traded on the NASDAQ, NYSE, and AMEX exchanges from: 
# http://www.nasdaq.com/screening/company-list.aspx

tickerdf = pd.read_csv('SandP500_wiki.csv')    # read in S&P 500 tickers 
tickers = tickerdf['Ticker symbol']       # extract list of tickers from dataframe
verbose_flag = False                      # flag that turns on ticker print statements
start_date = dt.datetime(2010, 1, 1)      # date when to start downloading data
ticker_df_list = []                       # initialize list of dataframes for each ticker                          

for ticker in tickers: 
    try:
        r = DataReader(ticker, "google", start=start_date)   # download price data from yahoo 
        r['Ticker'] = ticker 
        ticker_df_list.append(r)
        if verbose_flag:
            print "Obtained data for ticker %s" % ticker  
    except:
        if verbose_flag:
            print "No data for ticker %s" % ticker  

df = pd.concat(ticker_df_list)        # build single df of all data
cell= df[['Ticker','Close']]          # extract ticker and close price information 
cell.reset_index().sort(['Ticker', 'Date'], ascending=[1,0]).set_index('Ticker')
cell.to_pickle('google_close_price.pkl')         # pickle data

deg_freedom = 5  # Student's t distribution degree of freedom parameter 
num_srs = 500    # number of time series to generate  
num_pts = 2000   # number of points per time series 
rpaths = np.random.standard_t(deg_freedom,size=(num_pts,num_srs))  # generate paths
evalplot = la.eig(pd.DataFrame(rpaths).corr())[0]                  # compute eigenvalues

plt.figure(figsize=(16,5))                                           
plt.plot(rpaths.cumsum(axis=0),alpha=0.2)
plt.xlabel("Time Step",fontsize=16)
plt.ylabel("Simulated Value",fontsize=16)
plt.title("Simulated Paths and Mean Path (Red)", fontsize=20)
sns.tsplot(rpaths.cumsum(axis=0).T, color = "r")
plt.figure(figsize=(16,5))
sns.distplot(rpaths.T[1],bins=50);
plt.xlabel("Simulated Value Bins",fontsize=16)
plt.ylabel("Probability Mass",fontsize=16)
plt.title("Histogram of Final Values and Overlaid Kernel Density Estimate", fontsize=20)
plt.figure(figsize=(16,5))
plt.hist(evalplot,bins=75);
plt.xlabel("Eigenvalue Bins",fontsize=16)
plt.ylabel("Bin Count",fontsize=16)
plt.title("Histogram of Correlation Matrix Eigenvalues", fontsize=20)

def eigden(lam,n,r):
    '''
    Definition of the Marchenko Pastur density 
    '''
    q = float(n)/r
    lplus = 1+1/q+2*np.sqrt(1/q)
    lminus = 1+1/q-2*np.sqrt(1/q)
    return q/(2*np.pi*lam)*np.sqrt((lplus-lam)*(lam-lminus))

lamvls = np.linspace(0.001,3,2000) 
plt.figure(figsize=(14,5))
plt.hist(evalplot,bins=75,normed=True)
plt.plot(lamvls,[eigden(lam,num_pts,num_srs) for lam in lamvls], color="r", linewidth=4)
plt.xlabel("Eigenvalue Bins",fontsize=16)
plt.ylabel("Probability Mass",fontsize=16)
plt.title("Histogram of Correlation Matrix Eigenvalues and Marcenko Pastur Density", fontsize=20)
plt.show()

def lamplus(n,r): 
    '''
    Upper eigenvalue limit of the Marchenko Pastur density 
    '''
    q = float(n)/float(r)
    return (1+1/q+2*np.sqrt(1/q))

# Read in closing price data if it is not already a local variable
if 'cell' not in locals():
    df = pd.read_pickle('google_close_price.pkl')
else: 
    df = cell 
    
dte1 = '2010-07-01'
dte2 = '2015-10-01'
tickers = sorted(list(set(df['Ticker'].values)))                   # sorted list of unique tickers  
tkrlens = [len(df[df.Ticker==tkr][dte1:dte2]) for tkr in tickers]  # find lengths of times series for each ticker 
tkrmode = mode(tkrlens)[0][0]                                      # find mode of time series lengths 

# idenfity tickers whose lenghts equal the mode and call these good tickers
good_tickers = [tickers[i] for i,tkr in enumerate(tkrlens) if tkrlens[i]==tkrmode]  

rtndf = pd.DataFrame()  # initialize a return dataframe
non_norm_rtndf = pd.DataFrame()
std_lst = [] # list of standard deviations of return time series

# Normalize all time series to have mean zero and variance one and compute their returns 
for tkr in good_tickers: 
    tmpdf = df[df.Ticker==tkr]['Close'][dte1:dte2]
    tmprtndf = ((tmpdf-tmpdf.shift(1))/tmpdf).dropna()
    std_lst.append(np.std(tmprtndf))
    rsdf = (tmprtndf-tmprtndf.mean())/tmprtndf.std()
    rtndf = pd.concat([rtndf, rsdf],axis=1)
    non_norm_rtndf = pd.concat([non_norm_rtndf, tmprtndf],axis=1)

rtndf = rtndf.dropna()
rtndf.columns = good_tickers
non_norm_rtndf = non_norm_rtndf.dropna()
non_norm_rtndf.columns = good_tickers
t,m = rtndf.shape
cmat = rtndf.corr()                   # compute correlation matrix 
evls, evcs = la.eig(cmat)             # compute eigenvalue/vector decomposition of matrix 
evallst = map(abs,evls)               # take abs of evals (very small imag parts)             

filtvals = [val for val in evallst if val < lamplus(t,m)]     # filter eigenvalues 
sevlist = [np.mean(filtvals)]*len(filtvals)                 
feval = evallst[:(len(evallst)-len(sevlist))] + sevlist       # build list of new eigenvalues

rcmat = abs(np.dot(np.dot(evcs,np.diag(feval)),la.inv(evcs))) # reconstruct candidate correlation matrix 
rcmat = (rcmat + rcmat.T)/2                                   # symmetrize the candidate matrix 
ncorr = nearcorr(rcmat, max_iterations=1000)                  # find nearest correlation matrix 

print evls[0:10]
plt.figure(figsize=(14,5))
plt.hist([val for val in evls if val < 3],bins=100,normed=True);
plt.plot(lamvls,[eigden(lam,len(rtndf),len(rtndf.columns)) for lam in lamvls]);
plt.xlabel("Eigenvalue Bins",fontsize=16)
plt.ylabel("Probability Mass",fontsize=16)
plt.title("Histogram of S&P 500 Correlation Matrix Eigenvalues and Marcenko Pastur Density", fontsize=20)

plt.figure(figsize=(16,13))
corrdiff = np.array(ncorr - cmat)
corrdiff = np.array([map(lambda x: min(x,0.1), row) for row in corrdiff])
corrdiff = np.array([map(lambda x: max(x,-0.1), row) for row in corrdiff])
sns.heatmap(corrdiff)
plt.axis('off')
plt.title('Difference Between Initial and Filtered Correlation Matrices', fontsize=20)

plt.figure(figsize=(16,5))
plt.hist(corrdiff.flatten(),bins = 100)
plt.xlabel("Component Difference Size",fontsize=16)
plt.ylabel("Bin Count",fontsize=16)
plt.title("Initial and Filtered Correlation Matrix Component Differences", fontsize=20)
plt.show()

ocov = np.array([[std_lst[i]*std_lst[j]*cmat.as_matrix()[i][j] for i in xrange(0,len(std_lst))] for j in xrange(0,len(std_lst))])
ncov = np.array([[std_lst[i]*std_lst[j]*ncorr[i][j] for i in xrange(0,len(std_lst))] for j in xrange(0,len(std_lst))])
dly_rtn = np.array(non_norm_rtndf.cumsum().tail(1)/len(non_norm_rtndf))[0]

lam = np.linspace(-2,2,200)

def mv_opt_port(cov,dly_rtn,div_param,lam_lst):
    n = len(cov)
    vlst = []
    rtnlst = []
    
    for v in lam_lst:
        x = Variable(n)
        objective = Minimize(quad_form(x,cov)-v*dly_rtn.T*x + div_param*norm(x,1))
        constraints = [sum_entries(x)==1]
        p = Problem(objective, constraints)
        L = p.solve()
        
        wopt = np.array(x.value).flatten()
        vlst.append(wopt.T.dot(ocov).dot(wopt))
        rtnlst.append(wopt.dot(dly_rtn))
    
    return (wopt,vlst, rtnlst)


wopt, vtmp, rtntmp = mv_opt_port(ocov, dly_rtn, 0.003, lam)    
wopt1, vtmp1, rtntmp1 = mv_opt_port(ncov, dly_rtn, 0.003, lam)    

plt.figure(figsize=(10,8))

xplt = 252*np.array(vtmp)
yplt = 252*np.array(rtntmp)
x1plt = 252*np.array(vtmp1)
y1plt = 252*np.array(rtntmp1)

pltvec=pd.concat([non_norm_rtndf.std(),non_norm_rtndf.mean()],axis=1).as_matrix()
pltarr = pd.concat([non_norm_rtndf.std(),non_norm_rtndf.mean()],axis=1).as_matrix()
pltarr = np.array([val for val in pltarr if val[0]<0.1])
plt.scatter(np.sqrt(252)*pltarr.T[0],252*pltarr.T[1],s=2,c='r')
plt.plot(np.sqrt(xplt),yplt, np.sqrt(x1plt),y1plt)
plt.xlim(0.08,0.45)
plt.ylim(-0.5,0.6) 
plt.xlabel("Annualized Volatility",fontsize=16)
plt.ylabel("Annualized Expected Return",fontsize=16)
plt.title("Efficient Frontiers for Raw (blue) and Filtered (green) Covariance Matrices", fontsize=20)
plt.show()

