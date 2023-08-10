# This is to print in markdown style
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))

import pandas as pd

# Read dataset locally
#df = pd.read_csv('../data/db-readability-length.csv', index_col=0)

# Read dataset from url:
import io
import requests
url="https://raw.githubusercontent.com/trangel/stats-with-python/master/data/db-readability-length.csv"
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')),index_col=0)


del df['Length']
X = df['Readability'].values
df.head()

import numpy as np
# Sometimes it's important to remove outliers, as some of the tests don't work in presence of outliers
# Only keep data in range 1 to 99 th percentile:
X = df[df['Readability'].between(np.percentile(X,1), np.percentile(X,99), inclusive=True )].values

# Standarize and sort data
from sklearn.preprocessing import StandardScaler
import numpy as np
scaler = StandardScaler()

X = X.reshape(len(X))
X = scaler.fit_transform(X)
X = np.sort(X)


# Get also a normally distributed random sample:
from scipy.stats import norm

# Get random samples
n = len(X)

# Get random numbers, normaly distributed
A = norm.rvs(size=n)
A = np.sort(A)

# Second normal distribution, for comparison
B = norm.rvs(size=n)
B = np.sort(B)

import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="ticks")

# Load the example dataset for Anscombe's quartet
#df = sns.load_dataset("anscombe")

bins = np.arange(-10,10,0.5)

# Show the results of a linear regression within each dataset
ax1 = sns.distplot(A,bins=bins,label='Normal distribution A')
ax2 = sns.distplot(B,bins=bins,label='Normal distribution B')
ax3 = sns.distplot(X,bins=bins,label='X')

plt.pyplot.xlabel('x')
plt.pyplot.ylabel('Distribution')
plt.pyplot.legend(bbox_to_anchor=(0.45, 0.95), loc=0, borderaxespad=0.)

plt.pyplot.xlim((-10,10))
plt.pyplot.show()

def get_quartiles(X,Y):
    """Gets 100 quartiles for distributions X and Y
    returns Q1, Q2 with the corresponding quartiles"""
    # Get quartiles, 100 of them
    Q1 = []; Q2=[]
    nq = 10
    n1 = len(X)
    n2 = len(Y)
    for i in range(nq):
        j1 = int(i * n1/nq)
        j2 = int(i * n2/nq)
        # quartiles for the two distributions will be stored in X and Y respectively
        Q1.append(X[j1])
        Q2.append(Y[j2])
    Q1 = np.array(Q1)
    Q2 = np.array(Q2)
    return Q1, Q2

def plot_q_q(Q1,Q2):
    """Makes q-q plot
    Input Quartiles for two distributions"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    line1, = ax.plot(Q1, Q1, '--', linewidth=2,
                 label='')
    line2, = ax.plot(Q1,Q2, linestyle = 'dotted', linewidth = 4,
                 label='Q-Q')
    #ax.legend(loc='lower right')
    ax.set_title('Q - Q plot')
    ax.set_ylabel('Distribution 2')
    ax.set_xlabel('Distribution 1')
    plt.show()

Q1, Q2 = get_quartiles(X,A)
plot_q_q(Q1,Q2)

Q1, Q2 = get_quartiles(A,B)
plot_q_q(Q1,Q2)

from scipy import stats

# Test our initial distribution
printmd('## Skew and Kurtosis tests for a random distribution')
printmd('#### Is our X distribution normal?')
print('Normal skew test teststat     = %6.3f pvalue = %6.4f' % stats.skewtest(X))
print('Normal kurtosis test teststat = %6.3f pvalue = %6.4f' % stats.kurtosistest(X))
printmd('**Conclusion** The pvalue for both tests are zero, meaning that we can reject $H_0$\n')
printmd('This conclusion is telling that X is not normal')

printmd('#### Comparison of 2 normal distributions')
printmd('For the sake of comparison, we repeat the test for a distribution which is actually normal')
printmd('Skew and Kurtosis tests for a normal random distribution')
print('Normal skew test teststat     = %6.3f pvalue = %6.4f' % stats.skewtest(A))
print('Normal kurtosis test teststat = %6.3f pvalue = %6.4f' % stats.kurtosistest(A))
printmd('**Conclusion** The pvalue for both tests are $>$ 0.05, so we accept $H_0$\n')
printmd('This means that A and B are similar, as expected since these two are actual normal distributions')


# Get CDF functions:

from scipy.stats import norm
from scipy.special import kolmogorov

# Get CDF for X, this may not be the most effective or elegant way to do this:
percentiles = np.arange(0,1,(1.0-0.0)/len(X))*100.0
CDF_X=[]; Values=[]
for percentile in percentiles:
    Values.append(np.percentile(X,percentile))
    CDF_X.append(percentile/100.0)
CDF_X=np.array(CDF_X)
Values=np.array(Values)


# CDF for a normal distribution, using the same points as for X
rv = norm()
CDF_A = rv.cdf(Values)

# Get the maximum distance between the two distributions:
KS_statistic = (CDF_X - CDF_A).max()
# Get the index for the max. between the two distributions, for plotting
KS_index = (CDF_X - CDF_A).argmax()

# Get the pvalue from the corresonding scipy function:
# https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.special.kolmogorov.html
pvalue=kolmogorov(np.sqrt(len(X))*KS_statistic)
printmd("KS statistic = {}".format(KS_statistic))
printmd("pvalue = {}".format(pvalue))

import matplotlib as plt
xmax = X.max()
xmin = X.min()
xstep = 0.01

print(xmin,xmax,xstep)

xx = np.arange(xmin,xmax,xstep)

fig, ax = plt.pyplot.subplots()

ax.plot(Values, CDF_X, 'g-', lw=5, alpha=0.6, label = 'CDF for X' )
ax.plot(Values, CDF_A, 'r-', lw=5, alpha=0.6, label = 'CDF for normal distribution')

ax.axvline(x=X[KS_index])

ax.text(0.1 , 0.5, r'Max. diff.',color='b')

#plt.xlim(-3,7)
plt.pyplot.ylim(0,1)

ax.legend(loc='right')
#ax.set_title('PDF function')
ax.set_ylabel('Distribution')
ax.set_xlabel('x')


plt.pyplot.show()

stats.kstest(X,'norm',N=len(X))

from scipy import stats
stats.shapiro(X)

stats.anderson(X, dist='norm')



