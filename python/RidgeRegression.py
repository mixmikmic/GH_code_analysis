import pandas as pd ; import numpy as np ; import random ; from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Read in the NC State diabetes data set.  This is the same one available from
# sklearn.datasets.load_diabetes(), but I prefer to get it from the original page
# linked in the sklearn docs:  http://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
# Using that instead, when I read it into a Pandas DataFrame, I get column headings.
df = pd.read_table('diabetes.rwrite1.txt',delimiter=' ')
df.shape

df.head(3)

# Note that the data have been normalized so that for each column the mean is zero and the
# length is 1.  (If you want to manually normalize the original data yourself, go to  
# http://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt to get the original data.
# For each column, subtract its mean and then divide each column by its length (i.e., sqrt 
# of the sum of its squared values).  That produces the normalized values shown above.

# Here's a quick little histogram of one column, to illustrate the normalization
plt.rcParams["figure.figsize"] = (18,4)
plt.hist(df.tc,bins=21)

# declare a function for plotting a correlation matrix for a Pandas DataFrame
def plot_corr(df, size):
    """
    Plots correlation matrix for each pair of columns.
    Blue-to-red runs from not-corr (or 0) to correlated (or 1)
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize = (size,size))
    cax = ax.matshow(corr,cmap='seismic')
    fig.colorbar(cax,fraction=0.0458, pad=0.04)
    plt.xticks(np.arange(len(corr.columns)), corr.columns)
    plt.yticks(np.arange(len(corr.columns)), corr.columns)

# after some playing around, I choose to use columns tc and hdl since 
# they are one of the least correlated pairings of features.  
# Here's a plot for features tc and hdl (and the target y).
plot_corr(df[['tc','hdl','y']],5)

# also, make a scatter plot of tc and hdl against each other, 
# which is another way to assess correlation more visually.
def scat_corr(ax,feat1,feat2,lab1,lab2,col):
    '''
    Scatter plots feat1 against feat1 for 
    visually assessing correlation
    '''
    ax.scatter(feat1,feat2,color=col,s=5)
    ax.set_xlabel(lab1)
    ax.set_ylabel(lab2)

plt.rcParams["figure.figsize"] = (5,5)
fig, ax1 = plt.subplots()
scat_corr(ax1,df.tc,df.hdl,'df.tc','df.hdl','k')

# get rid of all columns other than tc, hdl and y
for col in ['age', 'sex', 'bmi', 'map', 'ldl', 'tch', 'ltg', 'glu']:  del df[col]
df.head(5)

# Since the hdl measurements are already normalized, it doesn't matter what the scaling 
# factor between hdl and "zdl" is.  Instead, I just fake "zdl" measurements by adding 
# some pseudorandom noise onto the already normalized hdl values.
np.random.seed(datetime.now().microsecond)
df['zdl'] = df['hdl'] + np.random.uniform(-0.005, 0.005, len(df.tc))

# Show that hdl and zdl are highly correlated
plot_corr(df,5)

# and also show the correlation with a scatter plot
plt.rcParams["figure.figsize"] = (5,5)
fig, ax1 = plt.subplots(1,1)
scat_corr(ax1,df.hdl,df.zdl,'df.hdl','df.zdl','k')

# for ease of reading below, form matrix X-- each column is one feature
X = df[ ['tc','hdl','zdl'] ].values
X.shape

# form vector y of the target
y = df['y'].values
y.shape

# a peek at the first few rows to compare to the data frame
X[:5], y[:5]

# fit RidgeRegression models for a range of alpha values
from sklearn import linear_model
n_alphas = 20
alphas = np.logspace(-4, 0, n_alphas)
clf = linear_model.Ridge()

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_)

# make an array of each feature's weight for plotting & labeling
w1 = np.array(coefs)[:,0]
w2 = np.array(coefs)[:,1]
w3 = np.array(coefs)[:,2]

# plot each weight versus alpha
plt.rcParams["figure.figsize"] = (18,8)

plt.plot(alphas, w1, label=r'$w_1$')
plt.plot(alphas, w2, label=r'$w_2$')
plt.plot(alphas, w3, label=r'$w_3$')

plt.xscale('log')
plt.xlim(plt.xlim()[::-1])  # reverse axis
plt.xlabel(r'$\alpha$',fontsize='xx-large')
plt.ylabel('weights',fontsize='xx-large')
plt.legend(fontsize='xx-large',loc='lower left')

