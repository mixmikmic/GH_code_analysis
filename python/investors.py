from modules import *

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter

plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize'] = (10, 6)

# connect to mysql db, read cb_investments, cb_objects, and cb_funds as dataframes, disconnect
conn = dbConnect()
inv = dbTableToDataFrame(conn, 'cb_investments')
objs = dbTableToDataFrame(conn, 'cb_objects')
fund = dbTableToDataFrame(conn, 'cb_funds')
conn.close()

inv.head()

fund.head()

objs.head()

a = pd.merge(inv, objs, left_on = 'funded_object_id', right_on='id', suffixes=['_i', '_o'])

allData = pd.merge(a, fund, left_on = 'investor_object_id', right_on='object_id', suffixes=['_a', '_f'])

allData.head()

allData.shape

allData.columns

# create dictionary with list of companies funded for each investor
investors = pd.unique(allData.investor_object_id).tolist()
i = 0
f = [None] * len(investors)
for investor in investors:
    allData_investor = allData[allData.investor_object_id == investor]
    f[i] = allData_investor.funded_object_id
    i = i+1
d = dict(zip(investors, f))

#companies is a list of all companies
companies = pd.unique(allData.funded_object_id).tolist()

# initialize empty dataframe with rownames as the unique companies
cmat = pd.DataFrame([], index = companies)

# for each investor, have list of companies and store as investors

# need a counter for column names
i = 0

# loop through the investors
for investor, c in d.items():
    #create column of data frame
    column = company_vector(c, companies)
    #create index name
    index = str(i)
    #add column
    cmat[index] = column
    #update index
    i = i + 1

cmat[500:510]

len(investors)

cmat.shape

#generate matrix where 1 is empty, 0 is not
sparse_matrix = (cmat == 0).astype(int)
#calculate row_sums, number times company is not in an investor portfolio
row_sums = sparse_matrix.sum(axis = 1)
#find the sparsity, the total number of empty cells divided by the number of cells
sparsity = row_sums.sum() / (cmat.shape[0]*cmat.shape[1])
print(f"cmat is comprised of {100*sparsity:.2f}% zeros.")

from sklearn.manifold import MDS
from scipy.spatial import distance

def simple_scatterplot(x,y,title,labels):
    # Scatterplot with a title and labels
    fig, ax = plt.subplots(figsize=(16,14))
    ax.scatter(x, y, marker='o')
    plt.title(title, fontsize=14)
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i],y[i]))
    return ax

def fit_MDS_2D(distances):
    # A simple MDS embedding plot:
    mds = MDS(n_components=2, dissimilarity='precomputed',random_state=123)
    #fit based on computed Euclidean distances
    mds_fit = mds.fit(distances)
    #get 2D Euclidean points of presidents
    points = mds_fit.embedding_
    return points

# normalize distance matrix to represent the probability of investing in a 
# given company for a given investor
column_sums = cmat.sum(axis = 0).values
norm = cmat / column_sums
norm.head()

norm = np.array(norm)
norm.shape

from scipy.stats import entropy

def JSdiv(p, q):
    """Jensen-Shannon divergence.
    
    Compute the J-S divergence between two discrete probability distributions.
    
    Parameters
    ----------
    
    p, q : array
        Both p and q should be one-dimensional arrays that can be interpreted as discrete
        probability distributions (i.e. sum(p) == 1; this condition is not checked).
        
    Returns
    -------
    float
        The J-S divergence, computed using the scipy entropy function (with base 2) for
        the Kullback-Leibler divergence.
    """
    m = (p + q) / 2
    return (entropy(p, m, base=2.0) + entropy(q, m, base=2.0)) / 2

#Intialize empty matrix of JSdiv of points
JSd_dists = np.zeros(shape = (norm.shape[1], norm.shape[1]))
#Intialize empty matrix of Euclidean distances
euc_dists = np.zeros(shape = (norm.shape[1], norm.shape[1]))
#loop through columns
for i in range(norm.shape[1]):
    #catch first column to compare
    cur_col = norm[:, i]
    #loop through remaining columns
    for j in range(i, norm.shape[1], 1):
        #catch second column to compare
        comp_col = norm[:, j]
        
        #compute JSdiv
        JSd_dist = JSdiv(cur_col, comp_col)
        JSd_dists[i, j] = JSd_dist
        
        #compute Euclidean Distance
        euc_dist = distance.euclidean(cur_col, comp_col)
        euc_dists[i, j] = euc_dist
        
        #the matrices are symmetric, saves runntime to do this
        if (i != j):
            JSd_dists[j, i] = JSd_dist
            euc_dists[j, i] = euc_dist

# Fit with MDS
points_euclidean = fit_MDS_2D(euc_dists)

# Create scatter plot of projected points based on Euclidean Distances
simple_scatterplot(points_euclidean[:,0],points_euclidean[:,1],
                   "Naive MDS - Euclidean Distances",
                   investors
                  );

# Fit with MSD
points_JSd = fit_MDS_2D(JSd_dists)

#create scatter plot of projected points based on JSDiv
simple_scatterplot(points_JSd[:,0],points_JSd[:,1],
                   "Naive MDS - JSdiv",
                   investors
                  );   

def plot_embedding(data, title='MDS Embedding', savepath=None, palette='viridis', 
                   size=7):
    """Plot an MDS embedding dataframe for all presidents.
    
    Uses Seaborn's `lmplot` to create an x-y scatterplot of the data, encoding the 
    value of the investor field into the hue (which can be mapped to any desired
    color palette).
    
    Parameters
    ----------
    data : DataFrame
        A DataFrame that must contain 3 columns labeled 'x', 'y' and 'investor'.
        
    title : optional, string
        Title for the plot
        
    savepath : optional, string
        If given, a path to save the figure into using matplotlib's `savefig`.
        
    palette : optional, string
        The name of a valid Seaborn palette for coloring the points.
    
    size : optional, float
        Size of the plot in inches (single number, square plot)
        
    Returns
    -------
    FacetGrid
        The Seaborn FacetGrid object used to create the plot.
    """
    #process data
    x = data['x']
    y = data['y']
    investor = data['investor']
    
    #set boolean for using or not using annotation
    do_annotate = False
    
    #create scatterplot using linear model without a regression fit
    p = sns.lmplot(x = "x", y = "y", data = data, hue = "investor", palette = palette, size = size, fit_reg= False, legend=False)
    p.ax.legend(bbox_to_anchor=(1.01, 0.85),ncol=2)
    
    #this is used in order to annotate
    ax = plt.gca()
    
    #make grid and set title
    plt.grid()
    plt.title(title,fontsize=16)
    
    #adjust border for file saving so not cut-off
    plt.tight_layout()
    
    #save file
    if (savepath != None):
        plt.savefig(savepath)

#create embed_peu data frame
embed_peu = pd.DataFrame([])
embed_peu['x'] = points_euclidean[:, 0]
embed_peu['y'] = points_euclidean[:, 1]
embed_peu['investor'] = investors

plot_embedding(embed_peu, 'Naive MDS - euclidean distance', 'results/mds_naive.png');

#create edf2 data frame from JSdiv metric
edf2 = pd.DataFrame([])
edf2['x'] = points_JSd[:, 0]
edf2['y'] = points_JSd[:, 1]
edf2['investor'] = investors

plot_embedding(edf2, 'MDS - Jensen-Shannon Distance', 'results/mds_jsdiv.png');

