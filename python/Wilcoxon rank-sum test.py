import pandas as pd
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))

# Read dataset locally
#df = pd.read_csv("../data/data-for-drug-experiment.csv",index_col=0)
    
# Read dataset from url:
import io
import requests
url="https://raw.githubusercontent.com/trangel/stats-with-python/master/data/data-for-drug-experiment.csv"
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')),index_col=0)
    
df

def rank_distribution(dist,column):
    """Get distribution ranks for non-parametric test
    Column: str, column name to get ranking
    Input: dataframe with distribution values.
    Output: dataframe including ranks
    """
    # Sort values
    dist.sort_values(by=column,inplace=True)
    ## Convert series to dataframe, to add ranks
    #dist = dist.to_frame()
    dist['Potential rank']=0
    jj=0
    for ii in dist.index:
        jj = jj + 1
        dist.loc[ii,'Potential rank']=jj
        
    # Get actual ranks by averaging ranks of repeated scores  
    # rank table with average (mean) values of ranks group by BDI scores:   
    rank = dist.groupby(column).agg({'Potential rank':'mean'})
    rank.columns=['Rank']
    #
    dist = dist.join(rank,on=column)

    return dist

BDI_Sunday = df[['Drug','BDI Sunday']].copy()
BDI_Sunday = rank_distribution(BDI_Sunday,'BDI Sunday')

BDI_Sunday.head(6)

BDI_Wednesday = df[['Drug','BDI Wednesday']].copy()

BDI_Wednesday = rank_distribution(BDI_Wednesday,'BDI Wednesday')

BDI_Wednesday.head(6)

# For Sunday's BDI:

rank_sum = BDI_Sunday[['Drug','Rank']].groupby('Drug').agg('sum')
rank_sum.columns=['Rank sum']
rank_sum_sunday = rank_sum
rank_sum

W = rank_sum.values.min()
printmd('$W_s$ {}'.format(W))

import numpy as np
n1 = 10
n2 = 10
W_mean = n1 *(n1 + n2 + 1.0)/2.0
SE  = (n1*n2 * (n1+n2+1.0)/12.0)**0.5

# Using the z-score equation the Sunday score is:   
z = (W - W_mean)/SE

printmd('** Sunday **')
printmd('$W_s$ {}'.format(W))
printmd('$W_s$ mean {}'.format(W_mean))
printmd('SE {}'.format(round(SE,2)))
printmd('$z$-score {}'.format(round(z,2)))

from scipy import stats
DF=10000  # set a large degrees of freedom.
pvalue = 2* stats.t.cdf(z,DF) # get the P value from the CDF of a t-distribution
# Multiply by two, since this is two-tailed test

printmd('p-value {}'.format(round(pvalue,4)))

# Compute now the Wednesday z-score
# Using the z-score equation the Sunday score is:   

rank_sum = BDI_Wednesday[['Drug','Rank']].groupby('Drug').agg('sum')
rank_sum.columns=['Rank sum']

W = rank_sum.values.min()

z = (W - W_mean)/SE

# Get the P-value as above
DF=10000  # set a large degrees of freedom.
pvalue = 2* stats.t.cdf(z,DF) # get the P value from the CDF of a t-distribution
# Multiply by two, since this is two-tailed test


printmd('** Wednesday **')
printmd('$W_s$ {}'.format(W))
printmd('$W_s$ mean {}'.format(W_mean))
printmd('SE {}'.format(round(SE,2)))
printmd('$z$-score {}'.format(round(z,2)))
printmd('p-value {}'.format(round(pvalue,4)))

from scipy import stats

# Let's do the Wednesday sample as an example:   
# SELECT 'BDI Wednesday' FROM df WHERE 'Drug'='Alcohol'
x = df.loc[df['Drug']=='Alcohol']['BDI Wednesday'].values
# SELECT 'BDI Wednesday' FROM df WHERE 'Drug'='Ecstasy'
y = df.loc[df['Drug']=='Ecstasy']['BDI Wednesday'].values

results = stats.ranksums(x, y)
z = results[0]
pvalue = results[1]
printmd('** Wednesday **')
printmd('$z$-score {}'.format(round(z,2)))
printmd('p-value {}'.format(round(pvalue,4)))

# Let's do the Sunday sample :   
# SELECT 'BDI Sunday' FROM df WHERE 'Drug'='Alcohol'
x = df.loc[df['Drug']=='Alcohol']['BDI Sunday'].values
# SELECT 'BDI Sunday' FROM df WHERE 'Drug'='Ecstasy'
y = df.loc[df['Drug']=='Ecstasy']['BDI Sunday'].values

results = stats.ranksums(x, y)
z = results[0]
pvalue = results[1]
printmd('** Sunday **')
printmd('$z$-score {}'.format(round(z,2)))
printmd('p-value {}'.format(round(pvalue,4)))

rank_sum_sunday

R1=rank_sum_sunday['Rank sum'][1]
printmd('**Sunday**')
printmd('Rank sum for group 1 (Ecstasy data) is {}'.format(R1))

n1=10
n2=10
U = n1*n2 + n1*(n1+1)/2 - R1
printmd('Test statistic, U = {}'.format(U))

# Let's do the Sunday sample :   
# SELECT 'BDI Sunday' FROM df WHERE 'Drug'='Alcohol'
x = df.loc[df['Drug']=='Alcohol']['BDI Sunday'].values
# SELECT 'BDI Sunday' FROM df WHERE 'Drug'='Ecstasy'
y = df.loc[df['Drug']=='Ecstasy']['BDI Sunday'].values

stats.mannwhitneyu(x, y, use_continuity=True, alternative=None)



