import pandas as pd # for dataframes manipulation
import numpy as np
import matplotlib.pyplot as plt
import csv

# Import the data
firms = pd.read_csv('SP_500_firms.csv', index_col = 0)
close_prices = pd.read_csv('SP_500_close_2015.csv')

firms_small = firms.iloc[:504, :]
close_prices_small = close_prices.iloc[:, :497]

#print(firms_small[:5])
#print(close_prices_small[:5])

def StockReturns(cl_p, Date = True):
    """
    Input:  The dataframe with the prices of the stocks
            A logical argument that equals True if the first column
            in the dataframe represents dates, False otherwise
    Output: A dataframe with the daily returns of the stocks - same 
            number of columns as the input vector and one row less 
            than the input vector
    """    
    if Date:
        d_ret = pd.DataFrame(cl_p.iloc[1:, 0], columns=['Date'])
        j = 1
    else:
        d_ret = pd.DataFrame()
        j = 0
    for i in range(j, cl_p.shape[1]):
        d_ret[cl_p.columns[i]] = (cl_p.iloc[1:,i].values - cl_p.iloc[:-1,i].values) / cl_p.iloc[:-1,i].values
    return d_ret
    
    
# Daily returns including the Date column    
daily_returns = StockReturns(close_prices_small) 
#print(daily_returns.iloc[:5, :5])

# Daily returns excluding the Date column
daily_returns_2 = StockReturns(close_prices_small.iloc[:, 1:], Date = False)
#print(daily_returns_2.iloc[:5, :5])   

def Correlations(d_ret):
    """
    Input:  A dataframe with the daily returns of the stocks - The 
            first column can either indicate dates or not
    Output: A list of tuples. Each tuple in the list have 3 elements:
            1. The correlation between two firms
            2 and 3. The firms for which we compute the correlation
    """
    cor = d_ret.corr()
    n = int(cor.shape[0])
    cor_list = []
    for i in range(1, n):
        for j in range(0, i):
            cor_list.append((cor.iloc[i, j], cor.columns.values[i], cor.columns.values[j]))
    return cor_list
    

correlation_list = Correlations(daily_returns)
#print(correlation_list[:5])

def SortCorrs(cor_list): 
    """
    Input:  A list of tuples. Each tuple in the list have 3 elements:
            1. The correlation between two firms
            2 and 3. The firms for which we compute the correlation
    Output: Returns the same list of tuples ordered based on the
            first element of the tuples, e.g. the correlation
    """
    return sorted(cor_list, reverse = True)
    

ordered_list = SortCorrs(correlation_list)
#print(ordered_list[:5])

def clusteringAlg(ord_list, k = 0):
    """
    Input:  
    ord_list: The ordered list of tuples which include the
              correlations between firms and the firms themselves
    k: The number of iterations for the clustering algorithm
    Output: 
    A list of sets where each set represents an individual 
    cluster
    """
    # Initialize the list of sets. Each set represents a cluster
    # which initialy includes only one firm
    sets = []
    for i in range(len(ord_list)):
        if not({ord_list[i][1]} in sets):
            sets.append({ord_list[i][1]})
        if not({ord_list[i][2]} in sets):
            sets.append({ord_list[i][2]})     
    # Repeat the algorithm k times
    # In each iteration we check the k-th tuple of correlations list
    # and whether the 2 firms in that tuple are already in the same
    # set. If they do, we move on to the next tuple, otherwise we merge
    for j in range(min(k, len(ord_list))):
        nd1 = ord_list[j][1]
        nd2 = ord_list[j][2]
        fl1, fl2 = False, False    
        for i in range(len(sets)):
            if (nd1 in sets[i]) and fl1 == False:
                idx1 = i
                fl1 = True
            if (nd2 in sets[i]) and fl2 == False:
                idx2 = i
                fl2 = True
        if idx1 != idx2:
            sets[idx1] = sets[idx1].union(sets[idx2])
            sets.remove(sets[idx2])
    return sets

def getStockDetails(ticker, clusterSets, firmDF):
    # if the ticker given is in a single set cluster
    if {ticker} in clusterSets:
        return firmDF.loc[ticker, :]
    else:
        for i in range(len(clusterSets)):
            if ticker in clusterSets[i]:
                return firmDF.loc[clusterSets[i], :].sort_values('Sector')

getStockDetails("BAC", clusteringAlg(ordered_list,30), firms)

getStockDetails("BAC", clusteringAlg(ordered_list,50), firms)

getStockDetails("BAC", clusteringAlg(ordered_list,100), firms)

getStockDetails("BAC", clusteringAlg(ordered_list,500), firms)

getStockDetails("BAC", clusteringAlg(ordered_list,1000), firms)

