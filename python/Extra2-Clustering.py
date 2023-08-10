import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def stockReturns(priceDF):
    
    # Method 1: Compute daily returns by parsing the Data Frame as numpy matrix (Fastest)
    compTickers = priceDF.columns[1: ]    
    priceMat = priceDF.loc[ : , compTickers].as_matrix()    
    diffMat = (priceMat[1: ] - priceMat[ :-1]) / priceMat[ :-1]
    
    return pd.DataFrame(data = diffMat, index = priceDF.index[1: ], columns = compTickers)

def calCorrelations(dailyReturn):
    
    # Method 1: Pandas built-in function to calculate pairwise correlation (Fastest)
#    return dailyReturn.corr()
    
    # Method 2: Manual calculation of pairwise correlation using numpy matrix (Faster)
    col = dailyReturn.columns
    ncol = len(col)
    corrMat = np.identity(ncol)
    
    G = nx.Graph()
    G.add_nodes_from(col.values)
        
    n = len(dailyReturn)
    for i in range(0, ncol):
        for j in range(i + 1, ncol):
            x = dailyReturn[col[i]]
            y = dailyReturn[col[j]]
            xsum = sum(x)
            ysum = sum(y)
            corrMat[i][j] = (n * sum(x * y) - xsum * ysum) / (np.sqrt(n * sum(x**2) - xsum**2) * np.sqrt(n * sum(y**2) - ysum**2))
            corrMat[j][i] = corrMat[i][j]
            G.add_edge(col[i], col[j], weight = corrMat[i][j])
        
    return pd.DataFrame(data = corrMat, index = col, columns = col), G

def stockClustering(graph, k):
            
    # Method 1: Clustering using dictionary and list
    sortedEdges = sorted(graph.edges(data = True), key = lambda edge: edge[2]['weight'], reverse = True) # O(e log e), e = number of edges = n(n - 1)/2 for dense graph
    
    listSets = dict()    
    for node in graph.nodes(): # O(n), n = number of companies
        listSets[node] = {node}
        
    for i in range(0, k): # Execute k times
        (a, b, data) = sortedEdges[i]
        mergedSet = listSets[a].union(listSets[b]) # O(len(set(a)) + len(set(b))) = O(n) in worst case since the biggest possible set is the set with all n companies
        for node in mergedSet: # O(n)
            listSets[node] = mergedSet
    # The whole chunk above is O(kn + kn) = O(kn)
            
    resultSets = []
    for nodeSet in listSets.values(): # O(n)
        if nodeSet not in resultSets:
            resultSets.append(nodeSet)
        
    return resultSets

priceDF = pd.read_csv('SP_500_close_2015.csv')
firmDF = pd.read_csv('SP_500_firms.csv', index_col = 0)

dailyReturn = stockReturns(priceDF)

corrDF, corrGraph = calCorrelations(dailyReturn)

cluster = stockClustering(corrGraph, 1000)

def getStockDetails(ticker, clusterSets, firmDF):
    
    # if the ticker given is in a single set cluster
    if {ticker} in clusterSets:
        return firmDF.loc[ticker, :]
    else:
        for i in range(len(clusterSets)):
            if ticker in clusterSets[i]:
                return firmDF.loc[clusterSets[i], :].sort_values(['Sector', 'Name'])
            

from sklearn.cluster import AgglomerativeClustering

def stockClusteringAgglomerative(corrDF, num_clusters):
    
    # Store column names as company's ticker symbol
    compTickers = corrDF.columns
    
    # Transform correlation matrix into distance matrix
    affinityMatrix = abs(corrDF.as_matrix() - 1)
    
    # Run agglomerative clustering with average-linkage with precomputed distance matrix
    model = AgglomerativeClustering(linkage = 'average', affinity = 'precomputed', n_clusters = num_clusters)
    aggFit = model.fit(affinityMatrix)
    
    resultSets = [set()] * num_clusters
    i = 0
    
    # Return the clustering results in the form of "list of sets", each set representing a cluster
    for l in aggFit.labels_:
        resultSets[l] = resultSets[l].union({compTickers[i]})
        i += 1
        
    return resultSets

from sklearn.cluster import KMeans

def stockClusteringKMeans(dailyReturn, num_clusters):
    
    # Store column names as company's ticker symbol
    compTickers = dailyReturn.columns
    
    # Transpose the daily return matrix so that each row contains all the price changes of a particular stock
    dailyReturnArray = dailyReturn.as_matrix().transpose()
    
    # Run K-Means clustering with random centroid initialisation
    kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(dailyReturnArray)
    
    resultSets = [set()] * num_clusters
    i = 0
    
    # Return the clustering results in the form of "list of sets", each set representing a cluster
    for l in kmeans.labels_:
        resultSets[l] = resultSets[l].union({compTickers[i]})
        i += 1
    
    return resultSets

getStockDetails('GS', cluster, firmDF)

# Call cluster-linkage clustering using correlation DF, and num_cluster = number of clusters produced earlier
clusterAggAveLinkage = stockClusteringAgglomerative(corrDF, len(cluster))

getStockDetails('GS', clusterAggAveLinkage, firmDF)

# Call K-Means clustering using daily return DF, and num_cluster = number of clusters produced earlier
clusterKMeans = stockClusteringKMeans(dailyReturn, len(cluster))

getStockDetails('GS', clusterKMeans, firmDF)

getStockDetails('ZION', clusterKMeans, firmDF)

getStockDetails('USB', clusterKMeans, firmDF)

# Re-run K-Means clustering with num_cluster = 50
clusterKMeans = stockClusteringKMeans(dailyReturn, 50)

getStockDetails('GS', clusterKMeans, firmDF)

getStockDetails('JNJ', clusterKMeans, firmDF)

