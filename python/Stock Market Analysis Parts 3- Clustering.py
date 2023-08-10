#Populate the correlation value for each company within S&P 500 and sort based on the correlation value (from higher to lower)
col = dailyret.columns.tolist()

def sortCorr():
    edges = list()
    for i in range(0, cor.shape[0]):
       for j in range(i + 1, cor.shape[0]):
           edges.append((cor.iloc[i,j], col[i], col[j]))
    edges = sorted(edges, reverse = False)
    return edges

sortCorr = sortCorr()

#Initiate clusters with nodes pointing at themselves
col = dailyret.columns.tolist()
def initiateCluster():
    nodePointers = {}
    for i in col:
        nodePointers[i] = i
    return nodePointers

#Create cluster function    
def tryCluster(k):
    nodePointers = initiateCluster()
    L = []
    if k > len(sortCorr) or k < 0: #compare k with the sorted correlation table length
        print('Invalid k')
    else:
        for i in range(k):
            input2 = sortCorr[i]
            startingpoint = input2[1]
            endingpoint = input2[2]
            while nodePointers[startingpoint] != startingpoint:
                startingpoint = nodePointers[startingpoint]
            while nodePointers[endingpoint] != endingpoint:
                endingpoint = nodePointers[endingpoint]
            
            nodePointers[startingpoint] = endingpoint
            c = [startingpoint, endingpoint]
            L.append(c) #to populate connected companies
    return nodePointers, L

#%matplotlib inline
import networkx as nx
import matplotlib.pyplot as plt

def displayGraph(testedge):
    P = nx.Graph()
    for i in range(len(testedge)):
        P.add_nodes_from([testedge[i][0]])
        
    for i in range(len(testedge)):
        P.add_edges_from([(testedge[i][0], testedge[i][1])])
    
    #get default size
    fig_size = plt.rcParams["figure.figsize"]
    
    #change size
    fig_size[0] = 7
    fig_size[1] = 7
    #plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    nx.draw_spring(P, node_size = 200, arrows = False, with_labels = True)
    
    #plt.savefig("simple_path.png") # save as png
    plt.show() # display

initiateCluster()
#run cluster for 10 iterations
test1Pointer, testedge = tryCluster(10)
#display the graph
displayGraph(testedge)

initiateCluster()
#run cluster for 100 iterations
test1Pointer, testedge = tryCluster(100)
#display the graph
displayGraph(testedge)

initiateCluster()
#run cluster for 100 iterations
test1Pointer, testedge = tryCluster(1000)
#display the graph
displayGraph(testedge)

#Function takes number of k iteration to produce list of clusters
def cluster(k):    
    
    #initiate cluster    
    nodePointers = {}
    for i in col:
        nodePointers[i] = i
    
   # nodePointers = tryCluster()

    #running clustering
    for i in range(k):
       inputa = sortCorr[i]
       startingpoint = inputa[1]
       endingpoint = inputa[2]
       while nodePointers[startingpoint] != startingpoint:
           startingpoint = nodePointers[startingpoint]
       while nodePointers[endingpoint] != endingpoint:
           endingpoint = nodePointers[endingpoint]
       nodePointers[startingpoint] = endingpoint

    #turning the dictionary with correlated items to a list
    npl= [(k,v) for k,v in nodePointers.items()]

    #code for allocating correlated share pairs into clusters. clusters will be separated by 'new cluster' statement.
    cl1=[] #list for elements of clusters
    for i in range(0,len(npl)):           
        if cl1==[]: cl1.append("new cluster")                       
        else: 
            if cl1[-1]!="new cluster": cl1.append("new cluster")    
    #if an element is not in the list of clusters- add this element
        if npl[i][0]!=npl[i][1]:                                
            if npl[i][0] not in cl1: cl1.append(npl[i][0])
            if npl[i][1] not in cl1: cl1.append(npl[i][1]) 
            j = 0
            while j < (len(npl)):       #loop to go through nodes: nodes connexted to elements added to the list should be added to the list 
                if npl[j][0] in cl1: 
                    if npl[j][1] not in cl1: 
                        cl1.append(npl[j][1])
                        j=0
                if npl[j][1] in cl1: 
                    if npl[j][0] not in cl1: 
                        cl1.append(npl[j][0])
                        j=0
                j+=1        

    #Creating dictionary showing clusters and companies per clusters
    count = 0
    for i in cl1:
        if i == 'new cluster':
            count += 1
    clusters = dict()
    j = 1       
    for i in range(count-1): 
        cluslist = list()
        while cl1[j] != 'new cluster':
            cluslist.append(cl1[j])
            j += 1
            if cl1[j] == 'new cluster':
                j = j + 1
                break
        clusters[i] = cluslist 
    return clusters,npl,cl1,k

clusters = cluster(5000)

#Populating a list for elemnts which are not part of clusters
def nonclust(k):    
    cl0=[] #list of remaining single elements, which are not part of any clusters
    npl = cluster(k)[1]
    cl1= cluster(k)[2]
    for i in range(0,len(npl)):           
        if npl[i][0] not in cl1:  #if company index is NOT in the list of clusters (cl1)- add to the list of non-clusters (cl0)  
            cl0.append(npl[i][0]) 
    return cl0

cl0 = nonclust(5000)
#len(nonclust(2))

###Part 3: Outcome statements: 

#creating a graph showing how clusters behave with increasing number of k
def cluststat(lim):
    k=0
    clustsummary= pd.DataFrame()                        #creating a dataframe
    npl = cluster(k)[1]
    while k<=lim:                                       #setting up a loop for running cluster allocation for k from o to 'lim'
        cl0= nonclust(k)
        numclust = len(cluster(k)[0])                   # number of clusters
        if numclust==0: avclust = 0
        else: avclust = (len(npl)-len(cl0))/ numclust   # size of an average cluster

        clustcomp = len(npl)-len(cl0)                   # number of clustered elements

        #appending stats elements to the dataframe
        clustsummary= clustsummary.append({'k':k,'number of clusters': numclust, 'avg cluster size' : round(avclust,1), 'clusters elements': clustcomp, 'elements not in clusters': len(cl0)}, ignore_index=True)
        #print (k, numclust, round(avclust,1), clustcomp, len(cl0)) # print statement is not necessary, but it informs you of the progress       
        k += 400  #interval for k increments
    return clustsummary

clustsummary = cluststat(5000) # running the stats function for k iterations
#clustsummary2= cluststat(200) # running the stats function for k iterations


#plotting for data sets in relation to cluster stats
fig= plt.figure()
ax1= fig.add_subplot(2,2,1); plt.plot(clustsummary[['k']],clustsummary[['number of clusters']])
ax2= fig.add_subplot(2,2,2); plt.plot(clustsummary[['k']],clustsummary[['clusters elements']])
ax3= fig.add_subplot(2,2,3); plt.plot(clustsummary[['k']],clustsummary[['avg cluster size']])
ax4= fig.add_subplot(2,2,4); plt.plot(clustsummary[['k']],clustsummary[['elements not in clusters']])
#assigning graphs titles
ax1.set_title('number of clusters')  
ax2.set_title('elements in clusters')
ax3.set_title('average cluster size'); ax3.set_xlabel('number of k iterations')
ax4.set_title('elements not in clusters'); ax4.set_xlabel('number of k iterations')
plt.subplots_adjust(hspace=0.4) #setting space between the graphs 
plt.show()
#plt.savefig('clustsumary.png')  # saving graph

#Change company names dictionary to list
indinfo= [(k,v) for k,v in companyNames.items()]

#Please note that this code worked in Sypder to produce list as we explained in the table, but we found technical issues
#with Jupyter which always produce different results.We still put our code here as the evidence of our effort and to support our discussion

#analysis of clusters. Adding information related to the industry
def clustdesc(k):
    testdict=dict() #creating an empty dictionary
    clusters= cluster(k)[0]     
    for i in range(0,len(clusters)-1):  #going through clusters
        templ=list()                        #empty the list
        for j in range(0,len(clusters[i])):   #going through elements in a cluster i
            
            for k in range(0,len(indinfo)):       #going through elements of the stock-industry table
                if clusters[i][j]==indinfo[k][0]:  #for a company in the list- bring its industry and attach it to the temp list 
                    templ.append(indinfo[k][1][1])                
                testdict[i] = templ       #add industry information per cluster to the dictionary
    return testdict

#clustdesc(10)

#'normalising the prices, i.e. setting prices to 1 at the beginning of the year to enable ease of comparison
dailyRet3= priceData.copy()
for i in range (0,len(priceData)): 
    for j in range (0,(priceData.shape[1])):
        if i==0: dailyRet3.iloc[i,j] = 1
        else:dailyRet3.iloc[i,j]= dailyRet3.iloc[i-1,j]*(1+ dailyret.iloc[i,j])

#creating a dataframe for cluster 1 when run for 100 iterations
clust1temp = pd.DataFrame(columns = cluster(100)[0][0], index = dailyRet3.index)
for i in clust1temp.columns:
    for k in dailyRet3.columns:
        if i == k:
            clust1temp[i] = dailyRet3[k]
 
  
#creating a dataframe for cluster 1 when run for 1000 iterations
clust2temp = pd.DataFrame(columns = cluster(1000)[0][0], index = dailyRet3.index)
for i in clust2temp.columns:
    for k in dailyRet3.columns:
        if i == k:
            clust2temp[i] = dailyRet3[k]

get_ipython().magic('matplotlib inline')
            
#plotting a graph for cluster with k=100
clust1temp.index = pd.to_datetime(clust1temp.index)
clust1temp.plot(legend=None)
plt.show()

#plotting a graph for cluster with k=1000
clust2temp.index = pd.to_datetime(clust2temp.index)
clust2temp.plot(legend=None)
plt.show() 

nodePointers= initiateCluster()

b = nodePointers.copy()

# create two new dictionaries with same keys
aDict = b.copy()
aDict2 = b.copy()
highcorr = {}

# algorithm to create highly correlated clusters
def corrcluster_high(cut):
    cutoff = [x for x in range(0, len(sortCorr)) if sortCorr[x][0] >= cut] # no need to look through all data, correlation below cutoff is not possible to be added to cluster
    for i in col:
        aDict[i] = [i]  # dictionary for selfpointing nodes 
    for i in cutoff:
        for a in col:
            aDict2[a] = [1] # dictionary for correlation, resets after each iteration
        input2 = sortCorr[i]
        startingpoint = input2[1] # first stock
        endingpoint = input2[2] # second stock
        for k in aDict[startingpoint]: # go through all elements of cluster of the starting point
            for m in aDict[endingpoint]: # go through all elements of cluster of the ending point
                for j in range(0,len(sortCorr)): # look in sorted list for correlation between elements in starting point cluster and ending point cluster
                    if sortCorr[j][1] == m and sortCorr[j][2] == k: 
                        aDict2[startingpoint].append(sortCorr[j][0]) # append the correlation to the startingpoint dict
                    if sortCorr[j][2] == m and sortCorr[j][1] == k:
                        aDict2[endingpoint].append(sortCorr[j][0]) # append the correlation to the endingpoint dict
        if all (o >= cut for o in aDict2[startingpoint]):
            if all (b >= cut for b in aDict2[endingpoint]): # checking if all correlations between all element of bost clusters are over cutoff point
                for s in aDict[startingpoint]:  # connect clusters
                    if s not in aDict[endingpoint]:
                        aDict[endingpoint].append(s)
                for t in aDict[endingpoint]:
                    if t not in aDict[startingpoint]:
                        aDict[startingpoint].append(t)
    return aDict

#try function for at least 0.8 correlation
highcorr = corrcluster_high(0.80)

# copy clusters into new dictionary
highcorr_clean = {}
for key in highcorr.keys():
    if len(highcorr[key]) == 1:
        highcorr[key] = [1]
    if highcorr[key] != 1:
        for i in range(1, len(highcorr[key])):        
            highcorr[highcorr[key][i]] = [1]
    if len(highcorr[key]) > 1:
        highcorr_clean[key] = highcorr[key]

# create new dictionaries
aDict3 = {}
aDict4 = {}

# algorithm like the one above, modified to cluster for low correlation
def corrcluster_low(cut):
   sortCorr2 = sorted(sortCorr, reverse=False)
   lowerbound = min([x for x in range(0, len(sortCorr2)) if sortCorr2[x][0] > -cut])
   upperbound = max([x for x in range(0, len(sortCorr2)) if sortCorr2[x][0] <= cut])
   cutoff = range(lowerbound, upperbound)
   for i in col:
       aDict3[i] = [i]
   for i in cutoff:
       for a in col:
           aDict4[a] = [0]
       input2 = sortCorr2[i]
       startingpoint = input2[1]
       endingpoint = input2[2]
       for k in aDict3[startingpoint]:
           for m in aDict3[endingpoint]:
               for j in range(0,len(sortCorr2)):
                   if sortCorr2[j][1] == m and sortCorr2[j][2] == k:
                       aDict4[startingpoint].append(abs(sortCorr2[j][0]))
                   if sortCorr2[j][2] == m and sortCorr2[j][1] == k:
                       aDict4[endingpoint].append(abs(sortCorr2[j][0]))
       if all (o <= cut for o in aDict4[startingpoint]):
           if all (b <= cut for b in aDict4[endingpoint]):
               for s in aDict3[startingpoint]:
                   if s not in aDict3[endingpoint]:
                       aDict3[endingpoint].append(s)
               for t in aDict3[endingpoint]:
                   if t not in aDict3[startingpoint]:
                       aDict3[startingpoint].append(t)
   return aDict3                
               
#try function for at most 0.1 correlation               
lowcorr = corrcluster_low(0.1)             

# copy clusters into new dictonary
lowcorr_clean = {}
for key in lowcorr.keys():
   if len(lowcorr[key]) == 1:
       lowcorr[key] = [1]
   if lowcorr[key] != 1:
       for i in range(1, len(lowcorr[key])):        
           lowcorr[lowcorr[key][i]] = [1]
   if len(lowcorr[key]) > 1:
       lowcorr_clean[key] = lowcorr[key]

####New Approach - graphical representation

# normalising prices for newly created cluster (new approach)
clust3temp = pd.DataFrame(columns = highcorr_clean, index = dailyRet3.index)
for i in clust3temp.columns:
    for k in dailyRet3.columns:
        if i == k:
            clust3temp[i] = dailyRet3[k]
   

#New Approach graphical representation
import matplotlib.pyplot as plt  
#plotting the graph (new approach)
clust3temp.index = pd.to_datetime(clust3temp.index)
ax = clust3temp.plot(legend=None, title="New Approach")
ax.set_ylim(0.2,1.8)
plt.show()

#### Old Approach graphical representation
clust1temp.index = pd.to_datetime(clust1temp.index)
ax2=clust1temp.plot(legend=None, title="Old Approach")
ax2.set_ylim(0.2,1.8)
plt.show()

