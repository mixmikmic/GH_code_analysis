import numpy as np
from collections import Counter
import pandas as pd
#import matplotlib
import math
from sklearn.decomposition import PCA

# Make the graphs a bit prettier, and bigger
#pd.set_option('display.width', 5000) 
#pd.set_option('display.max_columns', 60) 

numberOfCharacterAttributes=3
numberOfnumericalAttributes=6
numberOfAttributes=numberOfCharacterAttributes+numberOfnumericalAttributes
numberOfSamples=5
 #This represents the ip address of the dataset ,or any other heavystring data preventing truncation
characterDataType="<U32"
#This is the number of rows imported from csv file.
DATASET_SIZE=20 
#This is the name of dataset that is imported derive the file path correctly if file has be changed
name_OF_FILE='dataset' 
generalizedData=0

#All dataset Parameters
dataset=pd.read_csv(name_OF_FILE)
dataset=dataset.drop(['No.'],axis=1)
dataset=np.array(dataset)

#comment the below line while computing on complete data
#for performance measure only selected 100 of records
dataset=dataset[np.random.randint(dataset.shape[0],size=100),:]
sampleData=dataset[np.random.randint(dataset.shape[0], size=DATASET_SIZE), :]

#data is an array of the attributes known as 'e'.
#a single data row
def numericalAndCharacterDivision(data):
    '''

    This function divides a single row of data set to numerical and character attributes
    Parameter:a single row of data
    Returns:Object of numerical attributes ,character attributes
    
    '''
    #get the single row of data
    #divide into two different group
    division=np.split(data,[numberOfCharacterAttributes])# 3 is used as in the set of attributes first 3 data
    #are of character attributes and rest of them are numerical attributes
    #return as object composed of numerical and character attributes    
    return (division[0],division[1])

#numericalAndCharacterDivision(np.array([1,2,3,4,5,6,7,8,9]))
#returns (array([1, 2, 3]), array([4, 5, 6, 7, 8, 9]))
    



def count(data):
    
    '''
    this function computes data as list
    '''
    
    #retrieving numerical and character attributes from data
    #numerical,character=data
    #numerical=np.array(numerical)
    shapeOfData=np.array(data).shape
    numberOfRows=shapeOfData[0]
    numberOfCol=shapeOfData[1]
    
    #initializing the character and numerical lists based on the dataset
    numerical=np.zeros([numberOfRows,numberOfnumericalAttributes])
    character=np.zeros([numberOfRows,numberOfCharacterAttributes],dtype=characterDataType)
    
    #insert the data based on the attributes so that calculation would be easy
    for i in range(np.array(data).shape[0]):
        characterAttributes,numericalAttributes=numericalAndCharacterDivision(np.array(data)[i])
        #print("n: ",numericalAttributes)
        numerical[i]=numericalAttributes
        character[i]=characterAttributes
        #print("C: ",characterAttributes.dtype)

    
    #Preprocessing, which replaces every 'nan' item from the data to '0'
    j=0
    for i in numerical:
        #print(numerical.shape)
        for k in range(numerical.shape[1]):            
            #print(numerical[j][k])
            if(numerical[j][k].astype('str') == 'nan'):
                numerical[j][k]=generalizedData              
        j=j+1
    #print("new Character: ",character)
    
    #numerical data as:[[2,3,4,5],[5,6,7,8]]
    #For Pn:    
    #its sum should be as 2+5=7,3+6=9,4+7=11,5+8=13
    #and divide the sum by the total number of items
    
    #print("numerical: ",numerical.astype('int'))
    numerical=numerical.astype('int')
    numericalSum=numerical.sum(axis=0)
    numericalRowSize=numerical.shape[0]

    #print("Column Size: ",numericalRowSize)
    #print("Sum: ",numericalSum)
    
    Pn=(numericalSum/numericalRowSize).astype('int')
    #print("Pn: ",Pn)
    #print("Sum: ",numericalSum)
    #print("Row: ",numericalRowSize)
   
    #For Ps
    #Ps is the frequent character attribute set which consist of q(q=m-p) most frequent character attribute
     
    Ps=np.empty([numberOfCharacterAttributes],dtype=characterDataType)
    
    #print("Character: ",character)
    
    splittedColumn=np.hsplit(character,numberOfCharacterAttributes)
    
    j=0
    
    #the splitted column is flattened that merges the data to be in same list
    #most common determines the frequency of element in the list
    #most_common(1) returns the first element from the count received sorted in descending order
    #Ps is provided with single data fitted to be made as a single row 'e'
    for i in splittedColumn:
        #print("Pre: ",i)
        Ps[j]=Counter(i.flatten()).most_common(1)[0][0]
        j=j+1
    
    #print("Ps: ",Ps)
    
    
    #All the datarow can be set as character set to prevent the loss of information from the character set.
    P=np.empty(numberOfAttributes).astype(characterDataType)
    j=0
    
    #From Ps and Pn we extract every element and merge to P to create a single dataset row
    for i in Ps:
        P[j]=i.astype(characterDataType)
        j=j+1
    
    for i in Pn:
        P[j]=i
        j=j+1
    
    return P


#computes the count of a single  atributes composed of both (numerical , character) attributes
result=count([['71.126.222.64','b','c',57272.0, 80.0 ,1.0 ,1.0, 52.0, 5840.0],['71.126.222.64','x','c',1,7,1,2,4,5],
              ['192.168.0.1','b','q',1,3,9,1,6,4]])
print("Result: ",result)

def nSimilarity(ei,ej):
    
    '''
    Parameter:ei and ej are the two records in the collection of dataset
    returns:Similarity based on classical euclidean distance
    This calculates only the similarity of numerical attribute
    ei and ej only contains the numerical attributes
    '''
        
    characterAttributeEi,ei=numericalAndCharacterDivision(ei)
    characterAttributeEj,ej=numericalAndCharacterDivision(ej)
    
    
    #Preprocessing
    #All the nan elements are converted to 0 for calculation purpose
       
    j=0
    for i in range(ei.shape[0]):
        try:
            #print(ei[j].dtype)
            if(ei[j].astype('str') == 'nan'):
                ei[j]=generalizedData                        
        except:
            #ei[j]=generalizedData
            pass
        j=j+1
        
    #print(ei)
    #return
    
    j=0
    for i in range(ej.shape[0]):
        try:
            if(ej[j].astype('str')=='nan'):
                ej[j]=generalizedData
        except:
            #ej[j]=generalizedData
            pass
        j=j+1
        
    #ei[ei == 'nan']=0
    ei=ei.astype(np.float32)
    
    #ej[ej=='nan']=0
    ej=ej.astype(np.float32)
  
    #calculates the euclidean distance of parameters
    return np.sqrt(np.sum((ei-ej)**2)).astype(np.int32)
    
  
resultDistance=nSimilarity(np.array(['71.126.222.64', '126.120.0.39', 'TCP' ,'35641', '150' ,'434', '10710' ,'63',
 '19473']),
                           np.array(['71.126.222.64', '132.244.39.212' ,'TCP', 57259.0, 80.0 ,2751.0 ,35280.0, 52,
 60632.0]))
print("Similarity n: ",resultDistance)

def pSimilarity(ei,ej):
    '''
    Parameter:ei and ej are the two records in the collection of dataset
    returns:Similarity based on the frequency of attributes
    This function calculates the similarity of all atributes (that includes both numerical and character attributes)
    '''
     #needs the number of Hjk and Hik of ei and ej
    #Lets suppose ei and ej as:(based on e)
    #ei: 192.168.0.1,101.16.1.89,80,3764,45213,5,1,67,78
    #ej: 192.168.0.2,102,17,6,73,21,4562,17234,4,1,32,56
    #result=((9+9)/(9*9)*A)
    #A=0 if ei=ej else A=1
    
    A=1
    #computing A first
    #if Hik==Hjk then A=0 else A=1
    if np.array_equal(ei,ej):#checks if both ei and ej have same element is array
        #A=0
        return 0
    else:
        A=1
        #pSimilarity=0.67
    
    characterAttributeEi,numericalAttributeEi=numericalAndCharacterDivision(ei)
    characterAttributeEj,numericalAttributeEj=numericalAndCharacterDivision(ej)
    
    #This line computes for the number of character attributes
    #pSimilarity=((characterAttributeEi.shape[0] + characterAttributeEj.shape[0])
    #/(characterAttributeEi.shape[0] * characterAttributeEj.shape[0]))*A
    
    #Preprocessing   
    #This line computes for the number of character attributes but doesnot count the empty data
    #every row sometimes doesnot contain every element 
    #so this line excludes the empty element in the data row
    numberOfEi=np.array(list(filter(None,characterAttributeEi))).shape[0]
    numberOfEj=np.array(list(filter(None,characterAttributeEj))).shape[0]    
    
  
    #This line computes for the number of character attributes if any number of list is zero (0)
    if(numberOfEi==0 or numberOfEj==0):
        numberOfEi=characterAttributeEi.shape[0]
        numberOfEj=characterAttributeEj.shape[0]
        return (( numberOfEi+numberOfEj )/(numberOfEi* numberOfEj))*A
    
    
    pSimilarity=((numberOfEi + numberOfEj)/(numberOfEi * numberOfEj)) * A
    
    
    return pSimilarity 
    

resultPSimilarity=pSimilarity(np.array([1,2,3,4,5,6,7,8,9]),np.array([9,56,3,4,5,6,7,8,8]))
print("P Similarity: ",resultPSimilarity)

def similar(ei,ej):
    '''
    Parameter:ei and ej are the two records in the collection of dataset
    returns:Similarity based on the frequency of attributes and euclidean distance of numerical attributes
    '''
    #computes the similarity of both nSimilarity and pSimilarity
    return nSimilarity(ei,ej)+pSimilarity(ei,ej)
    

resultSimilar=similar(np.array(['',2,3,4,5,6,7,8,9]),np.array([1,2,3,40,5,6,70,8,9]))
print("Similar: ",resultSimilar)

def getAllClusterCenter(dataset):
    
    '''
    Parameter:Dataset contains the data rows which contains both the numerical and character attributes
    Returns:All cluster count
    '''
    
    
    #1.we sample the dataset into group i.e. break the dataset into numberOfSample Size provided
    #2.we get the center data(m[i]) for each sample using count function
    #we select the first element of the previous result as the first cluster(m) center
    #therefore m1=m[1]
    #For second cluster,we calcuate the Similarity of the selected first cluster(m) with every
    #other center data(m[i]) 
    #From Similarity we select the maximum similar clusters. therefore,m2=max(m1,m[i]) where i>1
    #return object as (m1,m2)
    
    #divides the dataset perfect based on the number of samples
    #if the dataset cannot be divided into equal number of samples,we decrease the size of dataset
    #to make equal number rows in samples    
    numberOfRows=dataset.shape[0] #initial number of rows in dataset
    while((numberOfRows % numberOfSamples)!=0):#calculating the modulus (reminder),decrease until reminder is zero
        dataset=np.delete(dataset,1,axis=0) #deletes any 1 row from the dataset
        numberOfRows=dataset.shape[0] #newly defined number of rows in dataset
        
        
    #splits the modified dataset into equal number of samples (group)
    samples=np.split(dataset,numberOfSamples,axis=0)
    
    #This consist of the collection of center from the samples
    countCollection=np.empty([numberOfSamples,numberOfAttributes],dtype=characterDataType)
       
    j=0
    for i in samples:
        countCollection[j]=count(i) #performs the count operation
        j=j+1
        
    return countCollection


def search(dataset):
    
    '''
    Parameter:Dataset contains the data rows which contains both the numerical and character attributes
    Returns:Create two initial Cluster
    '''
    
    
    
    countCollection=getAllClusterCenter(dataset)
       
    #print("Count Collection: ",countCollection)
    
    firstCenter=countCollection[0] #derived from m=m1
   
    #As alternative:
    
    #m=center {m1,m2, m3……..ml} 
    #Above may mean to determine the center from the center collected from samples
    #For that simply enable below first center lines and comment the above firstCenter lines:
    #firstCenter=count(countCollection)
    #print("First: ",firstCenter)
    
    #For second center we need to calcuate the similarity among the collected center i.e. count with first center
    
    #We need to select the max Similar from result;which means that we need to select the smallest value
    # as similar is near to zero i.e. Similar(1,1)=0 where as Similar(1,99)=97 (may be not,but not equal to 0)
    
    #initial variable
    minVariable=similar(firstCenter,countCollection[1]) 
    index=0
    j=0
    
    for i in countCollection:
        #Excluded the first center if they are same
        if(np.array_equal(firstCenter,i)==False): 
            similarity=similar(firstCenter,i)
            if(minVariable>similarity):
                minVariable=similarity
                index=j
        j=j+1
        
    secondCenter=countCollection[index]
       
    return (firstCenter,secondCenter)

def calculateMinSimilarityBetweenCluster(clusters):
    minClusterSimilar=similar(clusters[0],clusters[1])
    for k in range(clusters.shape[0]):
        # This skips previously determined similars
        p=k+1 
        for p in range(p,clusters.shape[0]):
            similarity=similar(clusters[k],clusters[p])
            if(similarity>minClusterSimilar):
                minClusterSimilar=similarity
            #print("Min Cluster Similar: Ci: %d"% k +" Cp: %d " % p +" Sim: %d " % minClusterSimilar)
    return minClusterSimilar

def addCluster(clusterCollection,clusterToAdd):
    #for addition of the new data as cluster
    newCluserSize=clusterCollection.shape[0]+1 
    newCluster=np.empty([newCluserSize,numberOfAttributes],dtype=characterDataType)
    newIndex=0
    #This loop just adds the previous elments to new cluster
    #todo optimize this
    for i in clusterCollection:
        newCluster[newIndex]=i
        newIndex=newIndex+1                    
        
    #This line adds the cluster to add to the new cluster
    newCluster[newIndex]=clusterToAdd.astype(characterDataType)
    return newCluster

def determineLabelForCluster(allCluster,firstCluster,secondCluster):
    clusterLabelDict={}
    
    #initial cluster center
    #firstCluster , secondCluster
    
    #Distance between two cluster is considered as similarity between the cluster
    #print("Distance from a cluster center to vh: ",similar(vh,allCluster[0]))
    distanceCollection=np.zeros([allCluster.shape[0],2])
    maxDistanceCollection=np.zeros(allCluster.shape[0])    
    
    for (i,cluster) in enumerate(allCluster):
        distanceCollection[i][0]=similar(firstCluster,allCluster[i])
        distanceCollection[i][1]=similar(secondCluster,allCluster[i])
        maxDistanceCollection[i]=max(distanceCollection[i][0],distanceCollection[i][1])
        
    averageDistance=sum(maxDistanceCollection)/maxDistanceCollection.shape[0]
    
    print("Distance After Processing: ",distanceCollection)
    print("Max Distance: ",maxDistanceCollection)
    print("Average Distance: ",averageDistance)
    
    for (i,cluster) in enumerate(allCluster):
        #cluster is normal if max distance is less than average distance
        if(maxDistanceCollection[i]<averageDistance):
            #this is normal
            clusterLabelDict[i]='no'
            #print("normal")
        else:
            #This is attack
            clusterLabelDict[i]='yes'
            #print("Attack")
        pass     
    
    
    #needs some more operation for labelling the cluster
    #currently just passing the cluster index as the label
   
    #for (index,ei) in enumerate(allCluster):
        #clusterLabelDict[index]=index        
            
    return clusterLabelDict

def preprocessEi(ei):
    for (j,ej) in enumerate(ei):            
        try:
            x=float(ej)
            if(math.isnan(x)==True):
                ei[j]=generalizedData
        except:
            pass
    return ei

def algorithmHeuristicCluster():
    firstCluster,secondCluster=search(sampleData)
    #Here either all cluster center can be used or only two cluster during initialization
    
    #This selects all the clusters determined during the sampling for initialization
    #allClusters=getAllClusterCenter(dataset)
    
    #This selects only two of the cluster for initialization
    allClusters=np.array([firstCluster,secondCluster])
    
    #dictionary has a key and its value
    #Here,key would be index of the ei and value would be the cluster index
    dictionaryClusterEi={}
    
    numberOfClusterFormed=np.shape(allClusters)[0]
    
    for eiIndex,ei in enumerate(dataset):
        #this block calculates the minSimilar between the ei and cj
        
        #*************
        j=0
        #initialValue which may be the greatest among the dataset
        minSimilar=similar(ei,allClusters[0]) 
        clusterIndex=0
        for cj in allClusters:
            similarity=similar(ei,cj)
            if(similarity<minSimilar):
                minSimilar=similarity
                clusterIndex=j
            j=j+1
            
        #************
        
        #This block calculates the minSimilar between the Cluster C
        
        minClusterSimilar=calculateMinSimilarityBetweenCluster(allClusters)         
        if(minSimilar>minClusterSimilar):
            #merges the ei to cluster   
            allClusters=np.concatenate((allClusters,np.array([ei],dtype=characterDataType)),axis=0) 
               
            print("Center Created: ",ei," With index: ",allClusters.shape[0]-1," with ei index: ",eiIndex)
            #Since ei is the new cluster and it is concatenated as last index
            dictionaryClusterEi[eiIndex]=allClusters.shape[0]-1            
            
            #number of cluster is not needed for operation its just for information
            numberOfClusterFormed=numberOfClusterFormed+1
                    
        else:
            #holds the index for cluster for every ei
            dictionaryClusterEi[eiIndex]=clusterIndex
    
        
    #creates the label to every cluster index
    clusterLabel=determineLabelForCluster(allClusters,firstCluster,secondCluster)
        
    #new data set with the size of previous dataset and cluster label as [n,10]
    #   where n = number of dataset or row,10 is number of column
    newDataset=np.zeros([dataset.shape[0],10],dtype=characterDataType)
    
    #appends the cluster index to every ei in data record
    for (index,ei) in enumerate(dataset):
        ei=np.append(ei,clusterLabel[dictionaryClusterEi[index]])
        
        #Preprocessing removes nan from the ei and replaces with the value of generalizedData set above
        #newDataset[index]=ei.astype('str')
        newDataset[index]=preprocessEi(ei)
        
        #To see the data that is being saved disable the comment
        #print("ei: ",ei," Cluster Label: ",clusterLabel[dictionaryClusterEi[index]])
        
        
    #print("Dictionary (ei,Cluster Index): ",dictionaryClusterEi)
    #print("Dictionary (clusterIndex,Class Label): ",clusterLabel)
    
    
    #This creates a new file containing ei and the cluster label
    dataFrame=pd.DataFrame(newDataset)
    dataFrame.to_csv("test.csv",header=None)
    
    #This block is test for visualization of data
        
    return numberOfClusterFormed

#for i in range(0,10):
print("number of cluster: ",algorithmHeuristicCluster())





