import os
os.getcwd()

# reading cdhit results `.clstr`

# parse `.clstr`

def parseCluster(filename):
    
    # read filename.clstr
    f = open('../../cdhitResult/'+filename+'.clstr')
    lines = f.readlines()
    f.close()
    
    # count store in list
    clusterName = []
    clusterCount = []

    # parse it
    i = 0 # line index
    for line in lines:
        
        if line[0] == '>':
            #print(line.rstrip()[1:]) # Cluster 0
            clusterName.append(line.rstrip()[1:])
            if i > 0:
                #print(int(lines[i-1].split('\t')[0]) + 1) # last cluster's last entry +1 = how many genes
                clusterCount.append(int(lines[i-1].split('\t')[0]) + 1)
        if i == len(lines)-1:
            clusterCount.append(int(lines[i].split('\t')[0]) + 1)
            #print(len(line))
        i = i+ 1
    return(clusterName, clusterCount)
    

# test 
name , count = parseCluster('95cdhit')
print(len(name),len(count))
print(count)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.bar(range(len(count)),count)

plt.hist(count, bins = 100)

# generating /prescence data
from pandas import DataFrame
df = DataFrame(columns = name)
df

# prescence, abscence data
filename = '95cdhit'
f = open('../../cdhitResult/'+filename+'.clstr')
lines = f.readlines()
f.close()

for line in lines:
        
        if line[0] == '>':
            #print(line.rstrip()[1:]) # Cluster 0
            columnName = line.rstrip()[1:]
        else:
            ID = line.split('|')[1].split('...')[0]
            df.loc[ID, columnName] = True
df.head()
            

df = df.fillna(False)

countOfCore = df.sum(axis = 0)
countOfGenome = df.sum(axis = 1)

len([x for x in list(countOfCore) if x == 50]) # core genome size = 2944

countOfGenome
# each genome has 4500-5000 genes, with 2944 core genome and about 1500-2000 accessory genes

df.to_excel('../../cdhitResult/cdhitPattern.xlsx')
# save to excel and then plot with R

