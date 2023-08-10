import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math
import gzip
get_ipython().magic('matplotlib')

def openFile(name, mode):
    if name.lower().endswith('.gz'):
        return gzip.open(name, mode+'b')
    else:
        return open(name, mode)

dataFile = '/Users/dgrossman/data/spmf_fpgrowth_output.txt'
dataFile = '/Users/dgrossman/work/netStorage/davec/tbird.TFIDF.60s.FPGrowth.minsup002'

dataFile = '/Users/dgrossman/Downloads/PARIS_Results_500k.py'

#dataFile = '/Users/dgrossman/work/netStorage/davec/8Jan2016/tbird.log.preProc.200.supports.30sec.Transactions.FPGrowth.3minsup'
#dataFile = '/Users/dgrossman/work/netStorage/davec/8Jan2016/tbird.log.preProc.200.supports.20sec.Transactions.FPGrowth.02minsup'

dataClusters = '/Users/dgrossman/data/tbird.log.preProc.200.supports.clusters'

import re
edgeSet = set()
edgeDict = defaultdict(int)
procLine = list()
clusterDict = dict()

nodeFile = openFile(dataFile,'r')
clusterFile = openFile(dataClusters,'r')

#preproc for some later viz
for line in clusterFile:
    cluster,text = line.strip().split(',',1)
    text = re.sub(r'\s',' ',text)
    text = re.sub(r'\\','',text)
    text = re.sub(r'\(:\? ',' (:?',text)
    text = re.sub(r'\$','',text)
    text = re.sub(r'([\[\]\{\}\(\):]) ',r'\1',text)
    text = re.sub(r' ([\[\]\{\}\(\):])',r'\1',text)
    
    #kill the commas since they will break the CSV read used by the vis
    text = re.sub(r',','',text)
    clusterDict[cluster] = " ".join(text.split())
    
#make sure that the junkdrawer is initialized
clusterDict['-1'] = 'JunkDrawer'

#handle the 2 types of MBA output formats
for line in nodeFile:
    if re.search('#SUP',line):
        procLine.append(line.strip().split('#',1)[0].strip())
    else:
        procLine.append(line.strip())

# quick check that things look ok
#print procLine[:5]
#print clusterDict['1']

import itertools

for p in procLine:
    l = p.split(' ')
    if len(l) > 1:
        comb = itertools.combinations(l, 2)
        for start,finish in comb:
            val = (start,finish)
            edgeDict[val] += 1
            edgeSet.add(val)    

import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt


G=nx.Graph()
G.add_edges_from(edgeDict.iterkeys())
components = list(nx.connected_components(G))
nodeweight = defaultdict(int)

# make some relative size based on frequency
for k,v in edgeDict.iteritems():
    nodeweight[k[0]] += v
    nodeweight[k[1]] += v

#make the canvas the correct size for the number of
#images to be generated
numFigs = math.ceil(math.sqrt(len(components)))

fig = plt.figure()
edge2PicDict = dict()

for e in edgeDict.iterkeys():
    for index,c in enumerate(components):
        if e[0] in c:
            edge2PicDict[e]=index
            
temp = 1
for c in components:
    wordWeight = defaultdict(int)
    wordList = list()
    for cID in c:    
        for word in clusterDict[cID].split():
            #dont output punctuation
            if word not in  [' ','{','}','[',']',':'] :
                wordWeight[word] += nodeweight[cID]
    for k,v in wordWeight.iteritems():
        wordList.append((k,v))
    
    wordcloud = WordCloud(max_font_size=100,width=640,height=480, relative_scaling=.5).generate_from_frequencies(wordList)
    ax= plt.subplot(numFigs,numFigs,temp)
    plt.title('component:%i n:%i' % (temp-1,len(c)))
    plt.imshow(wordcloud)
    # puts the graphs into a file
    wordcloud.to_file('wordCloud%i.png'%(temp-1))
    plt.axis("off")
    
    temp=temp+1
    
plt.show()
fig.savefig('allWordCloud.png')

import math
dataOutFile = '/Users/dgrossman/work/magichour/d3/data3.csv'
header = 'source,target,image,value,sTitle,tTitle\n'
outFile = openFile(dataOutFile,'w')
outFile.write(header)
for edge,count in edgeDict.iteritems():
    o =  '%s,%s,%s,%f,%s %s,%s %s\n' % (edge[0],edge[1],'wordCloud%i.png'%(edge2PicDict[edge]),
                                        math.log(int(count))+1,
                                        edge[0],clusterDict[edge[0]],
                                        edge[1],clusterDict[edge[1]])
    outFile.write(o)
outFile.close()



