#Import packages
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler 

get_ipython().magic('matplotlib inline')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import os
os.getcwd()

################################
#####Import and clean data######
################################

#Define data filepath
dataPath='../modeling/ref_complete1/'


#Get Files and store in appropriate list
judgementList=glob.glob(dataPath+'signal_judgements*')
networkList=glob.glob(dataPath+'signal_network*')
semAcomList=glob.glob(dataPath+'signal_semACOM*')
semContextList=glob.glob(dataPath+'signal_semContext*')
sentimentList=glob.glob(dataPath+'signal_sentiment*')

#For each variable extract files and create total dataframe using only desired columns
judgementDF= pd.concat((pd.read_csv(fileName) for fileName in judgementList))[['group','avgPercJ','avgNumJ']].set_index('group')

networkDF= pd.concat((pd.read_csv(fileName) for fileName in networkList))[['group','subgraph_centrality','eigenvector_centrality']].set_index('group')

semAcomDF= pd.concat((pd.read_csv(fileName) for fileName in semAcomList))[['group','acom']].set_index('group')

semContextDF= pd.concat((pd.read_csv(fileName) for fileName in semContextList))[['groupName','t.cvCosineSim.']]
semContextDF=semContextDF.groupby('groupName').mean()
semContextDF.reset_index(inplace=True)
semContextDF.columns=['group','contextVec']
semContextDF=semContextDF.set_index('group')

sentimentDF= pd.concat((pd.read_csv(fileName) for fileName in sentimentList))[['group','X.PosWords','X.NegWords','X.PosDoc','X.NegDoc']].set_index('group')

#Merge dataframes into one based on groupname
signalDF=judgementDF.join([networkDF,semAcomDF,semContextDF,sentimentDF], how='left')
signalDF.reset_index(inplace=True)

#Add in group ranking
groupNameList=['WBC', 'PastorAnderson', 'NaumanKhan', 'DorothyDay', 'JohnPiper', 'Shepherd',
'Rabbinic', 'Unitarian', 'MehrBaba']
groupRankList=[1,2,3,4,4,4,6,7,8]

groupRankDF=pd.DataFrame([[groupNameList[i],groupRankList[i]] for i in range(len(groupNameList))],columns=['groupName','rank'])

signalDF['groupName']=signalDF['group'].map(lambda x: x.split('_')[0])

signalDF=signalDF.merge(groupRankDF, on='groupName')

##################################
#####Review Consolidated Data#####
##################################

signalDF.describe()

#Create Rank histogram
sns.distplot(signalDF['rank'],kde=False)
plt.suptitle('Rank Distribution')

#Create box plots
ax = sns.boxplot(x='rank',y='avgPercJ',data=signalDF)
fig= ax.get_figure()
plt.suptitle('Average Percent of Judgements versus Rank') 

ax = sns.boxplot(x='rank',y='avgNumJ',data=signalDF)
fig= ax.get_figure()
plt.suptitle('Average Number of Judgements versus Rank') 

ax = sns.boxplot(x='rank',y='subgraph_centrality',data=signalDF)
fig= ax.get_figure()
plt.suptitle('Subgraph Centrality versus Rank') 

ax = sns.boxplot(x='rank',y='eigenvector_centrality',data=signalDF)
fig= ax.get_figure()
plt.suptitle('Eigenvector Centrality versus Rank') 

ax = sns.boxplot(x='rank',y='acom',data=signalDF)
fig= ax.get_figure()
plt.suptitle('Distributional Score versus Rank') 

ax = sns.boxplot(x='rank',y='contextVec',data=signalDF)
fig= ax.get_figure()
plt.suptitle('Context Vector Similarity versus Rank') 

ax = sns.boxplot(x='rank',y='X.PosWords',data=signalDF)
fig= ax.get_figure()
plt.suptitle('Fraction of Positive Words versus Rank') 

ax = sns.boxplot(x='rank',y='X.NegWords',data=signalDF)
fig= ax.get_figure()
plt.suptitle('Fraction of Negative Words versus Rank')

ax = sns.boxplot(x='rank',y='X.PosDoc',data=signalDF)
fig= ax.get_figure()
plt.suptitle('Fraction of Positive Documents versus Rank') 

ax = sns.boxplot(x='rank',y='X.NegDoc',data=signalDF)
fig= ax.get_figure()
plt.suptitle('Fraction of Negative Documents versus Rank') 

#Create Scaled dataframe
scaleColList=[x for x in signalDF.columns if x not in ['group','groupName','rank']]
sc = StandardScaler()
sc=sc.fit(signalDF[scaleColList])
signalStdDF= pd.DataFrame(sc.transform(signalDF[scaleColList]),columns=scaleColList)
signalStdDF[['group','groupName','rank']]=signalDF[['group','groupName','rank']]
signalStdDF.head()

#Create box plots
ax = sns.boxplot(x='rank',y='avgPercJ',data=signalStdDF)
fig= ax.get_figure()
plt.suptitle('Average Percent of Judgements versus Rank') 

ax = sns.boxplot(x='rank',y='avgNumJ',data=signalStdDF)
fig= ax.get_figure()
plt.suptitle('Average Number of Judgements versus Rank') 

ax = sns.boxplot(x='rank',y='subgraph_centrality',data=signalStdDF)
fig= ax.get_figure()
plt.suptitle('Subgraph Centrality versus Rank') 

ax = sns.boxplot(x='rank',y='eigenvector_centrality',data=signalStdDF)
fig= ax.get_figure()
plt.suptitle('Eigenvector Centrality versus Rank') 

ax = sns.boxplot(x='rank',y='acom',data=signalStdDF)
fig= ax.get_figure()
plt.suptitle('Distributional Score versus Rank')

ax = sns.boxplot(x='rank',y='contextVec',data=signalStdDF)
fig= ax.get_figure()
plt.suptitle('Context Vector Similarity versus Rank') 

ax = sns.boxplot(x='rank',y='X.PosWords',data=signalStdDF)
fig= ax.get_figure()
plt.suptitle('Fraction of Positive Words versus Rank')

ax = sns.boxplot(x='rank',y='X.NegWords',data=signalStdDF)
fig= ax.get_figure()
plt.suptitle('Fraction of Negative Words versus Rank') 

ax = sns.boxplot(x='rank',y='X.PosDoc',data=signalStdDF)
fig= ax.get_figure()
plt.suptitle('Fraction of Positive Documents versus Rank') 

ax = sns.boxplot(x='rank',y='X.NegDoc',data=signalStdDF)
fig= ax.get_figure()
plt.suptitle('Fraction of Negative Documents versus Rank') 

