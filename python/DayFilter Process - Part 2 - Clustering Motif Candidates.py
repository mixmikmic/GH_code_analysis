import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import json
from datetime import datetime

get_ipython().magic('matplotlib inline')

from scipy.cluster.vq import kmeans, vq, whiten
from scipy.spatial.distance import cdist
from sklearn import metrics
import numpy as np

import seaborn as sns
sns.set_context(context='talk', font_scale=2)

df = pd.read_csv("./MotifProfiles.csv", parse_dates=True, index_col='Date')

df.head()

df = df.drop(['SAXstring'], axis=1)

def timestampcombine_parse(date,time):
    #timestampstring = date+" "+time
#     date = datetime.strptime(date, "%Y-%M-%d")
    time = datetime.strptime(time, "%H:%M:%S").time()
    pydatetime = datetime.combine(date, time)

    #pydatetime = pydatetime.replace(tzinfo=pytz.UTC)
    #return pydatetime.astimezone(singaporezone).replace(tzinfo=None)
    return pydatetime

df = df.T.unstack().reset_index()
df['timestampstring'] = map(timestampcombine_parse, df.Date, df.level_1)
df.index = df.timestampstring
df = df.drop(['Date','level_1','timestampstring'],axis=1)
df.columns = ["CP_TotalChiller_kW"]
df = df.resample('H').mean()

df.head()

df_norm = (df - df.mean()) / (df.max() - df.min()) #normalized

df['Time'] = df.index.map(lambda t: t.time())
df['Date'] = df.index.map(lambda t: t.date())
df_norm['Time'] = df_norm.index.map(lambda t: t.time())
df_norm['Date'] = df_norm.index.map(lambda t: t.date())

dailyblocks = pd.pivot_table(df, values='CP_TotalChiller_kW', index='Date', columns='Time', aggfunc='mean')
dailyblocks_norm = pd.pivot_table(df_norm, values='CP_TotalChiller_kW', index='Date', columns='Time', aggfunc='mean')

dailyblocks_norm.head()

dailyblocksmatrix_norm = np.matrix(dailyblocks_norm.dropna())
centers, _ = kmeans(dailyblocksmatrix_norm, 4, iter=10000)
cluster, _ = vq(dailyblocksmatrix_norm, centers)

clusterdf = pd.DataFrame(cluster, columns=['ClusterNo'])

dailyclusters = pd.concat([dailyblocks.dropna().reset_index(), clusterdf], axis=1) 

dailyclusters.head()

x = dailyclusters.groupby('ClusterNo').mean().sum(axis=1).sort_values()
x = pd.DataFrame(x.reset_index())
x['ClusterNo2'] = x.index
x = x.set_index('ClusterNo')
x = x.drop([0], axis=1)
dailyclusters = dailyclusters.merge(x, how='outer', left_on='ClusterNo', right_index=True)

dailyclusters = dailyclusters.drop(['ClusterNo'],axis=1)
dailyclusters = dailyclusters.set_index(['ClusterNo2','Date']).T.sort_index()

dailyclusters.head()

clusterlist = list(dailyclusters.columns.get_level_values(0).unique())
matplotlib.rcParams['figure.figsize'] = 20, 7

styles2 = ['LightSkyBlue', 'b','LightGreen', 'g','LightCoral','r','SandyBrown','Orange','Plum','Purple','Gold','b']
fig, ax = plt.subplots()
for col, style in zip(clusterlist, styles2):
    dailyclusters[col].plot(ax=ax, legend=False, style=style, alpha=0.1, xticks=np.arange(0, 86400, 10800))

ax.set_ylabel('Total Daily Profile')
ax.set_xlabel('Time of Day')
plt.savefig("./graphics/clusters_total_overlaid_profiles.png")

def ClusterUnstacker(df):
    df = df.unstack().reset_index()
    df['timestampstring'] = map(timestampcombine, df.Date, df.Time)
    df = df.dropna()
    return df

def timestampcombine(date,time):
    pydatetime = datetime.combine(date, time)
    #pydatetime = pydatetime.replace(tzinfo=pytz.UTC)
    #return pydatetime.astimezone(singaporezone).replace(tzinfo=None)
    return pydatetime

dailyclusters.head()

dfclusterunstacked = ClusterUnstacker(dailyclusters)
dfclusterunstackedpivoted = pd.pivot_table(dfclusterunstacked, values=0, index='timestampstring', columns='ClusterNo2')

clusteravgplot = dfclusterunstackedpivoted.resample('D').sum().plot(style="^",markersize=15)
clusteravgplot.set_ylabel('Daily Totals kW Cooling Energy')
clusteravgplot.set_xlabel('Date')
clusteravgplot.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster')

plt.savefig("./graphics/clusters_overtime.png")

dfclusterunstackedpivoted['Time'] = dfclusterunstackedpivoted.index.map(lambda t: t.time())
dailyprofile = dfclusterunstackedpivoted.groupby('Time').mean().plot(figsize=(20,7),linewidth=3, xticks=np.arange(0, 86400, 10800))
dailyprofile.set_ylabel('Average Daily Profile kW Cooling')
dailyprofile.set_xlabel('Time of Day')
dailyprofile.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster')
plt.savefig("./graphics/clusters_averagedprofiles.png")

def DayvsClusterMaker(df):
    df.index = df.timestampstring
    df['Weekday'] = df.index.map(lambda t: t.date().weekday())
    df['Date'] = df.index.map(lambda t: t.date())
    df['Time'] = df.index.map(lambda t: t.time())
    DayVsCluster = df.resample('D').mean().reset_index(drop=True)
    DayVsCluster = pd.pivot_table(DayVsCluster, values=0, index='ClusterNo2', columns='Weekday', aggfunc='count')
    DayVsCluster.columns = ['Mon','Tue','Wed','Thur','Fri','Sat','Sun']
    return DayVsCluster.T

DayVsCluster = DayvsClusterMaker(dfclusterunstacked)
DayVsClusterplot1 = DayVsCluster.plot(figsize=(20,7),kind='bar',stacked=True)
DayVsClusterplot1.set_ylabel('Number of Days in Each Cluster')
DayVsClusterplot1.set_xlabel('Day of the Week')
DayVsClusterplot1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster')
plt.savefig("./graphics/clusters_dailybreakdown.png")

DayVsClusterplot2 = DayVsCluster.T.plot(figsize=(20,7),kind='bar',stacked=True, color=['b','g','r','c','m','y','k']) #, color=colors2
DayVsClusterplot2.set_ylabel('Number of Days in Each Cluster')
DayVsClusterplot2.set_xlabel('Cluster Number')
DayVsClusterplot2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("./graphics/cluster_numberofdays.png")

AnalysisLabel = "UWC Chiller kW"

NumberOfClusterList = range(5,8)
KmeansIterationsSettingList = [10000]

#This clustermaker is modified to take a whole time series and pivot to day-types within the function.
#This allows normalization and silhouhette calculation within the function
def ClusterMaker2(dailyblocks, dailyblocks_norm, clusterno, iterations):
    
    dailyblocksmatrix_norm = np.matrix(dailyblocks_norm.dropna())
    #dailyblocks = whiten(dailyblocks)
    centers, _ = kmeans(dailyblocksmatrix_norm, clusterno, iter=iterations)
    cluster, _ = vq(dailyblocksmatrix_norm, centers)
    
    #Calc Silhouette Coeff for each sample
    silhouttecoeff = metrics.silhouette_samples(dailyblocksmatrix_norm, cluster, metric='euclidean')
    silhouettecoeffscore = metrics.silhouette_score(dailyblocksmatrix_norm, cluster, metric='euclidean')
    
    #Calc Average within-cluster sum of squares
    Distance = cdist(dailyblocksmatrix_norm, centers, 'euclidean')
    #cIdx = np.argmin(Distance,axis=1)
    dist = np.min(Distance, axis=1)
    AvgWithinSS = sum(dist)/dailyblocksmatrix_norm.shape[0]
    
    clusterdf = pd.DataFrame(cluster, columns=['ClusterNo'])
    
    dailyclusters = pd.concat([dailyblocks.dropna().reset_index(), clusterdf], axis=1) 
    dailyclusters_norm = pd.concat([dailyblocks_norm.dropna().reset_index(), clusterdf], axis=1)
    
    #Reorder the cluster numbers so that the largest clusters are the highest numbers
    x = dailyclusters.groupby('ClusterNo').mean().sum(axis=1).sort_values()
    x = pd.DataFrame(x.reset_index())
    x['ClusterNo2'] = x.index
    x = x.set_index('ClusterNo')
    x = x.drop([0], axis=1)
    dailyclusters = dailyclusters.merge(x, how='outer', left_on='ClusterNo', right_index=True)
    
    #dailyclusters['Date2'] = pd.to_datetime(dailyclusters['Date'])
    #dailyclusters['ClusterNo2'] = dailyclusters['ClusterNo2'].astype(float)
    
    dailyclusters = dailyclusters.drop(['ClusterNo'],axis=1)
    dailyclusters = dailyclusters.set_index(['ClusterNo2','Date']).T.sort_index()

    return dailyclusters, silhouttecoeff, silhouettecoeffscore, AvgWithinSS

def timestampcombine(date,time):
    pydatetime = datetime.combine(date, time)
    #pydatetime = pydatetime.replace(tzinfo=pytz.UTC)
    #return pydatetime.astimezone(singaporezone).replace(tzinfo=None)
    return pydatetime

def ClusterUnstacker(df):
    df = df.unstack().reset_index()
    df['timestampstring'] = map(timestampcombine, df.Date, df.Time)
    df = df.dropna()
    return df

def DayvsClusterMaker(df):
    df.index = df.timestampstring
    df['Weekday'] = df.index.map(lambda t: t.date().weekday())
    df['Date'] = df.index.map(lambda t: t.date())
    df['Time'] = df.index.map(lambda t: t.time())
    DayVsCluster = df.resample('D').mean().reset_index(drop=True)
    DayVsCluster = pd.pivot_table(DayVsCluster, values=0, index='ClusterNo2', columns='Weekday', aggfunc='count')
    DayVsCluster.columns = ['Mon','Tue','Wed','Thur','Fri','Sat','Sun']
    return DayVsCluster.T

def ClusterAnalysisProcess(dfclusters, clusterno, noofiterations, nameofanalysis):
    #plt.get_cmap('Blues')
    Analysisname = nameofanalysis+'_'+str(clusterno)+'Clusters_'+str(noofiterations)+'Iterations'
    pp = PdfPages('./graphics/'+Analysisname+'.pdf')
    max_sec = 86400
    
    clusterlist = list(dfclusters.columns.get_level_values(0).unique())
    #rcParams['figure.figsize'] = 20, 7
    styles2 = ['LightSkyBlue', 'b','LightGreen', 'g','LightCoral','r','SandyBrown','Orange','Plum','Purple','Gold','b']
    fig, ax = plt.subplots()
    for col, style in zip(clusterlist, styles2):
        dfclusters[col].plot(ax=ax, legend=False, style=style, alpha=0.1, xticks=np.arange(0, max_sec, 10800))
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Total Daily Profile')
    ax.set_xlabel('Time of Day')
    pp.savefig()
        
    dfclusterunstacked = ClusterUnstacker(dfclusters)
    dfclusterunstackedpivoted = pd.pivot_table(dfclusterunstacked, values=0, rows='timestampstring', cols='ClusterNo2')
    
    clusteravgplot = dfclusterunstackedpivoted.resample('D').mean().plot(style="^",markersize=15)
    clusteravgplot.set_ylabel('Daily Totals '+nameofanalysis)
    clusteravgplot.set_xlabel('Date')
    clusteravgplot.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster')
    pp.savefig()
    
    dfclusterunstackedpivoted['Time'] = dfclusterunstackedpivoted.index.map(lambda t: t.time())
    dailyprofile = dfclusterunstackedpivoted.groupby('Time').mean().plot(figsize=(20,7),linewidth=3, xticks=np.arange(0, max_sec, 10800))
    dailyprofile.set_ylabel('Average Daily Profile '+nameofanalysis)
    dailyprofile.set_xlabel('Time of Day')
    dailyprofile.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster')
    #dailyprofile.xaxis.set_major_locator(dates.HourLocator())
    pp.savefig()

    DayVsCluster = DayvsClusterMaker(dfclusterunstacked)
    DayVsClusterplot1 = DayVsCluster.plot(figsize=(20,7),kind='bar',stacked=True)
    DayVsClusterplot1.set_ylabel('Number of Days in Each Cluster '+nameofanalysis)
    DayVsClusterplot1.set_xlabel('Day of the Week')
    DayVsClusterplot1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster')
    pp.savefig()

    DayVsClusterplot2 = DayVsCluster.T.plot(figsize=(20,7),kind='bar',stacked=True, color=['b','g','r','c','m','y','k']) #, color=colors2
    DayVsClusterplot2.set_ylabel('Number of Days in Each Cluster '+nameofanalysis)
    DayVsClusterplot2.set_xlabel('Cluster Number')
    DayVsClusterplot2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pp.savefig()
    pp.close()
    
    ClustersForExport = dfclusters.T.reset_index()
    #ClustersForExport = ClustersForExport[['Date','ClusterNo']] 
    ClustersForExport.to_csv(Analysisname+'Clusters.csv')
    
    dfclusterunstackedpivoted.to_csv(Analysisname+'PivotedClusters.csv')

    return DayVsCluster, dfclusterunstackedpivoted

SilhouetteMetricsList = []

for NumberOfClusters in NumberOfClusterList:
    for KmeansIterationsSetting in KmeansIterationsSettingList:
        print("Running clustering with "+str(NumberOfClusters)+" Clusters and "+str(KmeansIterationsSetting)+" Iteration Setting")
        TotalClusters, SilCoeff, SilScore, AvgWithinSS = ClusterMaker2(dailyblocks, dailyblocks_norm, NumberOfClusters, KmeansIterationsSetting)
        TotalDayVsCluster, TotalClusterStackedPivoted = ClusterAnalysisProcess(TotalClusters, NumberOfClusters, KmeansIterationsSetting, AnalysisLabel)
        
        Silhouttedf = pd.DataFrame(SilCoeff, columns=['SilCoeff'])
        ClusterQual = pd.concat([TotalClusters.T.reset_index(), Silhouttedf], axis=1)
        
        Analysisname = AnalysisLabel+'_Silhouette_'+str(NumberOfClusters)+'_'+str(KmeansIterationsSetting)
        ClusterQual.to_csv(Analysisname+'.csv')
        
        ClusterQualSummary = ClusterQual.groupby(by='ClusterNo2').mean()
        SilhouetteMetricsList.append([NumberOfClusters, KmeansIterationsSetting, ClusterQualSummary.T.ix['SilCoeff'].mean(), SilScore, AvgWithinSS])

