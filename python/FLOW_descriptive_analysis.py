get_ipython().magic('matplotlib inline')
import pandas as pd
import matplotlib.pyplot as plt
import preprocess
from pg import DB
import numpy as np
import configparser

def PHistogram(title, xlabel, ylabel, df, col, color):
    ''' This function creates a new plot and plot histogram based on parameters given.
        INPUT: title: plot title
                xlabel, ylabel: axes labels
                df: dataframe to be plotted
                col: column name in the df to be plotted
                color: color of bars'''
    maxValue = max(df[col])    
    fig, ax = plt.subplots(figsize = [maxValue/2,5])

    bins = [i+0.5 for i in list(range(maxValue))]
    n,bins,patches=plt.hist(df[col],bins=bins,color=color)
    
    plt.xlim([0.5,maxValue+0.5])
    plt.ylim([0,14000])
    ax.get_xaxis().set_visible(False)
    for count,x in zip (n,bins[:-1]):
        ax.annotate(str(int(count)), xy=(x+0.5,0), xycoords=('data', 'axes fraction'),
            xytext=(0, -25), textcoords='offset points', va='top', ha='center')
        ax.annotate(str(int(x+0.5)), xy=(x+0.5,0), xycoords=('data', 'axes fraction'),
            xytext=(0, -10), textcoords='offset points', va='top', ha='center')
    plt.subplots_adjust(bottom=0.15)
    plt.title(title)
    ax.annotate(xlabel, xy=(maxValue+0.5,0), xycoords=('data', 'axes fraction'), 
        xytext=(0, -10), textcoords='offset points', va='top', ha='left')
    ax.set_ylabel(ylabel)

CONFIG = configparser.ConfigParser()
CONFIG.read('db.cfg')
dbset = CONFIG['DBSETTINGS']
db = DB(dbname=dbset['database'],host=dbset['host'],user=dbset['user'],passwd=dbset['password'])

data = pd.read_csv('coverage_summary.csv')
data = data.dropna()
data['year'] = data['year'].astype(int)
data['month'] = data['month'].astype(int)
data['season'] = data.apply(preprocess.makeseasons,axis=1)
centrelines = pd.DataFrame(db.query('SELECT centreline_id, feature_code_desc FROM prj_volume.centreline').getresult(), columns = ['centreline_id','feature_code_desc'])
data = data.merge(centrelines, on='centreline_id', how='inner')

data1 = pd.DataFrame(data.groupby(['centreline_id','dir_bin']).year.nunique())
data1.reset_index(inplace=True)

PHistogram('Distribution of Segments w.r.t. Number of Years Counted','Number of Years Counted','Number of Segments',data1,'year','blue')

data1 = data[data['count_type']==1]    
data1 = pd.DataFrame(data1.groupby(['centreline_id','dir_bin']).year.nunique())
data1.reset_index(inplace=True)

PHistogram('Distribution of Segments w.r.t. Number of Years Counted (ATR)','Number of Years Counted (ATR)','Number of Segments',data2,'year','blue')

data1 = pd.DataFrame(data.groupby(['centreline_id','dir_bin','season']).year.nunique())
data1.reset_index(inplace=True)

PHistogram('Frequency of Segment Counts in Spring','Number of Years Counted','Number of Segments (Spring)',data3[data3['season']=='Spring'],'year','green')
PHistogram('Frequency of Segment Counts in Summer','Number of Years Counted','Number of Segments (Summer)',data3[data3['season']=='Summer'],'year','red')
PHistogram('Frequency of Segment Counts in Autumn','Number of Years Counted','Number of Segments (Autumn)',data3[data3['season']=='Autumn'],'year','orange')
PHistogram('Frequency of Segment Counts in Winter','Number of Years Counted','Number of Segments (Winter)',data3[data3['season']=='Winter'],'year','silver')

fig, ax = plt.subplots(figsize = [10,5])
data1 = data.groupby(['year'],as_index=False).count()
plt.plot(data1['year'],data1['centreline_id'],label="ATR+TMC")
data1 = data[data['count_type'] == 1].groupby(['year'],as_index=False).count()
plt.plot(data1['year'],data1['centreline_id'],label="ATR")
plt.title('Number of Segment/Direction/Day Counted Each Year')
plt.legend(loc=2)

fig, ax = plt.subplots(figsize = [10,5])
data1 = data[(data['dow'] == 0) | (data['dow'] == 6)]
data1 = data1.groupby(['year','count_type'],as_index=False).count()
ax.bar(data1[data1['count_type']==1]['year'],data1[data1['count_type']==1]['centreline_id'],0.5,color='r',label='ATR')
ax.bar(data1[data1['count_type']==2]['year']+0.5,data1[data1['count_type']==2]['centreline_id'],0.5,color='b',label='TMC')
plt.title('Number of Segment/Direction/Day Counted Each Year on Weekends')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize = [10,5])
data1 = data[(data['dow']<0) | (data['dow']<6)]
data1 = data1.groupby(['year','count_type'],as_index=False).count()
ax.bar(data1[data5['count_type']==1]['year'],data1[data1['count_type']==1]['centreline_id'],0.5,color='r',label='ATR')
ax.bar(data1[data5['count_type']==2]['year']+0.5,data1[data1['count_type']==2]['centreline_id'],0.5,color='b',label='TMC')
plt.title('Number of Segment/Direction/Day Counted Each Year on Weekdays')
plt.legend()
plt.show()

data1 = data[data['year']>2009]
data1 = data1[['year','month','count_type','dir_bin','centreline_id']].drop_duplicates()
data1['year'] = data1['year'].astype(int)
fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(15,5))
ax1.set_xlim(1,12)
ax2.set_xlim(1,12)
for (year,count_type),group in data1.groupby(['year','count_type']):
    g1 = group.groupby(['month'],as_index=False).size()
    if count_type == 1:
        ax1.plot(g1.index,g1.values,label=int(year),linewidth=2)
    else:
        ax2.plot(g1.index,g1.values,label=int(year),linewidth=2)
plt.legend()
plt.title('Number of Segments by Month')
ax1.set_title('ATR')
ax2.set_title('TMC')
ax1.set_xlabel('Month')
ax2.set_xlabel('Month')
ax1.set_ylabel('Number of Segments')
plt.show()

# Time in scope
data1 = data[data['year']>2000]
data1 = pd.DataFrame(data1.groupby(['year','feature_code_desc']).centreline_id.nunique())
data1.reset_index(inplace=True)
# Set road types in scope
combine = list(data1['feature_code_desc'].unique())
combine.remove('Local')
combine.remove('Major Arterial')
combine.remove('Major Arterial Ramp')
combine.remove('Minor Arterial')
combine.remove('Collector')
combine.remove('Collector Ramp')
combine.remove('Expressway')
combine.remove('Expressway Ramp')
combine.remove('Laneway')

data1['year'] = data1['year'].astype(int)

# Construct Cross Tabulation
data1 = pd.crosstab(data1['year'],data1['feature_code_desc'],values=data1.centreline_id,aggfunc=np.sum,margins=True)
# Combine minor road Types
data1['Others'] = data1[combine].sum(axis=1)
data1.drop(combine, axis=1, inplace=True)
data1 = data1[list(data1.columns[0:-2]) + list(data1.columns[-1:]) + list(data1.columns[-2:-1])]
del data1.index.name
del data1.columns.name
data1 = data1.fillna(0)
data1[data1.columns] = data1[data1.columns].astype(int)

data1

data_tvrc = pd.read_csv('ATR Daily Volume.csv')
thresholds = {'Expressway Ramp':45000,'Major Arterial':50000,'Major Arterial Ramp':10000,'Minor Arterial':20000,'Collector':10000,'Local':4000,'Laneway':500}
for (fc,fc_desc),group in data_tvrc.groupby(['feature_code','feature_code_desc']):
    maxValue = (max(group['sum'])//1000+1) * 1000 
    fig, ax = plt.subplots(figsize = [15,4])
    bins = np.linspace(0,maxValue,21)
    n,bins,patches=plt.hist(group['sum'],bins=bins,color='b')
    binsize = bins[2] - bins [1]
    for count,x in zip (n,bins[:-1]):
        ax.annotate(str(int(count)), xy=(x+binsize/2,0), xycoords=('data', 'axes fraction'),
            xytext=(0, -25), textcoords='offset points', va='top', ha='center')
    plt.subplots_adjust(bottom=0.15)
    plt.title(fc_desc)
    try:
        plt.axvspan(thresholds[fc_desc], thresholds[fc_desc]+binsize/15, facecolor='r', alpha=1)
    except:
        pass
    ax.annotate('Volume', xy=(max(group['sum'])+300,0), xycoords=('data', 'axes fraction'), 
        xytext=(0, -12), textcoords='offset points', va='top', ha='left')
    ax.set_ylabel('Days')
    ax.set_xticks(bins)

db.close()

