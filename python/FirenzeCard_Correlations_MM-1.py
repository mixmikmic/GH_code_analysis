import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pylab import *

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform, pdist

import sys
sys.path.append('../../src/')
from utils.database import dbutils

conn = dbutils.connect()
cursor = conn.cursor()

nodes = pd.read_sql('select * from optourism.firenze_card_locations', con=conn)

df = pd.read_sql('select * from optourism.firenze_card_logs', con=conn)
df['museum_id'].replace(to_replace=39,value=38,inplace=True)
df['short_name'] = df['museum_id'].replace(dict(zip(nodes['museum_id'],nodes['short_name'])))
df['string'] = df['museum_id'].replace(dict(zip(nodes['museum_id'],nodes['string'])))
df['date'] = pd.to_datetime(df['entry_time'], format='%Y-%m-%d %H:%M:%S').dt.date
df['hour'] = pd.to_datetime(df['date']) + pd.to_timedelta(pd.to_datetime(df['entry_time'], format='%Y-%m-%d %H:%M:%S').dt.hour, unit='h')
df['hour_of_day'] = pd.to_datetime(df['entry_time'], format='%Y-%m-%d %H:%M:%S').dt.hour
df['day_of_week'] = pd.to_datetime(df['entry_time'], format='%Y-%m-%d %H:%M:%S').dt.dayofweek
df['total_people'] = df['total_adults'] + df['minors']

df1 = df.groupby(['short_name','date']).sum()['total_people'].to_frame()
df1.reset_index(inplace=True)
df1.head()

df2 = df1.pivot(index='date',columns='short_name',values='total_people')
df2.head()

df2['Santa Croce'].plot()

df2['M. Ferragamo'].plot()

M = df2.corr(method='kendall')
M.head()

# For dropna(), how="any" is too liberal, but how="all" fails to do anything because the diagonal is always 1. 
# Solution: Convert diagonal to NAs, dropna(how="all"), then convert diagonal back to 1.
M.values[[np.arange(M.shape[0])]*2] = np.nan # This sets diagonal to NA
M.dropna(axis=0,how='all',inplace=True) 
M.dropna(axis=1,how='all',inplace=True) 
M.values[[np.arange(M.shape[0])]*2] = 1
M.fillna(0,inplace=True) # Replace all remaining NAs with 0
M.head()

fig = plt.figure(figsize=(10,10))#,dpi=300)
ax = fig.add_subplot(111)
cmap=plt.cm.Spectral

cax = ax.matshow(M,cmap=cmap)
fig.colorbar(cax)

ax.set_xticklabels(['']+M.index.tolist(),rotation=90)
ax.set_yticklabels(['']+M.index.tolist())

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

fig = figure(figsize=(13.5,10))
axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
D = M.as_matrix()
Y = linkage(D, method='single', metric='correlation')
Z = dendrogram(Y, orientation='left')
axdendro.set_xticks([])
axdendro.set_yticks([])

# Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
index = Z['leaves']
D = D[index,:]
D = D[:,index]
im = axmatrix.matshow(D, aspect='equal', origin='lower')
axmatrix.set_xticks([])
axmatrix.set_yticks([])

# Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
colorbar(im, cax=axcolor)

fig.show()

# Aggregate function. See workflow at bottom for what each step does. 
def corr_matrix(df,time_column='date',name='short_name',count='total_people'):
    df1 = df.groupby([name,time_column]).sum()[count].to_frame()
    df1.reset_index(inplace=True)
    df2 = df1.pivot(index=time_column,columns=name,values=count)
    M = df2.corr(method='kendall')
    return M

def corr_matrix_sorted(df,time_column='date',name='short_name',count='total_people'):
    df1 = df.groupby([name,time_column]).sum()[count].to_frame()
    df1.reset_index(inplace=True)
    df2 = df1.pivot(index=time_column,columns=name,values=count)
    M = df2.corr(method='kendall')
    M.values[[np.arange(M.shape[0])]*2] = np.nan
    M.dropna(axis=0,how='all',inplace=True)
    M.dropna(axis=1,how='all',inplace=True)
    M.values[[np.arange(M.shape[0])]*2] = 1
    M.fillna(0,inplace=True)    
    index = dendrogram(linkage(M.as_matrix(), method='single', metric='correlation'), orientation='top')['leaves']
    M = M.reindex(index=M.index[list(reversed(index))],columns=M.columns[list(reversed(index))])
    return M

M = corr_matrix_sorted(df,'hour')

fig = plt.figure(figsize=(15,15))#,dpi=300)
ax = fig.add_subplot(111)
cmap=plt.cm.Spectral

cax = ax.matshow(M,cmap=cmap)
fig.colorbar(cax)

ax.set_xticklabels(['']+M.index.tolist(),rotation=90)
ax.set_yticklabels(['']+M.index.tolist())

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

M = corr_matrix_sorted(df,'date')

fig = plt.figure(figsize=(15,15))#,dpi=300)
ax = fig.add_subplot(111)
cmap=plt.cm.Spectral

cax = ax.matshow(M,cmap=cmap)
fig.colorbar(cax)

ax.set_xticklabels(['']+M.index.tolist(),rotation=90)
ax.set_yticklabels(['']+M.index.tolist())

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

M = corr_matrix_sorted(df,'hour_of_day')

fig = plt.figure(figsize=(15,15))#,dpi=300)
ax = fig.add_subplot(111)
cmap=plt.cm.Spectral

cax = ax.matshow(M,cmap=cmap)
fig.colorbar(cax)

ax.set_xticklabels(['']+M.index.tolist(),rotation=90)
ax.set_yticklabels(['']+M.index.tolist())

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

df1 = df.groupby(['short_name','hour_of_day']).sum()['total_people'].to_frame()
df1.reset_index(inplace=True)
df2 = df1.pivot(index='hour_of_day',columns='short_name',values='total_people')
df2.head()

df2['Santa Croce'].plot()

df2['M. San Marco'].plot()

df2[['Santa Croce','M. San Marco']].corr(method='kendall')

df2[['Planetario','M. San Marco']]

M = corr_matrix_sorted(df,'hour_of_day')

fig = plt.figure(figsize=(15,15))#,dpi=300)
ax = fig.add_subplot(111)
cmap = plt.cm.Spectral

cax = ax.matshow(M[np.abs(M)>.4],cmap=cmap)
fig.colorbar(cax)

ax.set_xticklabels(['']+M.index.tolist(),rotation=90)
ax.set_yticklabels(['']+M.index.tolist())

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()





dists = pd.DataFrame(squareform(pdist(nodes[['latitude','longitude']])), columns=nodes['short_name'], index=nodes['short_name'])

# Sort rows by lat, sort columns by lon
dists = dists.reindex(index=nodes.sort_values('latitude')['short_name'],columns=nodes.sort_values('longitude')['short_name'])

fig = plt.figure(figsize=(16,16))#,dpi=300)
ax = fig.add_subplot(111)
cmap=plt.cm.Spectral

cax = ax.matshow(dists,cmap=cmap)
fig.colorbar(cax)

ax.set_xticklabels(['']+dists.index.tolist(),rotation=90)
ax.set_yticklabels(['']+dists.index.tolist())

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()



