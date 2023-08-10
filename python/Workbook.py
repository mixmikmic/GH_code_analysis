from IPython.core.display import HTML
import requests
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import datetime
import seaborn as sns
from dateutil import parser
import pytz
import pylab as pl
from pyquery import PyQuery as pq
import numpy as np
import scipy
import pandas as pd
from lxml import etree
import os
from os import path
from pprint import pprint
import dateutil
import json
get_ipython().magic('matplotlib inline')

all_motions_json = []
for i in range(1, 637+1):
    try:
        j = json.load(open('jsons/%s.json' % i))
        all_motions_json.append(j)
    except:
        pass

j['fecha']

s_votes = []

for m in all_motions_json:
    v = m['votos']
    s = pd.DataFrame(v)
    s['sesion_id'] = m['sesion_id']
    s['datestr'] = m['fecha']
    s = s.set_index(['legislador_id', 'sesion_id'])
    s_votes.append(s)

df_votes = pd.concat(s_votes)

def vote_to_number(v):
    MAPPING = {
        'AFIRMATIVO': 1, # Yes
        'AUSENTE': 0, # Absent, did not come to the session
        'NEGATIVO': -1, # No
        'ABSTENCION': 0, # person is present, but not vote
    }
    return MAPPING.get(v, 0)
    
df_votes['vote-value'] = df_votes['voto'].apply(vote_to_number)

from datetime import date

df_votes['date'] = df_votes['datestr'].apply(lambda x: date(
        int(x.split('/')[2]), 
        int(x.split('/')[1]), 
        int(x.split('/')[0])
    ))
df_votes['datetime'] = df_votes['date'].apply(lambda d: datetime.datetime(*d.timetuple()[:7]))

#df_votes['date']
#10/12/2013 - 10/12/2015

s = df_votes['partido'].value_counts()
df_party_counts = pd.DataFrame(df_votes['partido'].value_counts()).reset_index()
df_party_counts.columns = ['party', 'count']
df_party_counts.to_csv('data/party-count.csv')

#df_votes

print('Total number of sessions:')
print(len(df_votes.reset_index()['sesion_id'].value_counts()))

print('Total number of legislators:')
print(len(df_votes.reset_index()['legislador_id'].value_counts()))

len(df_votes)

df_votes.to_csv('data/votes.csv')

df_member_profile = df_votes.reset_index()[['legislador_id', 'agrupacion', 'agrupacion_color', 'camara', 'mail', 'nombre', 'partido']].drop_duplicates()
print(len(df_member_profile))
df_member_profile = df_member_profile.drop_duplicates('legislador_id')
print(len(df_member_profile))
df_member_profile.to_csv('data/member-profile.csv')

df_member_profile[
    df_member_profile['legislador_id'] == 738
]

df_member_profile.set_index('legislador_id').ix[738]

df_selected_votes = df_votes[
    (df_votes['date'] >= date(2013, 12, 12))
    &
    (df_votes['date'] <= date(2015, 12, 12))
]

len(df_selected_votes)

df_votes_brief = df_selected_votes[['voto', 'vote-value']]

df_votes_brief['voto'].value_counts()

df_matrix = df_votes_brief.reset_index().pivot(index='legislador_id', columns='sesion_id', values='vote-value')

df_matrix = df_matrix.fillna(0)

df_matrix.to_csv('data/matrix-2013-2015.csv')

df_matrix.shape

X = np.matrix(df_matrix.fillna(0).as_matrix()).astype('int8')

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)
df_pc = pd.DataFrame(X_reduced, index=df_matrix.index, columns=['PC1', 'PC2', 'PC3'])

pca.explained_variance_

get_ipython().magic('matplotlib inline')
df_pc[['PC1', 'PC2']].plot(x='PC1', y='PC2', kind='scatter')

df_party_count_color = pd.read_csv('data/party-count-color.csv')

len(df_pc)

len(df_member_profile)

df_pc_member = pd.merge(df_pc.reset_index(), df_member_profile, how='left',
                        left_on='legislador_id', right_on='legislador_id')

len(df_pc_member)

len(df_pc)

plt.scatter(1, 2)
plt.scatter(2, 3)

plt.figure(figsize=(12, 6))
ax = plt.subplot(1, 1, 1)
from matplotlib.colors import ColorConverter

'''
for (i, m) in df_pc_member.iterrows():
    #print(m['PC1'])
    ax.scatter(m['PC1'], m['PC2'], color=m['_color'])
    #print(m)
'''

for (gname, group) in df_pc_member.groupby(['agrupacion', 'agrupacion_color']):
    print(gname)
    if gname == ('', ''):
        gname = ('null', '#000000')
    ax.scatter(group['PC1'], group['PC2'], color=gname[1], label=gname[0])

plt.legend(loc='upper left')
plt.xlabel('PC1')
plt.ylabel('PC2')

#df_pc[['PC1', 'PC2']].plot(x='PC1', y='PC2', kind='scatter')

get_ipython().magic('matplotlib inline')

plt.figure(figsize=(12, 6))
ax = plt.subplot(1, 1, 1)
from matplotlib.colors import ColorConverter

'''
for (i, m) in df_pc_member.iterrows():
    #print(m['PC1'])
    ax.scatter(m['PC1'], m['PC2'], color=m['_color'])
    #print(m)
'''

for (gname, group) in df_pc_member.groupby(['agrupacion', 'agrupacion_color']):
    print(gname)
    if gname == ('', ''):
        gname = ('null', '#000000')
    ax.scatter(group['PC1'], [0] * len(group['PC1']), color=gname[1], label=gname[0])

plt.legend(loc='upper left')
plt.xlabel('PC1')
plt.ylabel('PC2')

#df_pc[['PC1', 'PC2']].plot(x='PC1', y='PC2', kind='scatter')

df_pc_member['legislador_id'] = df_pc_member['legislador_id'].apply(lambda x: str(x))
open('data/pc-member-3d.json', 'w').write(json.dumps(
    df_pc_member.set_index('legislador_id').T.to_dict()
    ))
open('data/pc-member-3d.js', 'w').write(
    'var jsonData = %s' %
    json.dumps(
        df_pc_member.set_index('legislador_id').T.to_dict()
    ))

df_pc_member

get_ipython().magic('matplotlib inline')

x = np.array(X_reduced.T[0, :]).astype('float')
y = np.array(X_reduced.T[1, :]).astype('float')
z = np.array(X_reduced.T[2, :]).astype('float')

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for (gname, group) in df_pc_member.groupby(['agrupacion', 'agrupacion_color']):
    print(gname)
    if gname == ('', ''):
        gname = ('null', '#000000')
    ax.scatter(group['PC1'], group['PC2'], group['PC3'], color=gname[1], label=gname[0])
#ax.scatter(x, y, z, picker=True, s=100)

def onpick(X_3D, event):
    #print(event)
    if hasattr(onpick, '_label'):
        #pass
        onpick._label.remove()
    thisline = event.artist
    ind = event.ind
    #print(type(ind))
    #print(X_3D[0, ind])
    names = df_matrix.iloc[ind].index.values
    #print(names)
    label = ('\n'.join(names))
    pos = (X_3D[0, ind[0]], X_3D[1, ind[0]], X_3D[2, ind[0]])
    #onpick._label = ax.set_title(label
    onpick._label = ax.text(*pos, s=label)
    fig.canvas.draw()
    
fig.canvas.mpl_connect('pick_event', lambda e: onpick(X_reduced.T, e))

