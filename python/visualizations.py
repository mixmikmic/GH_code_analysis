import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')

pkl_file = open('../data/df.pkl', 'rb')
df = pickle.load(pkl_file)
pkl_file.close() 

df.columns

colors = ['k','r','chartreuse','seagreen','paleturquoise','deepskyblue','blue','darkorchid','plum','mediumvioletred','gold','orange']

c = dict(zip(list(df['resort'].unique()),colors))
c

plt.figure(figsize=(12,8))
plt.scatter(df['max_grade_(%)'][df['max_grade_(%)'] < 200],df['ability_nums'][df['max_grade_(%)'] < 200],
            c=[c[x] for x in df['resort']],alpha=.2)
plt.xlabel('Max Grade %')
plt.ylabel('Ability Level')
plt.title('Ability Level vs Max Grade by Resort')
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c.values()]
plt.legend(markers, c.keys(), numpoints=1);

plt.figure(figsize=(12,8))
plt.scatter(df['slope_length_(ft)'][df['slope_length_(ft)'] < 10000],df['ability_nums'][df['slope_length_(ft)'] < 10000],
            c=[c[x] for x in df['resort']],alpha=.2)
plt.xlabel('Slope Length (ft)')
plt.ylabel('Ability Level')
plt.title('Ability Level vs Slope Length (ft) by Resort')
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c.values()]
plt.legend(markers, c.keys(), numpoints=1);

plt.figure(figsize=(12,8))
plt.scatter(df['color_nums'],df['ability_nums'],
            c=[c[x] for x in df['resort']],alpha=.2)
plt.xlabel('Difficulty (color)')
plt.ylabel('Ability Level')
plt.title('Ability Level vs Difficulty (color) by Resort')
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c.values()]
plt.legend(markers, c.keys(), numpoints=1);

plt.figure(figsize=(12,8))
plt.scatter(df['slope_area_(acres)'][df['slope_area_(acres)'] < 100],df['ability_nums'][df['slope_area_(acres)'] < 100],
            c=[c[x] for x in df['resort']],alpha=.2)
plt.xlabel('Slope Area (Acres)')
plt.ylabel('Ability Level')
plt.title('Ability Level vs Slope Area (Acres) by Resort')
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c.values()]
plt.legend(markers, c.keys(), numpoints=1);

plt.figure(figsize=(12,8))
plt.scatter(df['slope_area_(acres)'][(df['slope_area_(acres)'] < 100)&(df['max_grade_(%)'] < 200)],
            df['max_grade_(%)'][(df['slope_area_(acres)'] < 100)&(df['max_grade_(%)'] < 200)],
            c=[c[x] for x in df['resort']],alpha=.2)
plt.xlabel('Slope Area (Acres)')
plt.ylabel('Max Grade %')
plt.title('Max Grade vs Slope Area by Resort')
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c.values()]
plt.legend(markers, c.keys(), numpoints=1);

df.groupby(['resort','colors']).size().unstack().plot(kind='bar',figsize = (16,8), color = ['m','k','b','g'], 
                                                      title = 'Number of runs of each color by resort');

df.groupby(['resort','colors']).size().unstack().fillna(0);

df_pct = df.groupby(['resort','colors']).size().unstack().fillna(0).copy()
df_pct['total'] = df_pct.sum(axis=1)
for col in df_pct.columns[:-1]:
    df_pct[col] = (df_pct[col]/df_pct['total'])*100
df_pct = df_pct.drop('total',axis=1)

df_pct.plot(kind='bar',figsize=(16,8),color=['m','k','b','g'], width = .7, fontsize=20);
plt.legend(loc=2,bbox_to_anchor=(.085,.99),fontsize=16);
plt.xticks(rotation=60);
plt.xlabel('Resort', fontsize=34);
plt.ylabel('Percent', fontsize=34);
plt.title('Percent of Runs by Color by Resort',fontsize=40);

df.groupby(['colors','resort']).size().unstack().plot(kind='bar',figsize = (16,8), color = colors, 
                                                      title='Number of runs per resort of each color');

df_pct2 = df.groupby(['colors','resort']).size().unstack().fillna(0).copy()
df_pct2['total'] = df_pct2.sum(axis=1)
for col in df_pct2.columns[:-1]:
    df_pct2[col] = (df_pct2[col]/df_pct2['total'])*100
df_pct2 = df_pct2.drop('total',axis=1)

df_pct2.plot(kind='bar',figsize=(16,8),color=colors, title='Percent of resorts for each color');

df[['trail_name','resort','ability_level','colors']][abs(df['ability_nums']/6 - df['color_nums']/4 > .5)]

plt.figure(figsize=(12,8))
plt.scatter(df['avg_width_(ft)'],df['vert_rise_(ft)'],c=[c[x] for x in df['resort']],alpha=.2)
plt.xlabel('Average Width (ft)')
plt.ylabel('Vertical Rise (ft)')
plt.title('Vertical Rise vs Average Width by Resort')
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in c.values()]
plt.legend(markers, c.keys(), numpoints=1);

plt.figure(figsize=(16,8))
plt.hist(df['avg_width_(ft)'], bins=1125)
plt.xlabel('Average Width (ft)');

plt.figure(figsize=(16,8))
plt.hist(df['vert_rise_(ft)'], bins=1125)
plt.xlabel('Vertical Rise (ft)');

df[df['vert_rise_(ft)'] > 3000]



