import numpy as np
import os
import pandas as pd

dataroot = 'C:/Users/lezhi/Dropbox/thesis/data/'
tgtroot = 'C:/Users/lezhi/Dropbox/thesis/img/'

cities = ['boston', 'chicago', 'newyork', 'sanfrancisco']
#cities = ['boston']
features = ['sky_tree', 'entropy_mean', 'color4']

for c in cities:
    df = pd.read_csv(dataroot+features[0]+'_'+c+'.csv', index_col=0)
    df[['lat','lng']] = df[['lat','lng']].astype(str)
    df = reformlatlng(df)
    for f in features[1:]:
        df1 = pd.read_csv(dataroot + f + '_' + c + '.csv', index_col=0)
#         if not f == 'neighborhood':
        df1[['lat','lng']] = df1[['lat','lng']].astype(str)
        df1 = reformlatlng(df1)
        df = pd.merge(df, df1, how='outer', on=['lat','lng'])
        
    
#     df['lat'] = [re.sub('0$','1',ele[:9]) for ele in df['lat']]
#     if c == 'sanfrancisco':
#         df['lng'] = [re.sub('0$','1',ele[:11]) for ele in df['lng']]
#     else:
#         df['lng'] = [re.sub('0$','1',ele[:10]) for ele in df['lng']]
        
    df.loc[pd.isnull(df['color']), 'entropy_mean'] = np.nan
    df = df.rename(columns={'entropy_mean': 'entropy'})
    df['neighborhood'] = pd.read_csv(dataroot + 'neighborhood_' + c + '.csv')['NAME']
    df['city'] = c
        
    df.to_csv(dataroot + c + '.csv')

def reformlatlng(df):
    df['lat'] = [re.sub('0$','1',ele[:9]) for ele in df['lat']]
    if c == 'sanfrancisco':
        df['lng'] = [re.sub('0$','1',ele[:11]) for ele in df['lng']]
    else:
        df['lng'] = [re.sub('0$','1',ele[:10]) for ele in df['lng']]
    return df



np.sum(pd.isnull(df['color']))

def iter_dir(rootdir, dostuff):
    citynames = np.array(sorted([d for d in os.listdir(rootdir) if os.path.isdir(rootdir)]))
    for cityname in citynames[np.array([9])]:   ######################
        citypath = rootdir + cityname 
        print citypath
        imgnames = sorted([f[:-4] for f in os.listdir(citypath) if os.path.isfile(os.path.join(citypath, f))])
        
        lat_lng_dir = np.array([name.replace('_',',').split(',') for name in imgnames])
        newnames = [(re.sub('0$','1', ele[0][:9]) + "," + re.sub('0$','1', ele[1][:11]) + "_" + ele[2]) for ele in lat_lng_dir]
        
        old_new = zip(imgnames, newnames)
        
        for item in old_new:
            dostuff(item, citypath)
            
# to prove that no repetivive image names exist            
#         print len(newnames), len(np.unique(np.array(newnames)))

def renameFile(tup, _dir):
    #print _dir + "/" + tup[0] + ".png"
    os.rename(_dir + "/" + tup[0] + ".png", _dir + "/" + tup[1] + ".png")

iter_dir(tgtroot, renameFile)

iter_dir(tgtroot, renameFile)

'{0:g}'.format(float('3.14134000'))

import re
line = re.sub('0$', '1', '-53.141584613434000')

line

df0 = pd.read_csv(dataroot+'labels_dense_'+ cities[0] +'_jpg.csv', index_col=0)
for c in cities[1:]:
    df = pd.read_csv(dataroot+'labels_dense_'+c+'_jpg.csv', index_col=0)
    df0 = pd.concat([df0,df])

df0 = pd.read_csv(dataroot+'labels_dense_all.csv', index_col=0)

df0['label'] = df0.apply(lambda x: x['city']+'_'+x['label'], axis=1)
#del df_all['label2']

df0.to_csv(dataroot+'labels_dense_all.csv')



