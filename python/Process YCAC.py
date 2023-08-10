import pandas as pd
import json

import functools

fns = ['/home/eamonn/Projects/corpuscule/ycac/ycac_all.corpus']

df = pd.DataFrame()

for fn in fns:
    df = df.append(pd.read_csv(fn,encoding='latin1'))

len(df)

@functools.lru_cache(maxsize=None)
def str_to_vec(s):
    l = eval(s)
    rep = ""
    for i in range(12):
        if i in l:
            rep += "1"
        else:
            rep += "0"
    return rep

df.head(5)

get_ipython().run_cell_magic('time', '', "df['canonical'] = df.PCsInNormalForm.apply(str_to_vec)")

df.head(5)

u = df.canonical.unique()
len(u)

df.head(10).file[1]

meta = pd.read_csv(open('/home/eamonn/Projects/corpuscule/ycac/csv/metadata/YCAC-metadata.csv', 'r'))

meta.Date.fillna(meta.Range, inplace=True)

import numpy as np

def process_range(d):
    try:
        if '-' in d:
            a = int(d.split('-')[0])
            b = int(d.split('-')[1])
            return int((a + b) / 2)
        else:
            return int(d)
    except:
        return np.nan

meta['EstDate'] = meta.Date.apply(process_range)

df['file'] = df.file.apply(lambda s: s.replace('.mid', ''))
df['Filename'] = df['file']

master = pd.merge(df, meta, on='Filename')

master.head(5)

top_composers = list(master.Composer_x.value_counts().keys()[:50])

for composer in top_composers:
    sub = master[master.Composer_x == composer]
    sub = sub[['file', 'canonical']]

    files = list(sub.file)
    vecs = list(sub.canonical)

    fv = pd.DataFrame(list(zip(files,vecs)),columns = ['file', 'canonical'])
    fv.reset_index()
    
    # done = fv.groupby('file')['canonical'].apply(list)
    docs = ['{} '.format(l) for l in list(fv.canonical)]
    
    del fv

    print("{} group done".format(composer))

    with open('{}.corpus'.format(composer), 'w') as f:
        f.writelines(docs)

    print("{} dump done".format(composer))

start_years = [x for x in range(1650, 1950, 10)]

eras = [(x, x+9) for x in start_years]

eras = [(1650,1850)]

for era in eras:
    start_year = era[0]
    end_year = era[1]
    
    sub = master[master.EstDate.isin(range(start_year, end_year))]
    sub = sub[['file', 'canonical']]

    label = "{}-{}".format(start_year, end_year)
    
    files = list(sub.file)
    vecs = list(sub.canonical)

    fv = pd.DataFrame(list(zip(files,vecs)),columns = ['file', 'canonical'])
    fv.reset_index()
    
    #done = fv.groupby('file')['canonical'].apply(list)
    docs = ['{} '.format(l) for l in list(fv.canonical)]

    print("{} group done".format(label))

    with open('{}.corpus'.format(label), 'w') as f:
        f.writelines(docs)
    
    print("{} dump done".format(label))



