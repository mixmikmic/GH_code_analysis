import pandas as pd

df = pd.read_json('http://api.census.gov/data/2010/sf1?get=P0120001&for=county:*')
df.columns = df.iloc[0]
df.drop(df.index[0], inplace=True)
df.head()

pd.read_json('http://api.census.gov/data/2010/sf1/variables/P0120001.json', typ='ser')

fields = ['P01200%02i'%i for i in range(3,26)]
url = 'http://api.census.gov/data/2010/sf1?get=%s&for=county:*'%','.join(fields)
print url
pops2010 = pd.read_json(url)
pops2010.columns = pops2010.iloc[0]
pops2010.drop(pops2010.index[0], inplace=True)
pops2010 = pops2010.applymap(float)
pops2010.set_index(['state', 'county'], inplace=True)
pops2010.head()

fields = ['PCT012%03i'%i for i in range(3,105)]

dflist = []
chunkSize = 40
for i in range(0, len(fields), chunkSize):
    chunk = fields[i:i+chunkSize]
    url = 'http://api.census.gov/data/2000/sf1?get=%s&for=county:*'%','.join(chunk)
    print url
    df_chunk = pd.read_json(url)
    df_chunk.columns = df_chunk.iloc[0]
    df_chunk.drop(df_chunk.index[0], inplace=True)
    df_chunk = df_chunk.applymap(float)
    df_chunk.set_index(['state', 'county'], inplace=True)
    dflist.append(df_chunk)

pops2000 = pd.concat(dflist,axis=1)
pops2000 = pops2000.applymap(float)
pops2000.head()

pops2010d = pd.DataFrame(index=pops2010.index)

decades = ['dec_%i'%i for i in range(1,10)]
breakpoints_2010 = [3, 5, 8, 12, 14, 16, 18, 22, 24, 26]
for dec, s, f in zip(decades, breakpoints_2010[:-1], breakpoints_2010[1:]):
    pops2010d[dec] = pops2010[['P0120%03i'%i for i in range(s,f)]].sum(axis=1)
    
pops2010d.head()

pops2000d = pd.DataFrame(index=pops2000.index)

decades = ['dec_%i'%i for i in range(1,10)]
breakpoints_2000 = [3, 13, 23, 33, 43, 53, 63, 73, 83, 104]
for dec, s, f in zip(decades, breakpoints_2000[:-1], breakpoints_2000[1:]):
    pops2000d[dec] = pops2000[['PCT012%03i'%i for i in range(s,f)]].sum(axis=1)

pops2000d.head()

frame = pd.concat([pops2000d, pops2010d], keys=[2000, 2010], axis=1)
frame.dropna(inplace=True)
frame.head()

frame.to_csv('Males by decade and county.csv')

pd.read_csv('Males by decade and county.csv', header=[0,1], index_col=[0,1])



