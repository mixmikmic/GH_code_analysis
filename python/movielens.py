import pandas as pd

PATH = "../data/u.item"
df = pd.read_csv(PATH, names=('movie_id', 'title','release_date'), sep='|', usecols=[0,1,2], encoding = "ISO-8859-1", index_col='movie_id')
df.head(5)

df['title'] = df['title'].str.replace(r"\(.*\)","")
df.head(5)

df['release_date'] = pd.to_datetime(df['release_date'])
df.sort_values('release_date').head(1)

df.sort_values('release_date', ascending=False).head(1)

# All titles
len(df['title'])

# Unique titles 
len(df['title'].unique())

titles = df["title"]
repeated = df[titles.isin(titles[titles.duplicated()])].sort_values("title")
repeated

dupes = repeated.groupby(['title', 'release_date']).size().reset_index()
dupes.columns=['title','release_date','dupe_count']
errors = dupes.title[dupes.dupe_count>1]
errors

remakes = dupes.title[dupes.dupe_count<2].drop_duplicates()
remakes

df['year'] = pd.DatetimeIndex(df['release_date']).year.astype(int)
df['month'] = pd.DatetimeIndex(df['release_date']).month.astype(int)
df['day'] = pd.DatetimeIndex(df['release_date']).day.astype(int)
len(df[df.month==12])

df['title'] = df['title'].str.lower()
animals = ['dog', 'cat', 'fish', 'monkey', 'elephant', 'tiger', 'lion', 'hamster', 'pig', 'insect', 'bird']
pattern = '|'.join(animals)
df[df.title.str.contains(pattern)]

df['words'] = df['title'].str.split()

for index, row in df.iterrows():
    df['words'][index] = len(df['words'][index])

df[df['words']>10]



