from houseofwine_gr import get

get('http://www.houseofwine.gr/how/megas-oinos-skoura.html')

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

import matplotlib as mpl
import seaborn as sns

plt.style.use('fivethirtyeight')

mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'Serif'
mpl.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']='white'
plt.rcParams['axes.grid'] =False
plt.rcParams['figure.facecolor']='white'

import pandas as pd

df = pd.read_json('./data/houseofwine.gr-wines.json', encoding='utf-8')

df.head()

from numpy import nan
df = df.replace('', nan, regex=True)

df = df.rename(columns={'alcohol_%': 'alcohol', 'avg_rating_%': 'avg_rating'}, inplace=False)

df['alcohol'] = df.alcohol.astype(float)
df['n_votes'] = df.n_votes.astype(int, errors='ignore')
df['price'] = df.price.astype(float)
df['year'] = df.year.astype(int, errors='ignore')

df['color'] = df.color.replace({'Λευκός': 'White',
                                'Ερυθρός': 'Red', 
                                'Ροζέ': 'Rosé'
                               }
                              )

ax = df['color'].value_counts().plot('bar')
ax.set(ylabel='Wines', title='Wine Color Frequency');

fig, ((ax1, ax2), 
      (ax3, ax4)
     ) = plt.subplots(ncols=2, nrows=2, figsize=(16,12))

df.year.dropna().astype(int).value_counts().plot('bar', ax=ax1)
ax1.set(title='Production Year Frequency', xlabel='Year');

sns.distplot(df[df.alcohol < 100].alcohol.dropna(), ax=ax2)
ax2.set(xlabel='Alcohol %', title='Alcohol % Distribution');

sns.distplot(df[df.price < 100].price.dropna(), ax=ax3)
ax3.set(xlabel='Price (< 100)')

sns.distplot(df.avg_rating.dropna(), ax=ax4)
ax4.set(xlabel='Average Rating');

df.tags.head(10)

df['tags'] = df.tags.map(set)

sweetness_values = {'Γλυκός', 'Ημίγλυκος', 'Ξηρός', 'Ημίξηρος'}
df['sweetness'] = df.tags.map(sweetness_values.intersection).map(lambda x: x.pop() if x else None)

translations = {'Γλυκός': 'Sweet', 'Ημίγλυκος': 'Semi-Sweet', 'Ξηρός': 'Dry', 'Ημίξηρος': 'Semi-Dry'}
df['sweetness'] = df['sweetness'].replace(translations)

df['sparkling'] = df.tags.map({'Αφρώδης', 'Ημιαφρώδης'}.intersection
                             ).map(lambda x: x.pop() if x else None
                                  ).replace({'Αφρώδης': 'Sparkling', 'Ημιαφρώδης': 'Semi-Sparkling'})
df['sparkling'] = df.sparkling.fillna('Not Sparkling')

df['is_mild'] = df.tags.map(lambda x: 'Ήπιος' in x)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, squeeze=False)

for attr, ax in zip(['sweetness', 'color', 'sparkling', 'is_mild'], (ax1, ax2, ax3, ax4)):
    df[attr].value_counts().sort_values(ascending=True).plot(kind='barh', ax=ax)
    attr_str = 'Mildness' if attr is 'is_mild' else attr.title()
    ax.set(xlabel='Number of Wines', title='{} Frequency'.format(attr_str));
    
fig.set_size_inches((12,8))
fig.tight_layout()

non_varietal_tags = {'Αφρώδης', 'Ημιαφρώδης', 'Ξηρός', 'Ημίξηρος', 'Γλυκός', 'Ημίγλυκος', 'Ήπιος'}
df['varieties'] = df.tags.map(lambda t: t.difference(non_varietal_tags))

def is_not_int(x):
    try:
        int(x)
        return False
    except ValueError:
        return True
    
df['varieties'] = df.varieties.map(lambda x: set(filter(is_not_int, x)))

df['is_varietal'] = df.varieties.map(set.__len__) == 1

df.loc[df.is_varietal, 'single_variety'] = df.loc[df.is_varietal, 'varieties'].map(lambda v: next(iter(v)))

df['is_blend'] = df.varieties.map(set.__len__) >= 2

df.loc[df.is_varietal, 'variety_type'] = 'Varietal'
df.loc[df.is_blend, 'variety_type'] = 'Blend'

ax = df.is_varietal.replace({True: 'Varietal', 
                        False: 'Blend'}
                      ).value_counts().plot('barh')
ax.set(title='How many wines are varietals and how many blends ?');

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10,14))
fig.tight_layout

varieties_hist = df[df.is_varietal].varieties.map(lambda x: next(iter(x))).value_counts()
varieties_hist.head(10).sort_values(ascending=True).plot('barh', ax=ax1)
ax1.set(title='Most Frequent Varietal Wines (Top 10)');

varietals_mean_price = df[['single_variety', 'price']].groupby('single_variety').mean().dropna()['price']
varietals_mean_price.sort_values(ascending=False).head(10).sort_values(ascending=True).plot('barh', ax=ax2)
ax2.set(title='Most Expensive Varietals', xlabel='', ylabel='');

varietals_mean_rating = df[['single_variety', 'avg_rating']].groupby('single_variety').mean().dropna()['avg_rating']
varietals_mean_rating.sort_values(ascending=False).head(10).sort_values(ascending=True).plot('barh', ax=ax3);
ax3.set(title='Top Rated Varietals', ylabel='');

fig.tight_layout()

def create_coocurrence_df(docs):
    # Source: https://stackoverflow.com/a/37822989
    count_model = CountVectorizer(lowercase=False, min_df=.1) # default unigram model
    X = count_model.fit_transform(docs)
    Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
    Xc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
    ret = pd.DataFrame(Xc.todense())#, index=count_model.get_feature_names(), columnscou=count_model.get_feature_names())
    ret.index = ret.columns = list(map(lambda f: f.replace('_', ' '), count_model.get_feature_names()))
    return ret

from sklearn.feature_extraction.text import CountVectorizer
docs = df.loc[df.is_blend, 'varieties'].map(lambda x: {s.replace(' ', '_') for s in x}).map(lambda x: ' '.join(x))

coocurrence = create_coocurrence_df(docs)

df.loc[df.is_blend, 'varieties'].head(10)

coocurrence

ax = coocurrence.sum().sort_values(ascending=False).head(10).sort_values(ascending=True).plot(kind='barh')
ax.set(title='Most Frequent Varieties Appearing in Blends (Top 10)');

ax = sns.heatmap(coocurrence, square=True, annot=True, fmt="d", cmap='Blues')
ax.set(title='Which varieties are usually blended together ?\n'.title());
plt.gcf().set_size_inches((8,6))

fig, axes = plt.subplots(nrows=3, figsize=(12, 18))
for c, ax in zip(['Red', 'White', 'Rosé'], axes):
    docs = df.loc[df.is_blend & (df.color==c), 'varieties'].map(lambda x: {s.replace(' ', '_') for s in x}).map(lambda x: ' '.join(x))
    cooc = create_coocurrence_df(docs)
    cmaps = {'Red': 'Reds', 'White': 'Blues', 'Rosé': sns.cubehelix_palette(8)}
    sns.heatmap(cooc, square=True, annot=True, fmt="d", cmap=cmaps.get(c), ax=ax)
    ax.set(title='Usually blended varieties for {}\n'.format(c).title());
plt.tight_layout()

