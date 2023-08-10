# Import libraries
import numpy as np
import pandas as pd
import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import json


#load in dataset
df = pd.read_csv("../data/movie_ratings_simple.csv")
len(df)

sns.set(font_scale=2)
sns.set_style("white")

#what do the years look like?
df['year'].hist(bins=max(df['year'])-min(df['year'])+1)
plt.xlabel("Year")
plt.ylabel("Number of films")
plt.savefig("../docs/filmsperyear.png")

df['stars'].hist(bins=9)
plt.xlabel("Average Rating")
plt.ylabel("Number of films")
plt.tight_layout()
plt.savefig("../docs/ratinghist.png")

filmsperyear=df[['year','stars']].groupby('year').count()
filmsperyear.columns=['NFilms']

starsperyear=df[['year','stars']].groupby('year').mean()
starsperyear.plot()
plt.gca().legend_.remove()
plt.ylim((1,10))
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.savefig("../docs/ratingbyyear.png")
#starsperyear

genres=pd.concat([df['Genre1'],df['Genre2'],df['Genre3']])
ungen = genres.unique()
print(len(genres.unique()))
genres.value_counts()[1:].plot(kind='bar')
plt.xticks(rotation='75')

dfG1 = df[['year','stars','Genre1']]
dfG1.columns = ['year','stars','Genre1']
dfG2 = df[['year','stars','Genre2']]
dfG2.columns = ['year','stars','Genre']
dfG3 = df[['year','stars','Genre3']]
dfG3.columns = ['year','stars','Genre']
dfG = dfG1.append(dfG2).append(dfG3)

dfG['Genre'] = dfG['Genre'].astype('category')
dfG['Genre'].describe()

def normalizeStars(row):
    return row['stars']/starsperyear.loc[row['year']]
genresbyyearrev=dfG.groupby(['year','Genre']).mean()
genredatabyyear = genresbyyearrev.reset_index()
genredatabyyear
genredatabyyear['norm']=genredatabyyear.apply(lambda x: normalizeStars(x),axis=1)
#genredatabyyear

sns.set(font_scale=1.3)
sns.set_style("white")

genredatabyyear = genredatabyyear[(genredatabyyear['Genre'].isin(['Action','Animation','Documentary','War']))]
genredatabyyear['Genre']=genredatabyyear['Genre'].cat.remove_unused_categories()
g = sns.FacetGrid(genredatabyyear, col='Genre',col_wrap=2)
g = g.map(plt.plot, 'year','norm')
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.set_xlabels('Year')
g.set_ylabels('Normalized Rating')
plt.tight_layout()
plt.savefig("../docs/ratingbygenre.png")

def normalizeYears(row):
    return row['stars']/filmsperyear.loc[row['year']]

genresbyyear2=dfG.groupby(['year','Genre']).count()
genredata2 = genresbyyear2.reset_index()
genredata2 = genredata2[(genredata2['Genre'].isin(['Action','Animation','Documentary','War']))]
genredata2['Genre']=genredata2['Genre'].cat.remove_unused_categories()
genredata2['norm']=genredata2.apply(lambda x: normalizeYears(x),axis=1)
g = sns.FacetGrid(genredata2, col='Genre',col_wrap=2)
g = g.map(plt.semilogy, 'year','norm')
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.set_xlabels('Year')
g.set_ylabels('Fraction of total films')
plt.tight_layout()
plt.savefig("../docs/nfilmsbygenre.png")

sns.set(font_scale=1.5)
sns.set_style("white")
directors=df['Director1']
print(len(directors.unique()))
vc=directors.value_counts()
subdir = vc[vc>1]
subdir[:30].plot(kind='bar')
plt.ylabel("Number of Films")
plt.tight_layout()
plt.savefig("../docs/director_nfilms.png")

sns.set(font_scale=2)
sns.set_style("white")
directorrating=df[['Director1','stars']].groupby('Director1').mean()
nfilms = subdir.reset_index()
nfilms.columns = ['Director1','nfilms']
directorByRating=(directorrating.reset_index()).merge(nfilms,on='Director1')
directorByRating.plot(kind='scatter',x='nfilms',y='stars',xlim=(0,50))
plt.xlabel('Number of films')
plt.ylabel('Average Director Rating')
plt.savefig("../docs/director_ratings_by_film.png")

sns.set(font_scale=1.5)
sns.set_style("white")
directorByRating[directorByRating['nfilms']>20].sort_values('stars',ascending=False).plot(kind='bar',x='Director1',y='stars')
plt.gca().legend_.remove()
plt.ylim(1,10)
plt.xlabel('')
plt.ylabel('Average Rating')
plt.tight_layout()
plt.savefig("../docs/bestdirector.png")

model = pd.read_csv("../data/multiclass-model_0919.csv",index_col=0)
rowtotals = model['Total']
model.drop(['Total','F1'],axis=1,inplace=True)
nevals=(model.sum().sum())
model = model.div(rowtotals,axis='rows')

def removeDiag(row):
    row.loc[str(row.name)]= 0
    return row

model_bad = model.copy().apply(lambda x: removeDiag(x) ,axis=1)

model_good = model.subtract(model_bad,axis='rows')


sns.set_style('white')
plt.imshow(model_bad, interpolation='nearest')
plt.hold(True)
plt.imshow(model_good, interpolation='nearest', alpha=.6,cmap=plt.cm.Blues)
plt.xticks(np.arange(0,10),model.columns.values)
plt.yticks(np.arange(0,10),model.index.values)
plt.gca().xaxis.tick_top()
plt.xlabel('Predicted Values')
plt.gca().xaxis.set_label_position('top') 
plt.ylabel('True Values')
plt.tight_layout()
plt.savefig("../docs/conf_matrix.png")

