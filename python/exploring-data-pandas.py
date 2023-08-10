get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sb

print(plt.style.available)
plt.style.use('seaborn')

plt.rcParams['figure.figsize'] = 8, 4

from pymongo import MongoClient

def get_mongo_database(db_name, host='localhost', port=27017, username=None, password=None):
    '''Get (or create) named database from MongoDB with/out authentication'''
    if username and password:
        mongo_uri = 'mongodb://{}:{}@{}/{}'.format(username, password, host, db_name)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)
    return conn[db_name]

def mongo_to_dataframe(db_name, collection_name, query={}, host='localhost',
                       port=27017, username=None, password=None, no_id=True):
    '''Create a Pandas DataFrame from MongoDB collection'''
    db = get_mongo_database(db_name, host, port, username, password)
    cursor = db[collection_name].find(query)
    df = pd.DataFrame(list(cursor))
    if no_id:
        del df['_id']
    return df

def dataframe_to_mongo(dframe, db_name, collection_name, host='localhost',
                 port=27017, username=None, password=None):
    '''save a dataframe to mongodb collection'''
    db = get_mongo_database(db_name, host, port, username, password)
    records = dframe.to_dict('records')  # 'records' puts it into our list-of-dicts format
    db[collection_name].insert_many(records)

def delete_collection(db_name, collection_name, host='localhost',
                 port=27017, username=None, password=None):
    '''save a dataframe to mongodb collection'''
    db = get_mongo_database(db_name, host, port, username, password)
    db[collection_name].delete_many({}) # empty filter deletes all entries


DB_NOBEL_PRIZE = 'nobel_prize' # use string constants or a spell error in retrieval will create new table.
COLL_WINNERS = 'winners' # winners collection


#----------------------------
# From json file.
#----------------------------
if True:
    with open('data/nwinners_all.json') as f:
        df = pd.read_json(f)

# Saving data
# Make sure mongodb is clear (so we don't duplicate data), then
# save to Mongo for next section
if False:
    db = get_mongo_database(DB_NOBEL_PRIZE)
    db[COLL_WINNERS].delete_many({})  # deletes everything (no filter)
    dataframe_to_mongo(df, DB_NOBEL_PRIZE, COLL_WINNERS) # save to Mongo for next section

#----------------------------
# From mongodb collection
#----------------------------
if False:  
    DB_NOBEL_PRIZE = 'nobel_prize' # use string constants or a spell error in retrieval will create new table.
    COLL_WINNERS = 'winners_all' # winners collection
    df = mongo_to_dataframe(DB_NOBEL_PRIZE, COLL_WINNERS)

df.info()

pd.to_datetime(df.date_of_birth) #DAY TODO - this is an issue -- all dates the same, 
                                 #time different when using the json input 
                                 #(use mongodb instead for dates)

# convert the date columns to a usable form
df.date_of_birth = pd.to_datetime(df.date_of_birth)
df.date_of_death = pd.to_datetime(df.date_of_death)

df.info()

by_gender = df.groupby('gender')
print(by_gender.size())
print(df.loc[(df.year<=2015),:].groupby('gender').size()) #TODO different than book even though same years

by_gender.size().plot(kind='bar')

by_cat_gen = df.groupby(['category', 'gender'])
by_cat_gen.get_group(('Physics', 'female'))[['name', 'year']] #get a group by a category and gender key

by_cat_gen.size()

by_cat_gen.size().plot(kind='barh')

print(by_cat_gen.size().unstack().head())
by_cat_gen.size().unstack().plot(kind='barh')

cat_gen_sz = by_cat_gen.size().unstack()
cat_gen_sz['total'] = cat_gen_sz.sum(axis=1)
cat_gen_sz = cat_gen_sz.sort_values(by='female', ascending=True)
cat_gen_sz[['female', 'total', 'male']].plot(kind='barh')

df[(df.category == 'Physics') & (df.gender == 'female')][['name', 'country','year']]

# Reducing the number of x axis ticks
def thin_xticks(ax, tick_gap=10, rotation=45):
    """ Thin x-ticks and adjust rotation """
    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    #print('debug: thin_xticks: ticklabels = {}'.format(ticklabels)) #debug
    ax.xaxis.set_ticks(ticks[::tick_gap])
    ax.xaxis.set_ticklabels(ticklabels[::tick_gap], rotation=rotation)
    ax.figure.show()

by_year_gender = df.groupby(['year','gender'])
year_gen_sz = by_year_gender.size().unstack()
ax = year_gen_sz.plot(kind='bar', figsize=(16,4))
thin_xticks(ax)

# the current unstacked group sizes use an automatic year index
by_year_gender = df.groupby(['year', 'gender'])
by_year_gender.size().unstack()

# However, there are some gap years where no prize was given and those
# gaps aren't in the dataframe right now.  Create a new index to make sure
# the gap years are represented:
new_index = pd.Index(np.arange(1901, 2016), name='year')
by_year_gender = df.groupby(['year','gender'])
year_gen_sz = by_year_gender.size().unstack().reindex(new_index)

year_gen_sz.loc[1935:1945, :] # now it has the gap years 1940, 41, 42

# Another problem with plot above is there are too many bars.
# have dedicated male and female plots but stacked so as to allow
# easy comparisons:
plt.style.use('seaborn')

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
ax_f = axes[0]
ax_m = axes[1]
fig.suptitle('Nobel Prize-winners by gender', fontsize=16)
ax_f.bar(year_gen_sz.index, year_gen_sz.female)
ax_f.set_ylabel('Female winners', fontsize=14)
ax_m.bar(year_gen_sz.index, year_gen_sz.male)
ax_m.set_ylabel('Male winners', fontsize=14)
ax_m.set_xlabel('Year')

# Let's look at national trends.  Start with simple histogram
df.groupby('country').size().sort_values(ascending=False).plot(kind='bar', figsize=(12,4))

nat_group = df.groupby('country')
ngsz = nat_group.size()
ngsz.index

# Read in our countries data
if True:
    df_countries = pd.read_json('data/country_data.json', orient='record')
else:
    df_countries = mongo_to_dataframe('nobel_prize', 'countries')

df_countries.iloc[0] # show the first row by position

# set the index to 'name'.  Then you can add the 'ngsz' Series
# that we created above which also has country name as index and
# it will match them up appropriately
df_countries = df_countries.set_index('name')
df_countries['nobel_wins'] = ngsz
df_countries['nobel_wins_per_capita'] =df_countries.nobel_wins / df_countries.population

df_countries.sort_values(by='nobel_wins_per_capita', ascending=False).nobel_wins_per_capita.plot(kind='bar', figsize=(12,4))

# filter for countries with more than two Nobel prizes
df_countries[df_countries.nobel_wins > 2].sort_values(by='nobel_wins_per_capita', ascending=False).nobel_wins_per_capita.plot(kind='bar')

# Now lets look at  wins by category
nat_cat_sz = df.loc[:2015,:].groupby(['country', 'category']).size().unstack()
nat_cat_sz[:5] # take the first five countries by category

# DAY TODO why doesn't this table match the book?  
# book shows Argentina having 2 Chemistry prizes, for example.

COL_NUM = 2
ROW_NUM = 3
fig, axes = plt.subplots(ROW_NUM, COL_NUM, figsize=(12,12))
for i, (label, col) in enumerate(nat_cat_sz.iteritems()):
    ax = axes[i//COL_NUM, i%COL_NUM]
    col = col.sort_values(ascending=False)[:10]
    col.plot(kind='barh', ax=ax)
    ax.set_title(label)
plt.tight_layout()

# increase the font size 
plt.rcParams['font.size'] = 20

new_index = pd.Index(np.arange(1901, 2016), name='year') #fills in the gap years
by_year_nat_sz = df.groupby(['year', 'country']).size().unstack().reindex(new_index)

by_year_nat_sz['United States'].cumsum().plot()

# for years where the US won no prize, cumsum() returns NaN
# let's replace those NaN with zeros
by_year_nat_sz['United States'].fillna(0).cumsum().plot()

# Compare the US to the rest of the world
by_year_nat_sz = df.groupby(['year', 'country']).size().unstack().fillna(0)
not_US = by_year_nat_sz.columns.tolist() # get list of country column names
not_US.remove('United States') #remove "United States" from list of names
print(not_US)
# now use not_US column name list to create a 'Not US' column with
# sum of all prizes for countries not in the not_US list.
by_year_nat_sz['Not US'] = by_year_nat_sz.loc[:,not_US].sum(axis=1)
ax = by_year_nat_sz.loc[:,['United States', 'Not US']].cumsum().plot()

# Look at regional differences
new_index = pd.Index(np.arange(1901, 2016), name='year') #fills in the gap years
by_year_nat_sz = df.groupby(['year', 'country'])                    .size().unstack().reindex(new_index).fillna(0)
    
# Create a region column with 2 or 3 largest countries in each region
regions = [
{'label':'N. America',
'countries':['United States', 'Canada']},
{'label':'Europe',
'countries':['United Kingdom', 'Germany', 'France']},
{'label':'Asia',
'countries':['Japan', 'Russia and Soviet Union', 'India']}
]
for region in regions:
    by_year_nat_sz[region['label']] =    by_year_nat_sz.loc[:,region['countries']].sum(axis=1)
    
by_year_nat_sz.loc[:,[r['label'] for r in regions]].cumsum().plot()

COL_NUM = 4 
ROW_NUM = 4

by_nat_sz = df.groupby('country').size()
by_nat_sz.sort_values(ascending=False, inplace=True) #sort countries from highest to lowest win haul
fig, axes = plt.subplots(COL_NUM, ROW_NUM, sharex=True, sharey=True, figsize=(12,12))
for i, nat in enumerate(by_nat_sz.index[1:17]): # enumerate from 2nd row (1) excluding the US (0)
    ax = axes[i//COL_NUM, i%ROW_NUM]
    by_year_nat_sz.loc[:,nat].cumsum().plot(ax=ax)
    ax.set_title(nat)

import seaborn as sns
plt.style.use('seaborn')

# create categorical buckets out of the continuous year
bins = np.arange(df.year.min(), df.year.max(), 10)
by_year_nat_binned = df.groupby(
        ['country', pd.cut(df.year, bins, precision=0)]).size().unstack().fillna(0)
plt.figure(figsize=(8,8))
#sns.heatmap(by_year_nat_binned[by_year_nat_binned.sum(axis=1) > 2], cmap="YlGnBu")
#sns.heatmap(by_year_nat_binned[by_year_nat_binned.sum(axis=1) > 2], cmap="gist_earth_r")
#sns.heatmap(by_year_nat_binned[by_year_nat_binned.sum(axis=1) > 2], cmap="RdBu_r")
sns.heatmap(by_year_nat_binned[by_year_nat_binned.sum(axis=1) > 2], cmap="OrRd")

df['award_age'].hist(bins=20)

sns.distplot(df['award_age'])

sns.boxplot(df.gender, df.award_age)

sns.violinplot(df.gender, df.award_age)

df['age_at_death'] = (df.date_of_death - df.date_of_birth).dt.days/365

df.date_of_birth.head()


age_at_death = df[df.age_at_death.notnull()].age_at_death
sns.distplot(age_at_death, bins=40)

df[df.age_at_death > 100][['name', 'category', 'year']]

df_temp = df[df.age_at_death.notnull()]

sns.kdeplot(df_temp[(df_temp.gender == 'male')].age_at_death, shade=True, label='male')
sns.kdeplot(df_temp[(df_temp.gender == 'female')].age_at_death, shade=True, label='female')
plt.legend()

sns.violinplot(df.gender, age_at_death)

df_temp=df[df.age_at_death.notnull()]
data = pd.DataFrame(
{'age at death':df_temp.age_at_death,
'date of birth':df_temp.date_of_birth.dt.year})
sns.lmplot('date of birth', 'age at death', data, size=6, aspect=1.5)

df = pd.read_json('data/nwinners_born_in.json', orient='records')
df

by_bornin_nat = df[df.born_in.notnull()].groupby(        ['born_in', 'country']).size().unstack()
by_bornin_nat.index.name = 'Born in'
by_bornin_nat.columns.name = 'Moved to'
plt.figure(figsize=(8,8))
ax = sns.heatmap(by_bornin_nat, vmin=0, vmax=8)
ax.set_title('The Nobel Diaspora')

df.date_of_birth = pd.to_datetime(df.date_of_birth).dt.date

df.loc[(df.born_in == 'Germany') & (df.country == 'United Kingdom'), ['name', 'date_of_birth', 'category', 'born_in', 'country']]



