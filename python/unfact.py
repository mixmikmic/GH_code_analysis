from matplotlib import pyplot
from pandas import DataFrame, scatter_matrix
from pymongo import MongoClient
from scipy import stats
get_ipython().magic('matplotlib inline')

MIN_SALIENCE = 0.01
news_collection = MongoClient().fakeko['news']

# news dataframe
news_list = news_collection.find({'skip': {'$ne': True}, 'text_analysed': True})
def get_news_fields(news):
    return [news['short_url'], news['domain'], news['sentiment_score'], 
            news['sentiment_magnitude'], news['language'], news['authors']]
news_data = [get_news_fields(news) for news in news_list]
df = DataFrame(data=news_data, columns=['short_url', 'domain', 'sentiment_score', 
                                        'sentiment_magnitude', 'language', 'authors'])
df['domain'] = df['domain'].astype('category')
df['language'] = df['language'].astype('category')

# entities dataframe
news_list = news_collection.find({'skip': {'$ne': True}, 'text_analysed': True})
def get_entities_fields(entity):
    return [entity['name'], entity['type'], entity['salience']]
entities = []
for news in news_list:
    for entity in news['entities']:
        if entity['salience'] >= MIN_SALIENCE:
            entities.append(get_entities_fields(entity))
entities_df = DataFrame(data=entities, columns=['name', 'type', 'salience'])

(df['domain'].value_counts() / df['domain'].count() * 100)[lambda x: x > 1]

terms = entities_df['name'][entities_df['type'] != 'OTHER']
(terms.value_counts() / terms.count() * 100).nlargest(30)

people = entities_df['name'][entities_df['type'] == 'PERSON']
(people.value_counts() / people.count() * 100).nlargest(30)

locations = entities_df['name'][entities_df['type'] == 'LOCATION']
(locations.value_counts() / locations.count() * 100).nlargest(30)

organisations = entities_df['name'][entities_df['type'] == 'ORGANIZATION']
(organisations.value_counts() / organisations.count() * 100).nlargest(30)

events = entities_df['name'][entities_df['type'] == 'EVENT']
(events.value_counts() / events.count() * 100).nlargest(30)

others = entities_df['name'][entities_df['type'] == 'OTHER']
(others.value_counts() / others.count() * 100).nlargest(30)

df['authors'].value_counts()

(df['language'].value_counts() / df['language'].count() * 100)

df.plot.scatter(x='sentiment_score', y='sentiment_magnitude')



