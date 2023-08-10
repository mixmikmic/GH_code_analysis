# for data manipulation
import numpy as np
import pandas as pd
# for MongoDB connection
import pymongo
import matplotlib as plt
# for statistical hypothesis testing
import scipy.stats
get_ipython().magic('matplotlib inline')

# for interactive plotting
import plotly.plotly as py
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.set_config_file(offline=True, theme='ggplot')
print __version__ # requires version >= 1.9.0

def read_mongo(collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame """
    # Connect to MongoDB and Make a query to the specific DB and Collection
    with pymongo.MongoClient(host, port) as client:
        table = client.appstore[collection]
        df = pd.DataFrame(list(table.find(query)))
        
    # Delete the _id
    if no_id:
        del df['_id']

    return df

apps_df = read_mongo('appitems')

apps_df.shape #5658 unique apps

apps_df.info()

rating_cleaned = {'1 star':1, "1 and a half stars": 1.5, '2 stars': 2, '2 and a half stars':2.5, "3 stars":3, "3 and a half stars":3.5, "4 stars": 4,
                 '4 and a half stars': 4.5, "5 stars": 5}

apps_df.overall_rating = apps_df.overall_rating.replace(rating_cleaned)
#apps_df.to_pickle('app_cleaned.pickle')

cate_cnt = apps_df.groupby(['category', 'overall_rating'])['id'].count().reset_index()
rate_cate_cnt = cate_cnt.pivot_table(index = 'category', columns = 'overall_rating', values = 'id', fill_value= 0)

rate_cate_cnt.iplot(kind = 'bar', barmode = 'stack', yTitle='Number of Apps', title='Distribution of Apps by Category and Rating', 
                    colorscale = 'Paired', theme='white', labels = 'Rating')

rating_df = apps_df[["name","overall_rating", "current_rating", 'num_current_rating', "num_overall_rating"]].dropna()

rating_df.iplot(kind = "bubble", x = "overall_rating", y = "current_rating", size = 'num_current_rating', text = 'name',
              xTitle='Overall Rating', yTitle='Current Rating')

rating_df.iplot(kind = "scatter", mode = "markers", x = "current_rating", y = "num_current_rating", text = "name", size = 5, xTitle = "Current Rating",
               yTitle = "Num of Current Rating")

# py.iplot(
#     {
#         'data': [
#             {
#                 'x': df[df['year']==year]['gdpPercap'],
#                 'y': df[df['year']==year]['lifeExp'],
#                 'name': year, 'mode': 'markers',
#             } for year in [1952, 1982, 2007]
#         ],
#         'layout': {
#             'xaxis': {'title': 'GDP per Capita', 'type': 'log'},
#             'yaxis': {'title': "Life Expectancy"}
#         }
# }, )

rating_df['weighted_rating'] = map(lambda a, b, c,d: np.divide(a,b)*c+(1-np.divide(a,b))*d, rating_df['num_current_rating'], 
                                   rating_df['num_overall_rating'], rating_df['current_rating'], rating_df['overall_rating'])

rating_df[['weighted_rating', 'current_rating','overall_rating']].iplot(kind='histogram', barmode='stack', theme='white', title = 'Distribution of Rating Metrics')

free_df = apps_df[(apps_df['is_InAppPurcased'] == 0)&(pd.notnull(apps_df['overall_rating']))][["name","overall_rating", "current_rating", 'num_current_rating', "num_overall_rating"]]

paid_df = apps_df[(apps_df['is_InAppPurcased'] == 1)&(pd.notnull(apps_df['overall_rating']))][["name","overall_rating", "current_rating", 'num_current_rating', "num_overall_rating"]]

free_df['weighted_rating'] = map(lambda a, b, c,d: np.divide(a,b)*c+(1-np.divide(a,b))*d, free_df['num_current_rating'], 
                                   free_df['num_overall_rating'], free_df['current_rating'], free_df['overall_rating'])
paid_df['weighted_rating'] = map(lambda a, b, c,d: np.divide(a,b)*c+(1-np.divide(a,b))*d, paid_df['num_current_rating'], 
                                   paid_df['num_overall_rating'], paid_df['current_rating'], paid_df['overall_rating'])

free = list(free_df['weighted_rating'])
paid = list(paid_df['weighted_rating'])

scipy.stats.kruskal(free, paid)



