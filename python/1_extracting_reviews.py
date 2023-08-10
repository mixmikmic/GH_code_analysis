# import necessary modules
from pymongo import MongoClient
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

# stop/start mongo service
get_ipython().system('sudo service mongod stop')
get_ipython().system('sudo service mongod start')

# connecting the MongoDB
client = MongoClient()
db = client.yelp

# print number of rows in each collection
for c in db.collection_names():
    print('{:10}'.format(c), '{:>10,}'.format(db[c].count()))

# connecting collections
reviews = db.review
businesses = db.business

# some Mongo testing
one_review = reviews.find_one()
one_business = businesses.find_one({'business_id':one_review['business_id']})
print('{:6}'.format('Name:'), one_business['name'])
print('{:6}'.format('City:'), one_business['city'])
pprint(one_business)

# how many businesses in each state
states = []
num_bsns = []
for i in businesses.aggregate([{"$group":{"_id":"$state", 'count':{"$sum":1}}}]):
    states.append(i['_id'])
    num_bsns.append(i['count'])

# create a dataframe
businesses_state = {'states':states, 'businesses':num_bsns}
bsns_state = pd.DataFrame(data=businesses_state, columns=['states', 'businesses'])

# get top 10 states
bsns_state_top10 = bsns_state.sort_values(by='businesses', ascending=False).head(10)

# plot the top 10 states by business count
plt.figure(figsize=(10,5))
plt.style.use('fivethirtyeight')
barlist = plt.barh(range(10), bsns_state_top10.businesses)
barlist[4].set_color('crimson')
plt.title('States with most businesses in Yelp dataset')
plt.yticks(range(10), bsns_state_top10.states)
plt.xlabel('Number of businesses');

get_ipython().run_cell_magic('time', '', "# return only OH business IDs\noh_bsns = []\noh_name = []\noh_cats = []\noh_city = []\noh_attr = []\noh_revs = []\noh_star = []\nfor i in businesses.find({'state':'OH'}, {'business_id':1, 'name':1, 'categories':1, 'stars':1,\n                                          'city':1, 'attributes':1, 'review_count':1, '_id':0}):\n    oh_bsns.append(i['business_id'])\n    oh_name.append(i['name'])\n    oh_cats.append(i['categories'])\n    oh_city.append(i['city'])\n    oh_attr.append(i['attributes'])\n    oh_revs.append(i['review_count'])\n    oh_star.append(i['stars'])\n\noh_dict = {'id':oh_bsns, 'categories':oh_cats, 'city':oh_city, 'attributes':oh_attr, \n           'reviews':oh_revs, 'name':oh_name, 'stars':oh_star}\noh = pd.DataFrame(data=oh_dict, columns=['id', 'name', 'city', 'stars', 'reviews', 'categories', 'attributes'])\nprint('Number of businesses in Ohio in this Yelp dataset:', '{:,}'.format(len(oh_bsns)))\n\n# load all reviews into a dataframe\ndf_all = pd.DataFrame(list(reviews.find({})))")

# get dataframe of business IDs of only restaurants
def find_in_list(stringy, listy):
    """Returns True/False if stringy is found in listy"""
    if stringy in listy:
        return True
    else:
        return False

# get business IDs of only a particular business type
businesses_to_investigate = 'Restaurants'
oh_rest = oh[oh.categories.map(lambda x: find_in_list(businesses_to_investigate, x))]

# how many review counts do the restaurants have
revs_x = np.arange(0, 101, 5) # list of numbers to use as minimum reviews treshold
rest_y = [oh_rest[oh_rest.reviews >= x].reviews.sum() for x in revs_x]
plt.style.use('fivethirtyeight')
plt.figure(figsize=(10,5))
plt.plot(revs_x, rest_y)
plt.xlim((min(revs_x), max(revs_x)))
plt.xticks(revs_x, revs_x)
plt.title('Reviews kept by implementing a minimum reviews rule')
plt.xlabel('Minimum review count')
plt.ylabel('Number of reviews in dataset');

oh_rest = oh_rest[oh_rest.reviews >= 30]

# write to CSV restaurant information so we can use it later
oh_rest.to_csv('OH_restaurants.csv')

get_ipython().run_cell_magic('time', '', '# keep only reviews that belong to OH restaurants\ndf = df_all[df_all.business_id.isin(oh_rest.id.values)]')

# write to CSV for ease of reading in later
df.info()
df.to_csv('reviews_OH_restaurants.csv')

# plot how many reviews we have of each star
star_x = df.stars.value_counts().index
star_y = df.stars.value_counts().values

plt.figure(figsize=(8,5))
# colors are in the order 5, 4, 3, 1, 2
bar_colors = ['darkgreen', 'mediumseagreen', 'gold', 'crimson', 'orange']
plt.bar(star_x, star_y, color=bar_colors, width=.6)
plt.xlabel('Stars (Rating)')
plt.ylabel('Number of Reviews')
plt.title('Number of Reviews by Rating for Ohio %s' %(businesses_to_investigate));

pos_reviews = df.text[df.stars>3].values
neg_reviews = df.text[df.stars<3].values
print('Postive Reviews:  {:,}'.format(len(pos_reviews)))
print('Negative Reviews:  {:,}'.format(len(neg_reviews)))

