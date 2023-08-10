from pymongo import MongoClient
from datetime import datetime
import json
import pdb
import csv
import tqdm

ip = '52.54.120.114'

conn = MongoClient(ip, 27017)
conn.database_names()

db = conn.get_database('cleaned_data')

db.collection_names()

biz = db.get_collection('restaurants')
users = db.get_collection('users')
reviews = db.get_collection('restaurant_reviews')

biguser = []

for obj in users.find({'review_count':{'$gt':500}}):
    biguser.append(obj['user_id'])

userreview = {}

for i in tqdm.tqdm(range(0,len(biguser[0:20]))):
    ulist = []
    for obj in reviews.find({'user_id':biguser[i]}):
        del obj['_id']
        ulist.append(obj)
    userreview[str(biguser[i])] = ulist

with open('user_review_dictionary.json', 'w') as outfile:
    json.dump(userreview, outfile)

biznames =[]

for key in userreview.keys():
    for review in userreview[key]:
        biznames.append(review['business_id'])

restreview = {}

for i in tqdm.tqdm(range(0, len(biznames))):
    rlist = []
    for obj in reviews.find({'business_id':biznames[i]}):
        rlist.append(obj)
    restreview[biznames[i]] = rlist

for key in restreview.keys():
    for review in restreview[key]:
        if '_id' in review: del review['_id']


with open('rest_review_dictionary.json', 'w') as outfile:
    json.dump(restreview, outfile)

