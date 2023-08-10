from datetime import datetime, timedelta
import pymongo
from sys import version
from pymongo import MongoClient
print ' Reproducibility conditions for this notebook '.center(90,'-')
print 'Python version:       ' + version
print 'Pymongo version:      ' + pymongo.version
print '-'*90

try: 
    client = MongoClient("localhost", 27017)
    print "Connected to MongoDB as:", client
except pymongo.errors.ConnectionFailure, e:
    print "Could not connect to MongoDB: %s" % e 

db = client.test_database
print db

# to prevent colision cases in db with previous db connetions: 

for name in db.collection_names():
    if name != 'system.indexes':
        db.drop_collection(name)

db.collection_names()

db.create_collection("test")

document = {"x": "jpcolino", "tags": ["author", "developer", "tester"]}

db.test.insert_one(document)

print '-'*75
print 'Databases open in client: ', client.database_names()
print 'Collection names in db:   ', db.collection_names()
print '-'*75

result = db.test.insert_many([{"x": 1, "tags": ["dog", "cat"]},
                              {"x": 2, "tags": ["cat"]},
                              {"x": 2, "tags": ["mouse", "cat", "dog"]},
                              {"x": 3, "tags": []},
                              {"y": 4, "tags": 123456}])

# Updating a document with $rename

db.test.update_one({"y": 4},{"$rename": {"y":"x"}})

for doc in db.test.find():
    print doc

# Deleting a document with delete_one

db.test.delete_one({'x':'jpcolino'})

for doc in db.test.find():
    print doc

import bson

class Example(object):
    def __init__(self):
        self.a = 'a'
        self.b = 'b'
    def set_c(self, c):
        self.c = c
        
        
e = Example()
e.__dict__

e.set_c(123)
e.__dict__

db.test.insert_one(e.__dict__)

for doc in db.test.find():
    print doc

db.test.delete_one({'a':'a'})

for doc in db.test.find():
    print doc

# Retrieving basic information over the collection:

print 'Name of the Database: \n', db.test.name
print '-'*75
print 'Full descriptions: \n', db.test.acknowledged
print '-'*75
for ids in result.inserted_ids:
    print 'ObjectID: ', ids
print '-'*75
for i, doc in enumerate(db.test.find()[1:]):
    print 'Doc {0}: {1}'.format(i, doc)
print '-'*75

print 'Number of Documents: ', db.test.count()
print '-'*75
print "Number of Documents with field 'x':",db.test[{'x': 2}]
print '-'*75
print 'Number of Documents where x == 2: ', db.test.find({'x': 2}).count()
print '-'*75
print 'Number of Documents with x >= 2: ', db.test.find({'x':{'$gte': 2}}).count()
print '-'*75
# print 'Number of Documents with x >= 2: ', db.test[$not'x'].find().count()

# Using Regex to find tags = cats

import re
regex = re.compile(r'cat')
rstats = db.test.find({"tags":regex}).count()
print 'Number of Documents where you will find a "cat": ', rstats

from pymongo import DESCENDING
db.test.drop_indexes()
db.test.create_index([("x", pymongo.DESCENDING)], name = "id_x")

print db.test.index_information() 
for doc in db.test.index_information() :
    print doc

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is the 10x10 Hilbert matrix (Numpy matrix)
# y is a Numpy array

X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

X

y

from bson.binary import Binary
import cPickle

collection = db.test_database.np_inputs

# cleaning db before to store anything:

for name in db.collection_names():
    if name != 'system.indexes':
        db.drop_collection(name)

db.collection_names()

inputs = {}
inputs['X'] = Binary( cPickle.dumps( X, protocol=2) ) # Using cPickle with fast protocol=2.
inputs['y'] = y.tolist()

db.collection.insert_one({'inputs': inputs })

for i, doc in enumerate(db.collection.find()):
    print 'Doc {0}: \n {1}'.format(i, doc)

for doc in db.collection.find():
    if isinstance(doc.get(u'inputs'),dict):
        inputs = doc.get(u'inputs')
        y = inputs.get(u'y')
        X = np.hstack([cPickle.loads(inputs.get(u'X'))]) 

# Display of data retrived: 

print '-'*65
print 'y data stored as:', type(y)
print 'y: \t', y
print '-'*65
print 'X data stored as:', type(X)
print 'X: \n',X
print '-'*65

from sklearn import linear_model

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_)

# Display results
get_ipython().magic('matplotlib inline')
plt.figure(num=None, figsize=(18, 9))
plt.style.use('ggplot')

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight');

outputs = {}
outputs['coefs'] = Binary( cPickle.dumps( coefs, protocol=2) ) # Using cPickle with fast protocol=2
db.collection.insert_one({'outputs': outputs} )

del coefs

for doc in db.collection.find():
    if isinstance(doc.get(u'outputs'),dict):
        outputs = doc.get(u'outputs')
        coefs = np.hstack([cPickle.loads(outputs.get(u'coefs'))])
        

plt.figure(num=None, figsize=(18, 9))
plt.style.use('ggplot')
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight');        



