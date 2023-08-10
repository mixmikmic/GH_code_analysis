goodnight_moon = [
    "In the great green room There was a telephone And a red balloon",
    "And a picture of-",
    "The cow jumping over the moon",
    "And there were three little bears sitting on chairs And two little kittens",
    "And a pair of mittens",
    "And a little toy house",
    "And a young mouse",
    "And a comb and a brush and a bowl full of mush And a quiet old lady who was whispering “hush” Goodnight room",
    "Goodnight moon",
    "Goodnight cow jumping over the moon Goodnight light",
    "And the red balloon",
    "Goodnight bears",
    "Goodnight chairs",
    "Goodnight kittens",
    "And goodnight mittens",
    "Goodnight clocks",
    "And goodnight socks",
    "Goodnight little house",
    "And goodnight mouse",
    "Goodnight comb",
    "And goodnight brush",
    "Goodnight nobody",
    "Goodnight mush",
    "And goodnight to the old lady whispering “hush” Goodnight stars",
    "Goodnight air",
    "Good night noises everywhere"
]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

lsa_pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svd', TruncatedSVD(n_components=5))
])

lsa_pipe.fit(goodnight_moon)

lsa_pipe

from sklearn.externals import joblib

joblib.dump(lsa_pipe, 'lsa_pipe.p')

fit_lsa_pipe = joblib.load('lsa_pipe.p')

import pymongo
cli = pymongo.MongoClient(host='54.201.203.180')
cli.database_names()

books = cli.books

goodnight_moon_coll = books.goodnight_moon

goodnight_moon[0]

goodnight_moon_coll.insert_one({'text': goodnight_moon[0]})

for text in goodnight_moon[1:]:
    goodnight_moon_coll.insert_one({'text': text})

goodnight_moon_coll.count()

curs = goodnight_moon_coll.find()

curs = goodnight_moon_coll.find()
for doc in range(curs.count()):
    this_doc = curs.next()
    this_text = this_doc['text']
    this_text_vec = fit_lsa_pipe.transform([this_text])
    this_doc['text_vec'] = this_text_vec
    goodnight_moon_coll.update({'text':this_text}, this_doc)

