get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import cPickle, os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from ddlite import *

import os, codecs, json, shutil, glob

#assign variables to path to the json files
FREVIEW = os.path.join('yelp_data', 'yelp_academic_dataset_review.json')
FBUSINESS = os.path.join('yelp_data', 'yelp_academic_dataset_business.json')

#delete any existing files in reviews folder
files = glob.glob('/yelp_pizza_reviews/*')
for f in files:
    os.remove(f)

def getReviews(quantOfRest=100000, quantOfReviewsPerRest=5000000):
    #get restaurant ids from business ids
    restaurantIDs = []
    with codecs.open(FBUSINESS,'rU','utf-8') as f:
        for business in f:
            if "Restaurants" in json.loads(business)["categories"]:
                if "Pizza" in json.loads(business)["categories"]:
                    restaurantIDs.append(json.loads(business)['business_id'])
    print "Pizza restaurantIDs count", len(restaurantIDs)
    
    #create dictionary of RestaurantID to Reviews
    dictRestaurantIDsToReview = {}
    with codecs.open(FREVIEW,'rU','utf-8') as f:
        for review in f:
            reviewText = json.loads(review)['text']
            ID = json.loads(review)['business_id']
            if ID in restaurantIDs:
                if ID in dictRestaurantIDsToReview.keys():
                    if len(dictRestaurantIDsToReview.get(ID)) < quantOfReviewsPerRest:
                        dictRestaurantIDsToReview.get(ID).append(reviewText)
                else:
                    if len(dictRestaurantIDsToReview.keys()) < quantOfRest:
                        dictRestaurantIDsToReview[ID] = [reviewText]
                    else:
                        break
    return dictRestaurantIDsToReview

#get reviews in the form of a dictionary
dictRestaurantIDsToReview = getReviews(quantOfReviewsPerRest=50)

#save reviews to folder as text files.  Each restaurant has separate review file.
count = 0
for restID in dictRestaurantIDsToReview.keys():
    reviews = ""
    for review in dictRestaurantIDsToReview[restID]:
        review = review.encode('ascii', errors='ignore') + " "
        count += 1
        reviews += review
    open("yelp_pizza_reviews/reviews_" + restID + ".txt", "w+").write(reviews)

#try to remove .DS_Store file.  Otherwise DocParser throws an exception
try:
    os.remove("yelp_pizza_reviews/.DS_Store")
except:
    print "No .DS_Store file"
    
print count

dp = DocParser('yelp_pizza_reviews/')
docs = list(dp.readDocs())

docs = None

pkl_f = 'yelp_tag_saved_sents_v4.pkl'
try:
    with open(pkl_f, 'rb') as f:
        sents = cPickle.load(f)
except:
    get_ipython().magic('time sents = dp.parseDocSentences()')
    with open(pkl_f, 'w+') as f:
        cPickle.dump(sents, f)

print sents[0]

toppings = ["mushroom","pepperoni","sausage","hawaiian","pineapple","beef","pork","chicken",
            "Italian","salami","meatball","ham","bacon","spinach","tomato","onion","pepper"]

def gen_regex_match(topping):
    pattern = topping + r"\s\w+\spizza"
    m1 = RegexNgramMatch(label=topping+"m1", regex_pattern=pattern, ignore_case=True)
    pattern = topping + r"\s\w+\s\w+\spizza"
    m2 = RegexNgramMatch(label=topping+"m2", regex_pattern=pattern, ignore_case=True)
    pattern = topping + r"\s\w+\s\w+\s\w+\spizza"
    m3 = RegexNgramMatch(label=topping+"m3", regex_pattern=pattern, ignore_case=True)
    return [m1, m2, m3]

args = []
for topping in toppings:
    args += gen_regex_match(topping)

    
# old rules
#pizza_regex1 = RegexNgramMatch(label='Pizza', regex_pattern=r'\w+\spizza', ignore_case=True)
#pizza_regex2 = RegexNgramMatch(label='Pizza', regex_pattern=r'\w+\s\w+\spizza', ignore_case=True)
#pizza_regex3 = RegexNgramMatch(label='Pizza', regex_pattern=r'\w+\s\w+\s\w+\spizza', ignore_case=True)
#pizza_regex4 = RegexNgramMatch(label='Pizza', regex_pattern=r'\w+\s\w+\s\w+\s\w+\spizza', ignore_case=True)

#combine all matchers
CE = Union(*args)

E = Entities(sents, CE)

# Number of entities we extracted
len(E)

E[0].render()

E[1].mention(attribute='words')

E.dump_candidates('yelp_tag_saved_entities_v5.pkl')



