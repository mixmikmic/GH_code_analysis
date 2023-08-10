get_ipython().run_line_magic('pylab', 'inline')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.pipeline import FeatureUnion
from transformers import *
from scipy.sparse import coo_matrix
import simplejson as json
from datetime import datetime
from sklearn.cross_validation import train_test_split

# Loading the dataset
# Loading the reviews dataset
reviews_frame = pd.read_csv('reviews_restaurants_text.csv')

# Loading business and user dataset
def get_data(line, columns):
    d = json.loads(line)
    return dict((key, d[key]) for key in columns)

print ('Loading user dataset started--------')

# Loading user data
columns = ('user_id', 'name','average_stars')
with open('user.json') as f:
    user_frame = pd.DataFrame(get_data(line, columns) for line in f)
user_frame = user_frame.sort_values('user_id')
print ('Loading user dataset completed--------')

print ('Loading business dataset started--------')

# Loading business data
columns = ('business_id', 'name','categories','attributes','city','stars')
with open('business.json') as f:
    business = pd.DataFrame(get_data(line, columns) for line in f)

business = business.sort_values('business_id')

# Trimming the dataset by city
business_by_city = business['city'] == "Las Vegas"
business = business[business_by_city]

# Trimming the dataset by the category 'Restaurants'
business_frame = business
count = 0
for row in business_frame.itertuples():
#     count = count + 1
#     if (count%5000 == 0):
#         print (count)
    if 'Restaurants' not in row.categories:
        business_frame.drop([row.Index], inplace=True)
print (len(business_frame))
print ('Loading business dataset completed--------')

# Citation: https://github.com/lchiaying/yelp
# Feature Extraction
print ('Feature Extraction started---------')

encoding_category = One_Hot_Encoder('categories', 'list', sparse=False)
encoding_attribute = One_Hot_Encoder('attributes', 'dict', sparse=False)
encoding_city= One_Hot_Encoder('city', 'value', sparse=False)
rating = Column_Selector(['stars'])
encoding_union = FeatureUnion([ ('cat', encoding_category),('attr', encoding_attribute),('city', encoding_city), ('rating', rating) ])
encoding_union.fit(business_frame)

print ('Feature Extraction completed---------')

# Generating profile of the user

user = 'tL2pS5UOmN6aAOi3Z-qFGg' 

print ('Businesses for the reviews given by the selected user-----')

reviews_given_by_user = reviews_frame.ix[reviews_frame.user_id == user]
reviews_given_by_user['stars'] = reviews_given_by_user['stars'] - float(user_frame.average_stars[user_frame.user_id == user])
reviews_given_by_user = reviews_given_by_user.sort_values('business_id')

# list of ids of the businesses reviewed by the user
reviewed_business_id_list = reviews_given_by_user['business_id'].tolist()
reviewed_business = business_frame[business_frame['business_id'].isin(reviewed_business_id_list)]
reviewed_business = reviewed_business.sort_values('business_id')

print ('Profile creation started-------')

features = encoding_union.transform(reviewed_business)
profile = np.matrix(reviews_given_by_user.stars) * features

print ('Profile creation completed-------')

# Calculating cosine similarity of the unreviewed reviews with the user's profile
print ('Cosine similarity calculation started-----')

test_frame = business_frame[0:1000]
test_frame = test_frame.sort_values('business_id')
business_id_list = test_frame['business_id'].tolist()
features = encoding_union.transform(test_frame)
similarity = np.asarray(profile * features.T) * 1./(norm(profile) * norm(features, axis = 1))

print ('Cosine similarity calculation completed-----')

# Output the recommended restaurants
index_arr = (-similarity).argsort()[:10][0][0:10]
print ('Hi ' + user_frame.name[user_frame.user_id == user].values[0] + '\nCheck out these restaurants: ')
for i in index_arr:
    resturant = business_frame[business_frame.business_id == business_id_list[i]]
    print (str(resturant['name'].values[0]))

