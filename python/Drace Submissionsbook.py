import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn import preprocessing
color = sns.color_palette()
get_ipython().magic('matplotlib inline')

df_train = pd.read_json('price_vs_median72.json')
df_test = pd.read_json('test.json')

df_test.head()

append_train = pd.read_csv("neighborhood_values_train_72.csv")
df_train = pd.merge(df_train, append_train, how = 'inner', on=['listing_id', 'listing_id'])

append_test = pd.read_csv("neighborhood_values_test_72.csv")
df_test = pd.merge(df_test, append_test, how='inner', on=['listing_id', 'listing_id'])

df_train['price_vs_median_72_new'] = df_train['price']/df_test['median_72']
df_test['price_vs_median_72_new'] = df_test['price']/df_test['median_72']

from pandas import to_datetime
import numpy as np
import re


def basic_numeric_features(df):
    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    df["num_description_words"] = df[
        "description"].apply(lambda x: len(x.split(" ")))
    df['weekday_created'] = df.created.dt.dayofweek
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df["created_hour"] = df["created"].dt.hour
    return df


def num_keyword(df):
    # n_num_keyword: check if a key word makes a difference in terms of
    # interest_level:
    match_list = [map(lambda x: re.search('elevator|cats|dogs|doorman|dishwasher|no fee|laundry|fitness', x.lower()),
                      list(df['features'])[i]) for i in np.arange(0, len(df['features']), 1)]
    nfeat_list = []
    for i in match_list:
        if i is None:
            nfeat_list.append(0)
        else:
            if not any(i):  # check to filter out lists with no all None values
                nfeat_list.append(0)
            else:
                lis1 = []
                map(lambda x: lis1.append(1) if x is None else lis1.append(0), i)
                nfeat_list.append(sum(lis1))

    # new variable n_num_keyfeat_score
    nfeat_score = []
    for i in nfeat_list:
        if i <= 5:
            nfeat_score.append(0)
        elif i == 6:
            nfeat_score.append(1)
        elif i == 7:
            nfeat_score.append(2)
        elif i == 8:
            nfeat_score.append(3)
        elif i == 9:
            nfeat_score.append(4)
        elif i == 10:
            nfeat_score.append(5)
        else:
            nfeat_score.append(6)

    df['n_num_keyfeat_score'] = nfeat_score
    return df


def no_photo(df):
    df['n_no_photo'] = [1 if i == 0 else 0 for i in map(len, df['photos'])]
    return df


def count_caps(df):
    def get_caps(message):
        caps = sum(1 for c in message if c.isupper())
        total_characters = sum(1 for c in message if c.isalpha())
        if total_characters > 0:
            caps = caps / (total_characters * 1.0)
        return caps
    df['amount_of_caps'] = df['description'].apply(get_caps)
    return df


def has_phone(df):
    # http://stackoverflow.com/questions/16699007/regular-expression-to-match-standard-10-digit-phone-number
    phone_regex = "(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})"
    has_phone = df['description'].str.extract(phone_regex)
    df['has_phone'] = [type(item) == unicode for item in has_phone]
    return df


def n_log_price(df):
    # n_price_sqrt improves original 'price' variable smoothing extreme
    # right skew and fat tails.
    # Use either 'price' or this new var to avoid multicolinearity.
    df['n_log_price'] = np.log(df['price'])
    return df


def n_expensive(df):
    # 'Low' interest make 70% population. Statistical analysis shows price
    # among 'Low' interest exhibits the highest kurtosis and skew.
    # n_expensive is 1 when the price is above 75% percentile aggregate
    # prices and 0 otherwise.
    # you can use it along with either price or n_price_sqrt.
    threshold_75p = df[['price']].describe().loc['75%', 'price']
    df['n_expensive'] = [
        1 if i > threshold_75p else 0 for i in list(df['price'])]
    return df


def dist_from_midtown(df):
    from geopy.distance import vincenty
    # pip install geopy
    # https://github.com/geopy/geopy
    # calculates vincenty dist
    # https://en.wikipedia.org/wiki/Vincenty's_formulae
    lat = df['latitude'].tolist()
    long_ = df['longitude'].tolist()
    midtown_lat = 40.7586
    midtown_long = -73.9838
    distance = []
    for i in range(len(lat)):
        distance.append(
            vincenty((lat[i], long_[i]), (midtown_lat, midtown_long)).meters)
    df['distance_from_midtown'] = distance
    return df


def price_vs_mean_30(df):
    # userfriendly for def_nearest_neighbour created earlier.
    # Output: df with price_vs_median for each row
    # The code below solves NA issues and round some results to save execution errors
    temp = pd.read_json("price_vs_median30.json")['price_vs_median_30']
    mean = np.mean(temp) 
    import math
    df['price_vs_median_30'] = [mean if math.isnan(i)== True  else round(i,2) for i in temp]

    return df

def log_price_vs_median(df):
    # n_price_sqrt improves original 'price' variable smoothing extreme
    # right skew and fat tails.
    # Use either 'price' or this new var to avoid multicolinearity.
    df['log_price_vs_median'] = np.log(df['price_vs_median_72_new'])
    return df

df_train.created = pd.to_datetime(df_train.created)

#distance for boroughs, not sure if useful


queensCenter = ((40.800760+40.542920)/2,(-73.700272-73.962616)/2)
brookCenter = ((40.739877+40.57042)/2,(-73.864754-74.04344)/2)
bronxCenter = ((40.915255+40.785743)/2,(-73.765274-73.933406)/2)
manhattanCenter = ((40.874663+40.701293)/2,(-73.910759-74.018721)/2)
siCenter = ((40.651812+40.477399)/2,(-74.034547-74.259090)/2)
boroughDict = {}
boroughDict["queens"] = queensCenter
boroughDict["brooklyn"] = brookCenter
boroughDict["bronx"] = bronxCenter
boroughDict["manhattan"] = manhattanCenter
boroughDict["staten"] = siCenter

#This function returns the string representation of the likely borough, given a set of latitude/longitude coordinates
#If the distance to the borough center is too far away from the closest borough, we assume that the location
#is outside of NYC
def get_closest_borough(latitude,longitude,max_dist = 20):
    global boroughDict
    borough_distances = {borough:great_circle(boroughDict[borough],(latitude,longitude)).miles for borough in boroughDict}
    min_borough = min(borough_distances, key=borough_distances.get)
    if borough_distances[min_borough] < max_dist:
        return min_borough 
    else:
        return "outside_nyc"
    

def dist_to_nearest_college(df):
    Baruch = (40.7402, -73.9834)
    Columbia = (40.8075, -73.9626)
    Cooper_Union = (40.7299, -73.9903)
    FIT = (40.7475, -73.9951)
    Hunter_College = (40.7685, -73.9657)
    Julliard = (40.7738, -73.9828)
    NYU = (40.7295, -73.9965)
    Pace_University=(40.7111, -74.0049)
    schools = [Baruch,Columbia,Cooper_Union,FIT,Hunter_College, NYU, Pace_University, Julliard
              ]

    from geopy.distance import vincenty
    import numpy as np
    distance = []
    for i in range(0,len(df['latitude']),1):
        lat_long = (list(df['latitude'])[i],list(df['longitude'])[i])
        temp=[]
        for j in schools:
            temp.append(
            vincenty(lat_long, j).meters)
        distance.append(min(temp))
    df['dist_to_nearest_college']= distance
    return df    


df_train['interest_level'] = pd.Categorical(df_train['interest_level'], categories= ['low', 'medium', 'high'], ordered=True)

lbl = preprocessing.LabelEncoder()
lbl.fit(list(df_train['manager_id'].values))
df_train['manager_id'] = lbl.transform(list(df_train['manager_id'].values))
#list of id's to encode
manager_ids = list(df_train['manager_id'].values)
#new var to create
new_var = 'manager_id'#'manager_id_encoded'
#response var
resp_var = 'interest_level'


temp = pd.concat([df_train[new_var], pd.get_dummies(df_train[resp_var])], axis = 1).groupby(new_var).mean()
temp.columns = ['high_frac','low_frac', 'medium_frac']
temp['count'] = df_train.groupby(new_var).count().iloc[:,1]

# compute skill
temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']
unranked_managers_ixes = temp['count']<20
ranked_managers_ixes = ~unranked_managers_ixes
mean_values = temp.loc[ranked_managers_ixes, ['high_frac','low_frac', 'medium_frac','manager_skill']].mean()
temp.loc[unranked_managers_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values

df_train['Price_P_Room'] = df_train['price']/df_train['bedrooms']
df_train['BB_ratio'] = df_train['bedrooms']/df_train['bathrooms']
df_train['is_studio'] = np.where(df_train['bedrooms'] == 0, '1', '0')
df_train['bathroom_listed'] = np.where(df_train['bathrooms'] == 0, '0', '1')
df_train['n_log_price'] = np.log(df_train['price'])
df_train.bedrooms[df_train.bedrooms == 0] = 1
df_train.bathrooms[df_train.bathrooms ==0] = 1

num_keyword(df_train)
no_photo(df_train)
count_caps(df_train)
n_expensive(df_train)
dist_from_midtown(df_train)

#Indepth Features


def dist_to_nearest_tube(df):
    tube_lat_long = pd.read_csv('http://web.mta.info/developers/data/nyct/subway/StationEntrances.csv')         [['Station_Name','Station_Latitude','Station_Longitude']]    

    tube_lat_long = tube_lat_long.groupby('Station_Name').agg(['mean']) # unique stations only

    stations=[]
    for i in range(0,len(tube_lat_long),1):
            stations.append(
                (tube_lat_long.iloc[:,0][i],tube_lat_long.iloc[:,1][i]))

    from geopy.distance import vincenty
    import numpy as np
    distance = []
    for i in range(0,len(df['latitude']),1):
        lat_long = (list(df['latitude'])[i],list(df['longitude'])[i])
        temp=[]
        for j in stations:
            temp.append(
            vincenty(lat_long, j).meters)
        distance.append(min(temp))

    df['dist_to_nearest_tube']= distance
    return df



def manager_skill(df):
    #new var to create
    new_var = 'manager_id'#'manager_id_encoded'
    #response var
    resp_var = 'interest_level'
    # Step 1: create manager_skill ranking from training set:
    train_df = pd.read_json("train.json") # upload training scores => test data cannot create a rank skill
    temp = pd.concat([train_df[new_var], pd.get_dummies(train_df[resp_var])], axis = 1).groupby(new_var).mean()
    temp.columns = ['high_frac','low_frac', 'medium_frac']
    temp['count'] = train_df.groupby(new_var).count().iloc[:,1]
    temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']
    # Step 2: fill working dataset (e.g. test set) with ranking figures and replace new manager_id not present in our
    # training set with an average assumption:
    manager_skill=[]
    for i in df['manager_id']:
        for j in temp.index:
            if i==j:
                manager_skill.append(temp['manager_skill'][j])
            else:
                manager_skill.append(-1) # we flag this to replace it for average later and control for manager_ids not present in training _df
    # Step 3: Replacing new manager_id scores not available in training set with the mean: 
    mean_manager_skill= np.mean(manager_skill)
    manager_skill_clean = [mean_manager_skill if i==-1 else i for i in manager_skill] # replace NA (labelled as -1 earlier) for the mean

    df['manager_skill'] = manager_skill_clean
    return df

#apply all features to check
regex_col0 = {"nofee":'no fee', "doorman": 'doorman', "fitness": 'fitness|swimming', "hardwood": "hardwood",             "dishwash": 'dishwasher', "preWar": 'prewar|pre-war', 'furnished': 'furnished', "laundry": 'laundry',            "allow_pets": 'cats|dogs'}
import re

def create_regex_col0(df,regex,colname):
    def find_regex(lis):
        text = ' '.join(lis)
        r = re.compile(regex,flags=re.IGNORECASE)
        matches = r.findall(text)
        return len(matches)
    df[colname] = df['features'].apply(find_regex)
    
for name, regex in regex_col0.items():
    create_regex_col0(df_train,regex,name)

#apply some samples for desc, jake has updated version
regex_col = {"subway":'train|trains|subway|line', "luxurious": 'luxury', "quiet_nei": 'quiet', "available": "available (immediately|now)",             "space_desc": 'foot|feet', "buzzword": 'must see'}

def create_regex_col(df,regex,colname):
    def find_regex(text):
        r = re.compile(regex,flags=re.IGNORECASE)
        matches = r.findall(text)
        return len(matches)
    df[colname] = df['description'].apply(find_regex)

for name, regex in regex_col.items():
    create_regex_col(df_train,regex,name)

df_train.allow_pets.describe()

lbl = preprocessing.LabelEncoder()
lbl.fit(list(df_train['building_id'].values))
df_train['building_id'] = lbl.transform(list(df_train['building_id'].values))

#tack on manager skill

drace_df = df_train.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
new_manager_ixes = drace_df['high_frac'].isnull()
drace_df.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values

drace_df = drace_df.dropna()

drace_df = drace_df.fillna(mean_values)

dist_to_nearest_college(drace_df)

has_phone(drace_df)
basic_numeric_features(drace_df)

dist_to_nearest_tube(drace_df)

feats_used = ['bathrooms', 'manager_skill', 'bedrooms', 'created_day', 'latitude', 'longitude',             'n_log_price', 'price_vs_median_72', 'num_photos', 'num_features', 'num_description_words', 'weekday_created',             'created_hour', 'is_studio', 'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets',              'laundry', 'preWar','furnished','dishwash','hardwood','fitness','doorman','nofee', 'has_phone',                'dist_to_nearest_college']
x = drace_df[feats_used]
y = drace_df["interest_level"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train[feats_used], y_train)
y_pred = rf.predict_proba(x_test[feats_used])
log_loss(y_test, y_pred)

from sklearn import metrics

metrics.accuracy_score(rf.predict(x_test), y_test)

pd.Series(index = feats_used, data = rf.feature_importances_).sort_values().plot(kind = 'bar')

#compute cross-validation score accuracy across 5 folds
#cross_val_scores = cross_val_score(rf,x,y,cv=5)
#print "5-fold accuracies:\n",cross_val_scores
#print "Mean cv-accuracy:",np.mean(cross_val_scores)
#print "Std of cv-accuracy:",np.std(cross_val_scores)

drace_df['price_vs_median_72_new'] = drace_df['price']/df_test['median_72']

drace_df.columns

#some baseline features according to some good rule of thumbs for gbm params

feats_used = ['bathrooms', 'bedrooms', 'created_day', 'latitude', 'longitude',             'n_log_price', 'price_vs_median_72', 'num_photos', 'num_features', 'num_description_words', 'weekday_created',             'created_hour', 'is_studio', 'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets',              'laundry', 'preWar','furnished','dishwash','hardwood','fitness','doorman','nofee', 'has_phone',              'dist_to_nearest_college']
from sklearn.ensemble import GradientBoostingClassifier
x = drace_df[feats_used]
y = drace_df["interest_level"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
gbm0 = GradientBoostingClassifier(n_estimators=1000, max_features=5)
gbm0.fit(x_train,y_train)
y_pred2 = gbm0.predict_proba(x_test[feats_used])
log_loss(y_test, y_pred2)

pd.Series(index = feats_used, data = gbm0.feature_importances_).sort_values().plot(kind = 'bar')

#some baseline features according to some good rule of thumbs for gbm params

feats_used = ['bathrooms', 'manager_skill', 'bedrooms', 'created_day', 'latitude', 'longitude',             'n_log_price', 'price_vs_median_72', 'num_photos', 'num_features', 'num_description_words', 'weekday_created',             'created_hour', 'is_studio', 'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets',              'laundry', 'preWar','furnished','dishwash','hardwood','fitness','doorman','nofee', 'has_phone',                'dist_to_nearest_college']
from sklearn.ensemble import GradientBoostingClassifier
x = drace_df[feats_used]
y = drace_df["interest_level"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
gbm1 = GradientBoostingClassifier(n_estimators=1000, max_features=5)
gbm1.fit(x_train,y_train)
y_pred2 = gbm1.predict_proba(x_test[feats_used])
log_loss(y_test, y_pred2)

pd.Series(index = feats_used, data = gbm1.feature_importances_).sort_values().plot(kind = 'bar')

from sklearn.grid_search import GridSearchCV
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8),                        param_grid = param_test1, scoring='log_loss',n_jobs=4,iid=False, cv=5)
gsearch1.fit(drace_df[feats_used],y)

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

#DO NOT RUN THESE PARAM TESTS UNLESS YOU WANT TO VALIDATE FOR YOURSELF.  TAKES VERY LONG!!!
param_test2 = {'max_depth':range(1,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_features='sqrt',subsample=0.8),                        param_grid = param_test2, scoring='log_loss',n_jobs=4,iid=False, cv=5)
gsearch2.fit(drace_df[feats_used],y)

gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

param_test3 = {'min_samples_split':range(1,400, 20), 'min_samples_leaf':range(30,71,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=11,max_features='sqrt',subsample=0.8),                        param_grid = param_test3, scoring='log_loss',n_jobs=4,iid=False, cv=5)
gsearch3.fit(drace_df[feats_used],y)

gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

param_test4 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,min_samples_leaf=30,max_depth=11,min_samples_split=241, max_features='sqrt'),                        param_grid = param_test4, scoring='log_loss',n_jobs=4,iid=False, cv=5)
gsearch4.fit(drace_df[feats_used],y)

gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

x = drace_df[feats_used]
y = drace_df["interest_level"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
gbmtuned = GradientBoostingClassifier(n_estimators=80, max_features='sqrt', subsample=0.9, min_samples_leaf=30, min_samples_split=241, max_depth=11)
gbmtuned.fit(x_train,y_train)
y_pred2 = gbmtuned.predict_proba(x_test[feats_used])
log_loss(y_test, y_pred2)

#something really wrong here ... I'll have to check when not so tired.

pd.Series(index = feats_used, data = gbmtuned.feature_importances_).sort_values().plot(kind = 'bar')

df_test.created = pd.to_datetime(df_test.created)
basic_numeric_features(df_test)
num_keyword(df_test)
n_log_price(df_test)
count_caps(df_test)
dist_from_midtown(df_test)
dist_to_nearest_college(df_test)

drace_df_test = df_test.reset_index().merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id').set_index('index')
new_manager_ixes = drace_df_test['high_frac'].isnull()
drace_df_test.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values

drace_df_test = df_test.merge(temp, how='left', left_on='manager_id', right_index=True)
# drace_df_test.reset_index(df_test.index)
drace_df_test.head()

#apply all features to check
regex_col0 = {"nofee":'no fee', "doorman": 'doorman', "fitness": 'fitness|swimming', "hardwood": "hardwood",             "dishwash": 'dishwasher', "preWar": 'prewar|pre-war', 'furnished': 'furnished', "laundry": 'laundry',            "allow_pets": 'cats|dogs'}
import re

def create_regex_col0(df,regex,colname):
    def find_regex(lis):
        text = ' '.join(lis)
        r = re.compile(regex,flags=re.IGNORECASE)
        matches = r.findall(text)
        return len(matches)
    df[colname] = df['features'].apply(find_regex)
    
for name, regex in regex_col0.items():
    create_regex_col0(drace_df_test,regex,name)

#apply some samples for desc, jake has updated version
regex_col = {"subway":'train|trains|subway|line', "luxurious": 'luxury', "quiet_nei": 'quiet', "available": "available (immediately|now)",             "space_desc": 'foot|feet', "buzzword": 'must see'}

def create_regex_col(df,regex,colname):
    def find_regex(text):
        r = re.compile(regex,flags=re.IGNORECASE)
        matches = r.findall(text)
        return len(matches)
    df[colname] = df['description'].apply(find_regex)

for name, regex in regex_col.items():
    create_regex_col(drace_df_test,regex,name)

has_phone(drace_df_test)
drace_df_test['is_studio'] = np.where(drace_df_test['bedrooms'] == 0, '1', '0')

new_manager_ixes = drace_df_test['high_frac'].isnull()
drace_df_test.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
feats_used = ['bathrooms', 'bedrooms', 'created_day', 'latitude', 'longitude', 'manager_skill',              'n_log_price', 'price_vs_median_72_new', 'num_photos', 'num_features', 'num_description_words', 'weekday_created',             'created_hour', 'is_studio', 'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets',              'laundry', 'preWar','furnished','dishwash','hardwood','fitness','doorman','nofee', 'has_phone',              'dist_to_nearest_college']
x_Test = drace_df_test[feats_used]
submissionRF = rf.predict_proba(x_Test)

feats_used = ['bathrooms', 'bedrooms', 'created_day', 'latitude', 'longitude', 'manager_skill',              'n_log_price', 'price_vs_median_72', 'num_photos', 'num_features', 'num_description_words', 'weekday_created',             'created_hour', 'is_studio', 'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets',              'laundry', 'preWar','furnished','dishwash','hardwood','fitness','doorman','nofee', 'has_phone',              'dist_to_nearest_college']
rf.predict_proba(x_test[feats_used])

testing = pd.DataFrame(rf.predict_proba(x_test[feats_used]))

print testing[0].sum()
print testing[1].sum()
print testing[2].sum()

drace_df.interest_level.value_counts()

submissionRF = pd.DataFrame(submissionRF)

submissionRF.head()

submissionRF = pd.concat([drace_df_test.reset_index(drop=True), submissionRF], axis=1)

submission_rf = submissionRF[['listing_id',0,1, 2]]

submissionRF[['listing_id',0,1, 2]].head()

submission_rf.rename(columns={0: 'high', 1: 'low', 2: 'medium'}, inplace=True)
submission_rf = submission_rf[['listing_id', 'high', 'medium', 'low']]

submission_rf.head()

submission_rf.to_csv('C:\\Users\\Drace\\Documents\\submissionRF.csv', index = False)

sample = pd.read_csv('C:\Users\Drace\Documents\Neuromancers-Kaggle-master\sample_submission.csv')

sample.columns

submission_rf.columns

feats_used = ['bathrooms', 'bedrooms', 'created_day', 'latitude', 'longitude',             'n_log_price', 'price_vs_median_72_new', 'num_photos', 'num_features', 'num_description_words', 'weekday_created',             'created_hour', 'is_studio', 'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets',              'laundry', 'preWar','furnished','dishwash','hardwood','fitness','doorman','nofee', 'has_phone',              'dist_to_nearest_college']

submissiongbm0 = pd.DataFrame(gbm0.predict_proba(x_Test[feats_used]))
submissiongbm0 = pd.concat([drace_df_test.reset_index(drop=True), submissiongbm0], axis=1)
submissiongbm0.rename(columns={0: 'high', 1: 'low', 2: 'medium'}, inplace=True)
submissiongbm0 = submissiongbm0[['listing_id', 'high', 'medium', 'low']]

drace_df_test.building_id

submissiongbm0.to_csv('C:\\Users\\Drace\\Documents\\submissiongbm0.csv', index = False)

feats_used = ['bathrooms', 'manager_skill', 'bedrooms', 'created_day', 'latitude', 'longitude',             'n_log_price', 'price_vs_median_72_new', 'num_photos', 'num_features', 'num_description_words', 'weekday_created',             'created_hour', 'is_studio', 'n_num_keyfeat_score','amount_of_caps', 'distance_from_midtown', 'allow_pets',              'laundry', 'preWar','furnished','dishwash','hardwood','fitness','doorman','nofee', 'has_phone',                'dist_to_nearest_college']

submissiongbm1 = pd.DataFrame(gbm1.predict_proba(x_Test[feats_used]))
submissiongbm1 = pd.concat([drace_df_test.reset_index(drop=True), submissiongbm1], axis=1)
submissiongbm1.rename(columns={0: 'high', 1: 'low', 2: 'medium'}, inplace=True)
submissiongbm1 = submissiongbm1[['listing_id', 'high', 'medium', 'low']]

submissiongbm1.to_csv('C:\\Users\\Drace\\Documents\\submissiongbm1.csv', index = False)

