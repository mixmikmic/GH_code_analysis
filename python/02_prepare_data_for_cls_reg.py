import io
import requests

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')

# fix random seed for reproducibility
seed = 2302
np.random.seed(seed)

# names of items and traits

items_names = ['non_likable', 'secure', 'attractive', 'unsympathetic', 'indecisive', 
               'unobtrusive', 'distant', 'bored', 'emotional', 'not_irritated', 'active', 
               'pleasant', 'characterless', 'sociable', 'relaxed', 'affectionate', 'dominant', 
               'unaffected', 'hearty', 'old', 'personal', 'calm', 'incompetent', 'ugly', 
               'friendly', 'masculine', 'submissive', 'indifferent', 'interesting', 'cynical', 
               'artificial', 'intelligent', 'childish', 'modest']

traits_names = ['warmth', 'attractiveness', 'compliance', 'confidence', 'maturity']

# load ratings (averaged across listeners)

path = "https://raw.githubusercontent.com/laufergall/ML_Speaker_Characteristics/master/data/generated_data/"

url = path + "ratings_SC_means.csv"
s = requests.get(url).content
ratings =pd.read_csv(io.StringIO(s.decode('utf-8')))

ratings.head()

# speaker scores

path = "https://raw.githubusercontent.com/laufergall/Subjective_Speaker_Characteristics/master/data/generated_data/"

url = path + "factorscores_SC_malespk.csv"
s = requests.get(url).content
scores_m =pd.read_csv(io.StringIO(s.decode('utf-8')))

url = path + "factorscores_SC_femalespk.csv"
s = requests.get(url).content
scores_f =pd.read_csv(io.StringIO(s.decode('utf-8')))

# rename dimensions
scores_m.columns = ['sample_heard', 'warmth', 'attractiveness', 'confidence', 'compliance', 'maturity']
scores_f.columns = ['sample_heard', 'warmth', 'attractiveness', 'compliance', 'confidence', 'maturity']

# join male and feame scores
scores = scores_m.append(scores_f)
scores['gender'] = scores['sample_heard'].str.slice(0,1)
scores['spkID'] = scores['sample_heard'].str.slice(1,4).astype('int')

scores.head()

# for each trait, assign instances into 3 classes

classes_m = pd.DataFrame(data = scores_m['sample_heard'])
classes_f = pd.DataFrame(data = scores_f['sample_heard'])

# male speakers
for i in traits_names:
    th = np.percentile(scores_m[i],[33,66])
    classes_m.loc[scores_m[i]<th[0],i] = 'low'
    classes_m.loc[scores_m[i]>=th[0],i] = 'mid'
    classes_m.loc[scores_m[i]>th[1],i] = 'high'

# female speakers
for i in traits_names:
    th = np.percentile(scores_f[i],[33,66])
    classes_f.loc[scores_f[i]<th[0],i] = 'low'
    classes_f.loc[scores_f[i]>=th[0],i] = 'mid'
    classes_f.loc[scores_f[i]>th[1],i] = 'high'
    
# join male and female classes
classes = classes_m.append(classes_f)
classes['gender'] = classes['sample_heard'].str.slice(0,1)
classes['spkID'] = classes['sample_heard'].str.slice(1,4).astype('int')
classes.head()    

# random partition for train and test
# (stratified taking into account speaker gender)

indexes = np.arange(0,len(classes))
train_i, test_i, train_y, test_y = train_test_split(indexes, 
                                                    indexes, # dummy classes
                                                    test_size=0.25, 
                                                    stratify = classes['gender'], 
                                                    random_state=2302)

classes_train = classes.iloc[train_i,:] # 225 instances
classes_test = classes.iloc[test_i,:] # 75 instances

# save these data for other evaluations
classes_train.to_csv(r'..\data\generated_data\classes_train.csv', index=False)
classes_test.to_csv(r'..\data\generated_data\classes_test.csv', index=False)

# partitions of ratings and scores

ratings = ratings.rename(index=str, columns={'speaker_ID': 'spkID'})
ratings_train = ratings[ratings['spkID'].isin(classes_train['spkID'])] # shape (225, 36)
ratings_test = ratings[ratings['spkID'].isin(classes_test['spkID'])] # shape (75, 36)

ratings_scores_train = ratings_train.merge(scores.drop(['sample_heard','gender'],axis=1)) # shape (225, 41)
ratings_scores_test = ratings_test.merge(scores.drop(['sample_heard','gender'],axis=1)) # shape (75, 41)

path3 = "https://raw.githubusercontent.com/laufergall/ML_Speaker_Characteristics/master/data/extracted_features/"

url = path3 + "/eGeMAPSv01a_semispontaneous_splitted.csv"
s = requests.get(url).content
feats =pd.read_csv(io.StringIO(s.decode('utf-8')), sep = ';') # shape: 3591, 89

# extract speaker ID from speech file name
feats['spkID'] = feats['name'].str.slice(2, 5).astype('int')

# appending multilabels to features
feats_ratings_scores_train = pd.merge(feats, ratings_scores_train) # shape (2700, 130)
feats_ratings_scores_test = pd.merge(feats, ratings_scores_test) # shape (891, 130)

# 'name' + 'speaker_gender' + 'spkID' + 88 features + 5 trait scores + 34 item ratings

list(feats_ratings_scores_test)

feats_ratings_scores_train.to_csv(r'..\data\generated_data\feats_ratings_scores_train.csv', index=False)
feats_ratings_scores_test.to_csv(r'..\data\generated_data\feats_ratings_scores_test.csv', index=False)

# save feature names
dropcolumns = ['name','spkID','speaker_gender'] + items_names + traits_names
feats_names = list(feats_ratings_scores_train.drop(dropcolumns, axis=1))

myfile = open(r'..\data\generated_data\feats_names.txt', 'w')
for item in feats_names:
    myfile.write("%r\n" % item)
    
# save names of speaker items and traits
    
myfile = open(r'..\data\generated_data\items_names.txt', 'w')
for item in items_names:
    myfile.write("%r\n" % item)
    
myfile = open(r'..\data\generated_data\traits_names.txt', 'w')
for item in traits_names:
    myfile.write("%r\n" % item)

