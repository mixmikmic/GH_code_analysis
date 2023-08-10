import pandas as pd
get_ipython().magic('matplotlib inline')

df = pd.read_pickle('CleanedData_2.p')

for k,v in df.iterrows():
    if(v['thumbnail'] != 'self' and v['thumbnail'] != 'nsfw' and v['thumbnail'] != "default"):
        df.loc[k,'thumbnail'] = "link"
        
        

df.to_pickle('CleanedData_2.p')

df

df['thumbnail'].value_counts()

del df['downs']
del df['author_name']

df['Category'].unique()

import sklearn

temp2 = df
temp1 = pd.get_dummies(df['Category'])

pd.concat([temp1, temp2],axis=1)

temp1.columns = ["dd" + "_" + bla for bla in temp1.columns]

temp = df.copy(deep = True)

def convCategorical(df,col):
    if col in df.keys():
        temp1 = pd.get_dummies(df[col])
        temp1.columns = [col + "_" + str(bla) for bla in temp1.columns]
        df = pd.concat( [df,temp1] ,axis = 1)
        del df[col]
    
lis = ['Category','author_is_gold','thumbnail','type']

for col in lis:
    convCategorical(temp,col)

temp

col = 'thumbnail'

temp1 = pd.get_dummies(df[col])
temp1.columns = [col + "_" + str(bla) for bla in temp1.columns]
df = pd.concat([temp1, df],axis=1)

df.keys()

del df['permalink']
del df['thumbnail']
del df['author_is_gold']
del df['Category']

df

df.keys()

df.to_pickle('CleanedData_3.p')

df = pd.read_pickle('CleanedData_2.p')

df

attr = ['created_utc','title','ups','author_link_karma','author_comment_karma']

df.columns

tempdf = df.copy(deep = True)


for col in attr:
    if col != 'title':
        avg = tempdf[col].mean()
        diff = tempdf[col].max() - tempdf[col].min()
        tempdf[col] = tempdf[col].apply(lambda x: (x-avg)/diff) 

tempdf = tempdf[tempdf['author_link_karma'] < 8000000]

tempdf['author_link_karma'].hist(bins = 50)

tempdf['author_link_karma'].mean()

tempdf[attr]

df.to_pickle('Numerical_normalized.p')

df = pd.read_pickle('Numerical_normalized.p')

tempmin = df['ups'].min()
df['ups'] = df['ups'].apply(lambda x: x - tempmin)

tempmin = df['created_utc'].min()
df['created_utc'] = df['created_utc'].apply(lambda x: x - tempmin)
tempmin = df['author_link_karma'].min()
df['author_link_karma'] = df['author_link_karma'].apply(lambda x: x - tempmin)
tempmin = df['author_comment_karma'].min()
df['author_comment_karma'] = df['author_comment_karma'].apply(lambda x: x - tempmin)

df[attr]

df[attr]

import string
import re
def removepunctuation(x):
    x = x.replace('.','')
    x = x.replace(')','')
    x = x.replace('(','')
    x = x.replace('\'','')
    x = x.replace('-','')
    x = x.replace('?','')
    x = x.replace('$','')
    x = x.replace('!','')
    x = x.replace(':','')
    x = x.replace(',','')
    x = x.replace('%','')
    x = x.replace('+','')
    x = x.replace('*','')
    x = x.replace('/','')
    x = x.replace('&','')
    x = x.replace('@','')
    #replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    #x = str(x).translate(replace_punctuation)
    #retstr = x.translate(string.maketrans("",""), string.punctuation)
    return x
    
def removeunicode(x):
    return re.sub(r'[^\x00-\x7F]+',' ', x)
def lowercasestring(x):
    return x.lower()

def removedigits(s):
    s = re.sub(" \d+", " ", s)
    return s
    
def cleanstring(x):
    #x=replaceredundancy(x)
    x=removepunctuation(x)
    x=removeunicode(x)
    #x = trimstring(x)
    x=removedigits(x)
    x=lowercasestring(x)
    return x 

df['title_clean'] = df['title'].apply(cleanstring)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import string
import re
import nltk
import enchant
import sklearn

get_ipython().magic('matplotlib inline')

d = enchant.Dict("en_US")
def removenonsensewords(text):
    tokens = nltk.word_tokenize(text)
    
    stemmed = []
    #i=0
    for token in tokens:
        #print(i)
        #i=i+1
        if d.check(token):
            stemmed.append(token)
        
    return ' '.join(stemmed)

df['title_nonsense']= df['title_clean'].apply(removenonsensewords)

def listofbadwords():
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    monthnames = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    randomrepetitive = ['edu','unl','mt']
    
    totlist = stopwords + monthnames + randomrepetitive
    return totlist

totlist = listofbadwords()
def removebadwords(x):
    
    wordlist = x.split()
    wordlist = [word for word in wordlist if word.lower() not in totlist]
    x = ' '.join(wordlist)
    return x
df['title_badwords'] = df['title_nonsense'].apply(removebadwords)

def replacewithstem(text):
    tokens = nltk.word_tokenize(text)
    stemmer = nltk.stem.porter.PorterStemmer()
    
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
        
    return ' '.join(stemmed)

df['title_stemmed']= df['title_badwords'].apply(replacewithstem)


df['title_stemmed']

df.to_pickle('PostStemming.p')

df = pd.read_pickle('PostStemming.p')

vect.fit(df['title_stemmed'])

from sklearn.feature_extraction.text import HashingVectorizer

vect = HashingVectorizer(norm = None)#,n_features=1000)
df_title_text = vect.fit_transform(df['title_stemmed'])

import pickle

save_object(vect, "hashingVect.pkl")

df_title_text

from sklearn import svm

svmclass = svm.SVR()
svmclass.fit(df_title_text,df['ups'])  

df['predicted_ups'] = svmclass.predict(df_title_text)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    pkl_file = open(filename, 'rb')
    mydict2 = pickle.load(pkl_file)
    pkl_file.close()
    return mydict2

import pickle

save_object(svmclass, 'class1.pkl')

df.to_pickle('PostClass1.p')

df.columns

final_attr = [ u'thumbnail_default',        u'thumbnail_link',
              u'thumbnail_nsfw',        u'thumbnail_self',
                    u'type_img',              u'type_txt',
                    u'type_vid',  u'author_is_gold_False',
         u'author_is_gold_True',      u'Category_Android',
                u'Category_Art',    u'Category_AskReddit',
            u'Category_Bitcoin', u'Category_Conservative',
                u'Category_DIY',        u'Category_Jokes',
                u'Category_MMA',        u'Category_Music',
                u'Category_WTF',        u'Category_apple',
         u'Category_askscience',       u'Category_bestof',
               u'Category_cats',       u'Category_comics',
             u'Category_creepy',        u'Category_drunk',
           u'Category_facepalm',         u'Category_food',
              u'Category_funny',         u'Category_guns',
          u'Category_lifehacks',       u'Category_movies',
                u'Category_nba',         u'Category_news',
           u'Category_politics',  u'Category_programming',
               u'Category_rage',      u'Category_science',
                u'Category_sex',       u'Category_soccer',
              u'Category_space',   u'Category_technology',
          u'Category_teenagers',        u'Category_trees',
             u'Category_trippy',       u'Category_videos',
          u'Category_worldnews',  
        u'author_comment_karma',     u'author_link_karma',
                 u'created_utc',          u'num_comments',
               u'predicted_ups']

svmclass2 = svm.SVR()
svmclass2.fit(df[final_attr],df['ups'])  

svmclass2

save_object(svmclass2, 'class2.pkl')

df['final_prediction'] = svmclass2.predict(df[final_attr])

df[['ups','final_prediction']]

df.column

df.to_pickle('PostClass2.p')

df = pd.read_pickle('PostClass2.p')

df.loc[0,:]

df.columns



from sklearn.metrics import mean_squared_error
mean_squared_error(df['predicted_ups'],df['ups'])

from sklearn.cross_validation import train_test_split

df = pd.read_pickle('PostClass2.p')

df.columns

df['ups'].mean()


X_train, X_test, y_train, y_test = train_test_split(df, df['ups'], test_size=0.2, random_state=0)

svmclass3 = svm.SVR()
svmclass3.fit(X_train[final_attr],y_train)  

mean_squared_error(svmclass3.predict(X_test[final_attr]),y_test)

df['ups'].max()

df.columns

df[df['ups'] == 1]

df



