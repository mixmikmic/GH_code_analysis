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

df = pd.read_pickle('CleanedData_3.p')

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

from sklearn.feature_extraction.text import HashingVectorizer

vect = HashingVectorizer(norm = None)#,n_features=1000)
df_title_text = vect.fit_transform(df['title_stemmed'])

df_title_text

from sklearn import svm

svmclass = svm.SVR()
svmclass.fit(df_title_text,df['ups'])  

df['predicted_ups'] = svmclass.predict(df_title_text)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_object(svmclass, '')

ch2 = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k = 1500)
df_title_text_reduced = ch2.fit_transform(df_title_text, df['ups'])



