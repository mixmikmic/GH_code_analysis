import pandas as pd
from collections import Counter
import re
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

df_all = pd.read_csv("nyt.csv",index_col=False,na_filter=False)

print(df_all.head(30))

# Number of articles in each section
Counter(df_all.section_name)

# Getting only wanted sections
df_all = df_all[(df_all.section_name == 'Americas') |                (df_all.section_name == 'Economy')  |                (df_all.section_name == 'Politics') |                (df_all.section_name == 'Economy') |                (df_all.section_name == 'Campaign Stops') |                (df_all.section_name == 'Stocks and Bonds') |                (df_all.section_name == 'Small Business') |                (df_all.section_name == 'Middle East') |                (df_all.section_name == 'Europe') |                (df_all.section_name == '')] # can't afford to discard this, not sure why missing

df_all.shape

for idx, item in enumerate(df_all.body):
    df_all.body[idx] = re.sub('[^\x00-\x7F]+', "", item)
    if idx%500 == 0:
        print 'Here is the ',idx,'th item'

df_all.info()

df_all = df_all.reset_index()

import json
print(df_all.headline[29])
print(json.loads(df_all.headline[29])['main'])

# Getting only needed main headline
lst_head = []
for idx,item in enumerate(df_all.headline): 
    if not df_all.headline[idx] == "":
        obj = json.loads(df_all.headline[idx])
        lst_head.append(obj['main'].replace("'\"",""))

df_all = df_all.drop(["headline"],axis=1)

df_all['head_clean'] = lst_head

df_all.dropna(inplace=True)

df_all.info()
df_all.head(10)

# Cleaning the data
for idx, item in enumerate(df_all.body):
    df_all.body[idx] = re.sub('(\\n)',"",item)
    if idx%500 == 0:
        print 'Here is the ',idx,'th item'

df_all.head()

# Cleaning the data
for idx, item in enumerate(df_all.head_clean):
    df_all.head_clean[idx] = re.sub('(\\n)',"",item)
    if idx%500 == 0:
        print 'Here is the ',idx,'th item'

#Cleaning the data

slash_text = []
slash_title = []
for idx, item in enumerate(df_all.body):
    
    try:
        df_all.body[idx] = re.sub('(\\n)',"",item)
    except:
        df_all.body[idx] = 'Dummy_Text'
        wrong_text.append((idx,item))
    
    if idx%500 == 0:
        print 'Here is the ',idx,'th item'
        
        
for idx, item in enumerate(df_all.head_clean):
    
    try:
        df_all.head_clean[idx] = re.sub('(\\n)',"",item)

    except:
        df_all.head_clean[idx] = 'Dummy_Title'
        wrong_title.append((idx,item))
    if idx%500 == 0:
        print 'Here is the ',idx,'th item'

df_all.head()

df_all_ren = df_all.rename(columns={'body': 'text', 'pub_date': 'date', 'head_clean': 'title'})
df_all_ren['class'] = np.ones(len(df_all_ren), dtype=int)
df_all_ren = df_all_ren.drop('index', 1)
df_all_ren.head()

# Saving the clean data to csv file
df_all_ren.to_csv("nyt_unclean.csv", encoding='utf-8', index=False)

tdf = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
vectorizer = tdf.fit(df_all.body)
transformed_text = vectorizer.transform(df_all.body)
transformed_title = vectorizer.transform(df_all.head_clean)

a = transformed_text.todense()

def get_tfidf_term_scores(feature_names):
    '''Returns dictionary with term names and total tfidf scores for all terms in corpus'''
    term_corpus_dict = {}
    # iterate through term index and term 
    for term_ind, term in enumerate(feature_names):
        term_name = feature_names[term_ind]
        term_corpus_dict[term_name] = np.sum(transformed_title.T[term_ind].toarray())
        
    return term_corpus_dict

# list of features created by tfidf
feature_names = tdf.get_feature_names()

term_corpus_dict = get_tfidf_term_scores(feature_names)

print "Number of columns is: ",len(term_corpus_dict.keys())

def get_sorted_tfidf_scores(term_corpus_dict):
    '''Returns sort words from highest score to lowest score'''
    # sort indices from words wit highest score to lowest score
    sortedIndices = np.argsort( list(term_corpus_dict.values()))[::-1]
    # move words and score out of dicts and into arrays
    termNames = np.array(list(term_corpus_dict.keys()))
    scores = np.array(list(term_corpus_dict.values()))
    # sort words and scores
    termNames = termNames[sortedIndices]
    scores = scores[sortedIndices]
    
    return termNames, scores

termNames, scores = get_sorted_tfidf_scores(term_corpus_dict)

def getSelectScores(selectTerms):
    '''Returns a list of tfidf scores for select terms that are passed in'''
    score = [ term_corpus_dict[select_term]  for select_term in selectTerms]
    return score

selectTerms = ['trump', 'clinton','islamic', 'russia' , 'women', 'obama', 'men',
               'students', 'shooting', 'democrats', 'republicans', 'climate',
               'education', 'environment', 'tech', 'minorities', 'carbon',
               'muslim','ban']

selectScores = getSelectScores(selectTerms)

def plot_tfidf_scores(scores,termNames, selectScores, selectTerms,  n_words = 18):
    '''Returns one plot for Importance of Top N Terms
       and one plot for Importance of Select K Terms'''

    # Create a figure instance, and the two subplots
    fig = plt.figure(figsize = (14, 18))
    
    override = {'fontsize': 'large'}

    fig.add_subplot(221)   #top left
    n_words = 75
    sb.barplot(x = scores[:n_words], y = termNames[:n_words]);
    sb.plt.title("TFIDF - Importance of Top {0} Terms".format(n_words));
    sb.plt.xlabel("TFIDF Score");

    fig.add_subplot(222)   #top right 
    sb.barplot(x = selectScores, y = selectTerms);
    sb.plt.title("TFIDF - Importance of Select {0} Terms".format(len(selectTerms)));
    sb.plt.xlabel("TFIDF Score");
    sb.plt.ylabel(override)

plot_tfidf_scores(scores, termNames, selectScores, selectTerms,  n_words = 18)

