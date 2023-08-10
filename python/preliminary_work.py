import os
import numpy as np
import pandas as pd
import seaborn.apionly as sns
import matplotlib.pyplot as plt
import xgboost
from xgboost import plot_tree
from xgboost import plot_importance
import matplotlib.pyplot as plt
from graphviz import Digraph
# load data
import pydot
plt.style.use('ggplot')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
get_ipython().magic('matplotlib inline')
colors = sns.color_palette()

from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer
import re
from string import punctuation

stops = set(stopwords.words("english"))

def text_to_wordlist(text, remove_stopwords=True, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = str(text)
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i m", "i am", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'m", " am ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r"usa", "America", text)
    text = re.sub(r"canada", "Canada", text)
    text = re.sub(r"japan", "Japan", text)
    text = re.sub(r"germany", "Germany", text)
    text = re.sub(r"burma", "Burma", text)
    text = re.sub(r"rohingya", "Rohingya", text)
    text = re.sub(r"zealand", "Zealand", text)
    text = re.sub(r"cambodia", "Cambodia", text)
    text = re.sub(r"zealand", "Zealand", text)
    text = re.sub(r"norway", "Norway", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"pakistan", "Pakistan", text)
    text = re.sub(r"britain", "Britain", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iphone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"iii", "3", text)
    text = re.sub(r"california", "California", text)
    text = re.sub(r"texas", "Texas", text)
    text = re.sub(r"tennessee", "Tennessee", text)
    text = re.sub(r"the us", "America", text)
    text = re.sub(r"trump", "Trump", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, shorten words to their stems
    if stem_words:  
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)

def check_word_match(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = float((len(shared_words_in_q1) + len(shared_words_in_q2)))/float((len(q1words) + len(q2words)))
    return R

def word_shares(row):
    q1 = set(str(row['question1']).lower().split())
    q2 = set(str(row['question2']).lower().split())
    q1words = q1.difference(stops)
    q2words = q2.difference(stops)
    
    if len(q1words) == 0 or len(q2words) == 0:
            return '0:0:0:0'
    
    q1stops = q1.intersection(stops)
    q2stops = q2.intersection(stops)
    shared_words = q1words.intersection(q2words)
    
    shared_w_ratio = len(shared_words) / (len(q1words) + len(q2words))
    stops_q1_ratio = len(q1stops) / len(q1words)
    stops_q2_ratio= len(q2stops) / len(q2words)
    return '{}:{}:{}:{}'.format(shared_w_ratio, len(shared_words), stops_q1_ratio, stops_q2_ratio)


def similarity(row):
    q1 = set(str(row['question1']).lower().split())
    q2 = set(str(row['question2']).lower().split())
    x, y, xy = len(q1), len(q2), len(q1.intersection(q2))
    
    # Dice
    if (x+y)==0:
        dice = 0
    else:
        dice = 2*xy / (x + y)
   
    # Jaccard
    if len(q1.union(q2)) ==0:
        jaccard = 0
    else:
        jaccard = xy / len(q1.union(q2))
    
    # Simpson, Cosine
    if min(x, y)==0:
        simpson = 0
        cosine = 0
    else:
        simpson = xy / min(x, y)
        cosine = xy / np.sqrt(x*y)
    
    # 信頼度
    if x==0:
        confidence1 = 0
    else:
        confidence1= xy / x
    if y==0:
        confidence2 = 0
    else:
        confidence2= xy / y
        
    return '{}:{}:{}:{}:{}:{}:{}:{}:{}'.format(x, y, xy, dice, jaccard, simpson, cosine, confidence1, confidence2)

quora_train = pd.read_csv('/home/nacim/DATASET_KAGGLE/quora/train.csv')

print('Total number of question pairs for training: {}'.format(len(quora_train)))
print('Duplicate pairs: {}%'.format(round(quora_train['is_duplicate'].mean()*100, 2)))
qids = pd.Series(quora_train['qid1'].tolist() + quora_train['qid2'].tolist())
print('Total number of questions in the training data: {}'.format(len(
    np.unique(qids))))
print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

plt.figure()
plt.hist(qids.value_counts(), range=[0, 80],bins=100)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
plt.savefig('figures/log_histogram.png',bbox_inches='tight')

overallquestions = pd.Series(quora_train['question1'].tolist() + quora_train['question2'].tolist()).astype(str)

histo_char = overallquestions.apply(len)
print(r"mean train = {0} +/- std strain = {1}, median={2} ".format(histo_char.mean(),
                                                                   histo_char.std(),
                                                                   histo_char.median()))
print(r"min train = {0},  max strain = {1} ".format(histo_char.min(),histo_char.max()))

plt.figure()
plt.hist(histo_char, bins=100, range=[0, 250], normed=True,color='green')
plt.title('Normalised histogram of character count in questions', fontsize=13)
plt.xlabel('Number of characters', fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.tight_layout()
plt.savefig('character_counts.',bbox_inches='tight')

histo_words = overallquestions.apply(lambda x: len(x.split(' ')))
print(r"mean train = {0} +/- std strain = {1}, median={2} ".format(histo_words.mean(),
                                                                   histo_words.std(),
                                                                   histo_words.median()))
print(r"min train = {0},  max strain = {1} ".format(histo_words.min(),histo_words.max()))
plt.hist(histo_words, bins=100, range=[0, 50], color='blue', normed=True)
plt.title('Normalised histogram of word count in questions', fontsize=12)
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.savefig('word_counts.pdf',bbox_inches='tight')

qmarks = overallquestions.apply(lambda x: '?' in x)
fullstop = overallquestions.apply(lambda x: '.' in x)
capital_first = overallquestions.apply(lambda x: x[0].isupper())
capitals = overallquestions.apply(lambda x: max([y.isupper() for y in x]))
lower = overallquestions.apply(lambda x: max([y.islower() for y in x]))
numbers = overallquestions.apply(lambda x: max([y.isdigit() for y in x]))
print('Summary Statistics')
print('Qustions marks: mean={0}% +/- {1}%;\t median={2}%'.format(np.mean(qmarks)*100,
                                                              np.std(qmarks)*100,
                                                              np.median(qmarks)*100))

print('Full Stops: mean={0}% +/- {1}%;\t median={2}%'.format(np.mean(fullstop)*100,
                                                              np.std(fullstop)*100,
                                                              np.median(fullstop)*100))

print('With Numbers: mean={0}% +/- {1}%;\t median={2}%'.format(np.mean(numbers)*100,
                                                              np.std(numbers)*100,
                                                              np.median(numbers)*100))

print('Capital First: mean={0}% +/- {1}%;\t median={2}%'.format(np.mean(capital_first)*100,
                                                              np.std(capital_first)*100,
                                                              np.median(capital_first)*100))

print('With Capital ase Letters: mean={0}% +/- {1}%;\t median={2}%'.format(np.mean(capitals)*100,
                                                              np.std(capitals)*100,
                                                              np.median(capitals)*100))

print('With Lower Case Letters: mean={0}% +/- {1}%;\t median={2}%'.format(np.mean(lower)*100,
                                                              np.std(lower)*100,
                                                              np.median(lower)*100))



train_word_match = quora_train.apply(check_word_match, axis=1, raw=True)

fig = plt.figure(figsize=(10,6))
ax = fig.gca()
for dup_value,label in zip([0,1],['Not Duplicate','Duplicate']):
    x = train_word_match[quora_train['is_duplicate'] == dup_value]
    sns.distplot(x,label=label,kde=True,norm_hist=False,kde_kws={"lw": 3})
    plt.xlim([0,1])
plt.title('Label distribution over check_word_match', fontsize=15)
plt.xlabel('Probability', fontsize=15)
plt.legend()
plt.savefig('shared_words.pdf',bbox_inches='tight')

def clean_questionpairs(row):
    row['question1'] = text_to_wordlist(row['question1'])
    row['question2'] = text_to_wordlist(row['question2'])
    return row

if os.path.exists('data/train_clean.csv'):
    quora_train_clean = pd.read_csv('data/train_clean.csv')
else:
    quora_train_clean = quora_train.copy()
    quora_train_clean = quora_train_clean.apply(clean_questionpairs,axis=1)
    quora_train_clean.to_csv('data/train_clean.csv',index=False)

train_word_match_clean = quora_train_clean.apply(check_word_match, axis=1, raw=True)

fig = plt.figure(figsize=(10,6))
ax = fig.gca()
for dup_value,label in zip([0,1],['Not Duplicate','Duplicate']):
    x = train_word_match_clean[quora_train_clean['is_duplicate'] == dup_value]
    sns.distplot(x,label=label,kde=True,norm_hist=False,kde_kws={"lw": 3})
    plt.xlim([0,1])
plt.title('Label distribution over check_word_match from cleaned Dataset', fontsize=15)
plt.xlabel('Probability', fontsize=15)
plt.legend()
plt.savefig('shared_words_cleaned.pdf',bbox_inches='tight')

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().replace("-","").split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().replace("-","").split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/float((len(q1words) + len(q2words)))
    return R


#TF-IDF
from collections import Counter

train_qs = pd.Series(quora_train_clean['question1'].tolist() + quora_train_clean['question2'].tolist()).astype(str)

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / float((count + eps))

eps = 5000
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

print('Most common words and weights: \n')
print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])

def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = np.array([float(weights.get(w, 0)) for w in q1words.keys() if w in q2words])
    + np.array([float(weights.get(w, 0)) for w in q2words.keys() if w in q1words])
    total_weights = np.sum([float(weights.get(w, 0)) for w in q1words]) + np.sum([float(weights.get(w, 0))
                                                                                      for w in q2words])

    R = np.sum(shared_weights) / np.sum(total_weights)

    return R

data_values = pd.DataFrame()
data_values['TF-IDF'] = quora_train_clean.apply(tfidf_word_match_share, axis=1, raw=True)
data_values['common_words'] = quora_train_clean.apply(word_match_share, axis=1, raw=True)
data_values.to_csv('quora_features.csv', index=False)









