# Load Libraries - Make sure to run this cell!
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn import feature_extraction, tree, model_selection, metrics
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')

## Load data
df = pd.read_csv('../../data/dga_data_small.csv')
df.drop(['host', 'subclass'], axis=1, inplace=True)
print(df.shape)
df.sample(n=5).head() # print a random sample of the DataFrame

df[df.isDGA == 'legit'].head()

# Google's 10000 most common english words will be needed to derive a feature called ngrams...
# therefore we already load them here.
top_en_words = pd.read_csv('../../data/google-10000-english.txt', header=None, names=['words'])
top_en_words.sample(n=5).head()
# Source: https://github.com/first20hours/google-10000-english

def H_entropy (x):
    # Calculate Shannon Entropy
    prob = [ float(x.count(c)) / len(x) for c in dict.fromkeys(list(x)) ] 
    H = - sum([ p * np.log2(p) for p in prob ]) 
    return H

def vowel_consonant_ratio (x):
    # Calculate vowel to consonant ratio
    x = x.lower()
    vowels_pattern = re.compile('([aeiou])')
    consonants_pattern = re.compile('([b-df-hj-np-tv-z])')
    vowels = re.findall(vowels_pattern, x)
    consonants = re.findall(consonants_pattern, x)
    try:
        ratio = len(vowels) / len(consonants)
    except: # catch zero devision exception 
        ratio = 0  
    return ratio

# derive features
df['length'] = df.domain.str.len()
df['digits'] = df.domain.str.count('[0-9]')
df['entropy'] = df.domain.apply(H_entropy)
df['vowel-cons'] = df.domain.apply(vowel_consonant_ratio)

# encode strings of target variable as integers
df.isDGA = df.isDGA.replace(to_replace = 'dga', value=1)
df.isDGA = df.isDGA.replace(to_replace = 'legit', value=0)
print(df.isDGA.value_counts())

# check intermediate 2D pandas DataFrame
df.sample(n=5).head()

# ngrams: Implementation according to Schiavoni 2014: "Phoenix: DGA-based Botnet Tracking and Intelligence"
# http://s2lab.isg.rhul.ac.uk/papers/files/dimva2014.pdf

def ngrams(word, n):
    # Extract all ngrams and return a regular Python list
    # Input word: can be a simple string or a list of strings
    # Input n: Can be one integer or a list of integers 
    # if you want to extract multipe ngrams and have them all in one list
    
    l_ngrams = []
    if isinstance(word, list):
        for w in word:
            if isinstance(n, list):
                for curr_n in n:
                    ngrams = [w[i:i+curr_n] for i in range(0,len(w)-curr_n+1)]
                    l_ngrams.extend(ngrams)
            else:
                ngrams = [w[i:i+n] for i in range(0,len(w)-n+1)]
                l_ngrams.extend(ngrams)
    else:
        if isinstance(n, list):
            for curr_n in n:
                ngrams = [word[i:i+curr_n] for i in range(0,len(word)-curr_n+1)]
                l_ngrams.extend(ngrams)
        else:
            ngrams = [word[i:i+n] for i in range(0,len(word)-n+1)]
            l_ngrams.extend(ngrams)
#     print(l_ngrams)
    return l_ngrams

def ngram_feature(domain, d, n):
    # Input is your domain string or list of domain strings
    # a dictionary object d that contains the count for most common english words
    # finally you n either as int list or simple int defining the ngram length
    
    # Core magic: Looks up domain ngrams in english dictionary ngrams and sums up the 
    # respective english dictionary counts for the respective domain ngram
    # sum is normalized
    
    l_ngrams = ngrams(domain, n)
#     print(l_ngrams)
    count_sum=0
    for ngram in l_ngrams:
        if d[ngram]:
            count_sum+=d[ngram]
    try:
        feature = count_sum/(len(domain)-n+1)
    except:
        feature = 0
    return feature
    
def average_ngram_feature(l_ngram_feature):
    # input is a list of calls to ngram_feature(domain, d, n)
    # usually you would use various n values, like 1,2,3...
    return sum(l_ngram_feature)/len(l_ngram_feature)


l_en_ngrams = ngrams(list(top_en_words['words']), [1,2,3])
d = Counter(l_en_ngrams)

from six.moves import cPickle as pickle
with open('../../data/d_common_en_words' + '.pickle', 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

df['ngrams'] = df.domain.apply(lambda x: average_ngram_feature([ngram_feature(x, d, 1), 
                                                                ngram_feature(x, d, 2), 
                                                                ngram_feature(x, d, 3)]))

# check final 2D pandas DataFrame containing all final features and the target vector isDGA
df.sample(n=5).head()

df_final = df
df_final = df_final.drop(['domain'], axis=1)
df_final.to_csv('../../data/dga_features_final_df.csv', index=False)
df_final.head()

df_final = pd.read_csv('../../data/dga_features_final_df.csv')
print(df_final.isDGA.value_counts())
df_final.head()



