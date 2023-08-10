import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import datetime
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 6)

csv = pd.read_csv('data/robotics.csv.zip', index_col='id')
print(csv.shape)
csv.head()

def uniq_tags(df):
    tags = set()
    df['tags'].str.split(' ').apply(tags.update)
    return tags

def words_frequencies(df, column):
    st = LancasterStemmer() # reduce plutal, gerunts and so on...
    freq = {}
    for i, row in df.iterrows():
        words = row[column].split(' ')
        for word in words:
            word = st.stem(word)
            freq[word] = freq.get(word, 0) + 1

    return freq

def prepare_text(text):
    cleantext = re.sub("<.*?>", "", text).lower().replace('\n', '. ')
    splitter = re.compile("[^a-zA-Z0-9_\\+\\-/]")
    words = splitter.split(cleantext)
    stops = set(stopwords.words("english"))
    meaningful_words = [w.strip() for w in words if not w in stops]
    return " ".join(filter(None, meaningful_words))

def remove_html(text):
    # return re.sub(r'\s+', ' ', re.sub("<.*?>", "", text)).lower().strip()
    return re.sub("[0-9]+", "", re.sub(r'\s+', ' ', BeautifulSoup(text, "html.parser").get_text()).lower().strip())

def prepare_data(df, func):
    res = pd.DataFrame(index=df.index)
    res['tags'] = df['tags']
    res['title'] = df['title'].apply(func)
    res['content'] = df['content'].apply(func)
    return res

def prepare_test_data(df, func):
    res = pd.DataFrame(index=df.index)
    res['title'] = df['title'].apply(func)
    res['content'] = df['content'].apply(func)
    return res

data = prepare_data(csv, remove_html)
data.head()

tag_freq = words_frequencies(data, 'tags')
content_freq = words_frequencies(data, 'content')
title_freq = words_frequencies(data, 'title')

tags = pd.DataFrame(list(tag_freq.values()), columns=['count'], index=tag_freq.keys())
tags.sort_values(['count'], ascending=False).head(10)

tags.describe()

tags.sort_values(['count'], ascending=False).plot()

tags['count'].loc[lambda c: c > 100].count()

titles = pd.DataFrame(list(title_freq.values()), columns=['count'], index=title_freq.keys())
titles.sort_values(['count'], ascending=False).head(10)

titles.describe()

contents = pd.DataFrame(list(content_freq.values()), columns=['count'], index=content_freq.keys())
contents.sort_values(['count'], ascending=False).head(10)

contents.describe()

intersections_with_title=set(titles.index) & set(tags.index)
intersections_with_content=set(contents.index) & set(tags.index)
intersection_with_all = (set(contents.index) | set(titles.index)) & set(tags.index)
print("Tags {}".format(len(tags.index)))
print("Titles {}".format(len(intersections_with_title)))
print("Contents {}".format(len(intersections_with_content)))
print("All {}".format(len(intersection_with_all)))

# Set of tags that don't appear in niether in title or content 
set(tags.index) - (set(contents.index) | set(titles.index))

for i, row in data.iterrows():
    tags = set(row['tags'].split(' '))
    words = set(row['content'].split(' ')) | set(row['title'].split(' '))
    intersection = tags & words
    data.ix[i, 'intersection'] = len(intersection) / len(tags)
data.head()

data.describe()

import string, itertools
from nltk import word_tokenize, sent_tokenize, pos_tag_sents, tree2conlltags
from  nltk.chunk.regexp import RegexpParser
import itertools, nltk, string

grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
# grammar = r"""
#     NBAR:
#         {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        
#     NP:
#         {<NBAR>}
#         {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
# """
stop_words = stopwords.words('english')
punct = string.punctuation
chunker = RegexpParser(grammar)
text = data.ix[2,'content']
text

tagged_sents  = pos_tag_sents([word_tokenize(sent) for sent in sent_tokenize(text)])
tuples = [tree2conlltags(chunker.parse(tagged_sent)) for tagged_sent in tagged_sents]
all_chunks = list(itertools.chain.from_iterable(tuples)) # they said it's optimization for loops
all_chunks

import nltk
lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    #word = stemmer.stem_word(word)
    word = lemmatizer.lemmatize(word)
    return word

def group_func(t):
        (word,pos,chunk) = t
        return chunk != 'O'
    
punct = set(string.punctuation)
stop_words = set(nltk.corpus.stopwords.words('english'))
grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
chunker = nltk.chunk.regexp.RegexpParser(grammar)

def extract_candidate_chunks(text):
    # exclude candidates that are stop words or entirely punctuation
    # tokenize, POS-tag, and chunk using regular expressions
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = [nltk.chunk.tree2conlltags(chunker.parse(tagged_sent)) 
                  for tagged_sent in tagged_sents]
    all_chunks = list(itertools.chain.from_iterable(all_chunks))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower()
                  for key, group in itertools.groupby(all_chunks, group_func) if key]

    return set([cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand) and len(cand) > 2])

extract_candidate_chunks(text)

def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word) and len(tag) > 2]

    return candidates

extract_candidate_words(text)

def score_keyphrases_by_textrank(text, n_keywords=0.05):
    from itertools import takewhile, tee
    import networkx, nltk
    
    # tokenize for all words, and extract *candidate* words
    words = [word.lower()
             for sent in nltk.sent_tokenize(text)
             for word in nltk.word_tokenize(sent)]
    candidates = extract_candidate_chunks(text)
    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)
    
    return [tag for tag, score in sorted(keyphrases.items(), key=lambda x: x[1], reverse=True)]

score_keyphrases_by_textrank(text, 0.5)

for i, row in data.iterrows():
    tags = score_keyphrases_by_textrank("{} {}".format(data.ix[i, 'title'], data.ix[i, 'content']), 0.4)
    data.ix[i, 'predictions'] = ' '.join(tags)
data.head()

test = pd.read_csv('data/test.csv', index_col='id')
test.head()

test_data = prepare_test_data(test, remove_html)
test_data.head()

test.shape

remove_html(test.ix[29,'content'])

test_data.ix[29,'content']

soup = BeautifulSoup(test.ix[31,'content'], 'html.parser')
soup.get_text()

