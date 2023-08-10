import os,sys,re
from bs4 import BeautifulSoup

# wherever you place your dataset
TEXT_DATA_DIR = '/home/ubuntu/working/text_classification/20_newsgroup/'

docs = []
doc_classes = []
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                f = open(fpath)
                t = f.read()
                # skip header
                i = t.find('\n\n')
                if 0 < i:
                    t = t[i:]
                t = BeautifulSoup(t).get_text()
                t = re.sub("[^a-zA-Z]"," ", t)
                docs.append(t)
                doc_classes.append(name)
                f.close()

# let's print the first ~1000 characters of the 1st document
print(docs[0][:1001])

from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

def SimpleTokenizer(doc):
    """Basic tokenizer using gensim's simple_preprocess

    Parameters:
    ----------
    docs (list): list of documents

    Returns:
    ----------
    tokenized documents
    """
    return [t for t in simple_preprocess(doc, min_len=3) if t not in STOPWORDS]


class StemTokenizer(object):
    """Stem tokens in a document

    Parameters:
    ----------
    docs (list): list of documents

    Returns:
    --------
    list of stemmed tokens
    """
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in SimpleTokenizer(doc)]


class LemmaTokenizer(object):
    """Lemmatize tokens in a document

    Parameters:
    ----------
    docs (list): list of documents

    Returns:
    --------
    list of lemmatized tokens
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.lemmatizer.lemmatize(t, pos="v") for t in SimpleTokenizer(doc)]

from sklearn.decomposition import LatentDirichletAllocation

# Lda model that will be used through all the experiments
NB_TOPICS = 10
lda_model = LatentDirichletAllocation(
    n_components=NB_TOPICS,
    learning_method='online',
    max_iter=10,
    batch_size=2000,
    verbose=1,
    max_doc_update_iter=100,
    n_jobs=-1,
    random_state=0)

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# 0-train/test split
MAX_NB_WORDS = 20000
train_docs, test_docs = train_test_split(docs, test_size=0.25, random_state=0)
vectorizer = CountVectorizer(min_df=10, max_df=0.5, max_features=MAX_NB_WORDS, tokenizer = LemmaTokenizer())

# 1-vectorize
tr_corpus = vectorizer.fit_transform(train_docs)
te_corpus = vectorizer.transform(test_docs)
n_words = len(vectorizer.vocabulary_)

# 2-train model
model = lda_model.fit(tr_corpus)

# 3-compute perplexity
gamma = model.transform(te_corpus)
perplexity = model.perplexity(te_corpus, gamma)/n_words

# 4-get vocabulary and return top N words. Let's 1st define a little helper
def get_topic_words(topic_model, feature_names, n_top_words):
    """Helper to get n_top_words per topic

    Parameters:
    ----------
    topic_model: LDA model
    feature_names: vocabulary
    n_top_words: number of top words to retrieve

    Returns:
    -------
    topics: list of tuples with topic index, the most probable words and the scores/probs
    """
    topics = []
    for topic_idx, topic in enumerate(topic_model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        tot_score = np.sum(topic)
        scores = [topic[i]/tot_score for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append([topic_idx, zip(topic_words, scores)])
    return topics


features = vectorizer.get_feature_names()
top_words = get_topic_words(model,features,10)

print("Perplexity of the LDA model with Lemmatization-preprocessing: {}".format(perplexity))
top_words

from gensim.models.phrases import Phraser, Phrases

class Bigram(object):
    """Bigrams to get phrases like artificial_intelligence

    Parameters:
    ----------
    docs (list): list of documents

    Returns:
    --------
    the document with bigrams appended at the end
    """
    def __init__(self):
        self.phraser = Phraser
    def __call__(self, docs):
        phrases = Phrases(docs,min_count=20)
        bigram = self.phraser(phrases)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    docs[idx].append(token)
        return docs

vectorizer = CountVectorizer(
    min_df=10, max_df=0.5,
    max_features=MAX_NB_WORDS,
    preprocessor = lambda x: x,
    tokenizer = lambda x: x)
tokenizer_ = LemmaTokenizer()
tokens = [tokenizer_(doc) for doc in docs]
phraser_ = Bigram()
ptokens = phraser_(tokens)

# 0-train/test split
train_docs, test_docs = train_test_split(ptokens, test_size=0.25, random_state=0)

# 1-vectorize
tr_corpus = vectorizer.fit_transform(train_docs)
te_corpus = vectorizer.transform(test_docs)
n_words = len(vectorizer.vocabulary_)

# 2-train model
model = lda_model.fit(tr_corpus)

# 3-compute perplexity
gamma = model.transform(te_corpus)
perplexity = model.perplexity(te_corpus, gamma)/n_words

# 4-get vocabulary and return top N words.
features = vectorizer.get_feature_names()
top_words = get_topic_words(model,features,10)

print(perplexity)
top_words

from nltk.corpus import words

class WordFilter(object):
    """Filter words based on a vocabulary

    Parameters:
    ----------
    vocab: the vocabulary used for filtering
    doc  : the document containing the tokens to be filtered

    Returns:
    -------
    filetered document
    """
    def __init__(self, vocab):
        self.filter = vocab
    def __call__(self, doc):
        return [t for t in doc if t in self.filter]


wordfilter = WordFilter(vocab=set(words.words()))
tokens = [SimpleTokenizer(doc) for doc in docs]
ftokens = [wordfilter(d) for d in tokens]

# 0-train/test split
train_docs, test_docs = train_test_split(ftokens, test_size=0.25, random_state=0)

# 1-vectorize
tr_corpus = vectorizer.fit_transform(train_docs)
te_corpus = vectorizer.transform(test_docs)
n_words = len(vectorizer.vocabulary_)

# 2-train model
model = lda_model.fit(tr_corpus)

# 3-compute perplexity
gamma = model.transform(te_corpus)
perplexity = model.perplexity(te_corpus, gamma)/n_words

# 4-get vocabulary and return top N words.
features = vectorizer.get_feature_names()
top_words = get_topic_words(model,features,10)

print(perplexity)
top_words

