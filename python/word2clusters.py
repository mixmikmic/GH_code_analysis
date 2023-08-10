import warnings, numpy as np, re, json, pandas as pd, pickle, unicodedata, textblob
# try:
#     import gnumpy as gpu
# except ModuleNotFoundError:
#     pass
from TurkishStemmer import TurkishStemmer
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim, math
from gensim.models import doc2vec
import  nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
# from KaggleWord2VecUtility import KaggleWord2VecUtility

df = pd.read_csv("datasets/movie_data.csv")
df.head(5)

en_vects = gensim.models.KeyedVectors.load_word2vec_format(r"GoogleNews-vectors-negative300.bin", binary=True)

tr_vects = gensim.models.KeyedVectors.load_word2vec_format(r"wiki.tr/wiki.tr.vec", binary=False)

tr_vocabs_ = dict()
en_vocabs_ = dict()

stemmer = TurkishStemmer()
def tokenize(text, vects='en_vects'):
    if vects == 'tr_vects':
        tr_words_inreview = list()
        for word in text.split(" "):
            w = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').lower().decode("ascii")
            w = stemmer.stem(w.lower())
            if w in globals()[vects] and len(w)>2:
                tr_vocabs_[w] = globals()[vects][w]
                tr_words_inreview.append(w)
        return tr_words_inreview
    en_words = list()
    for word in text.split(" "):
        w = word.lower()
        if w in globals()[vects] and len(w)>2:
            en_vocabs_[w] = globals()[vects][w]
            en_words.append(w)
    return en_words

def tok_(frame):
    res = list()
    for row in frame.iterrows():
        res.append(tokenize(row[1]["Review"],row[1]["Language"]+"_vects"))
    return res

df.groupby("Language", as_index=False).apply(tok_)

df[["tokenized_reviews"]].head(5)

# df.to_csv("datasets/tokenized_reviews.csv", index=False)

print ("Turkish Vocab: %d words" %len(tr_vocabs_))
print ("Enlish Vocab: %d words" %len(en_vocabs_))

def get_vocabs_vects_XY(vocabs_dict):
    X = list()
    y = list()
    try:
        for word in vocabs_dict.vocab:
            X.append(vocabs_dict[word])
            y.append(word)
    except:
        for word in vocabs_dict:
            X.append(vocabs_dict[word])
            y.append(word)
    return np.array(X),np.array(y)

# X_en, y_en = get_vocabs_vects_XY(en_vects)
# X_tr, y_tr = get_vocabs_vects_XY(tr_vects)
X_en, y_en = get_vocabs_vects_XY(en_vocabs_)
X_tr, y_tr = get_vocabs_vects_XY(tr_vocabs_)

from sklearn.cluster import k_means

en_clusters=k_means(X_en, n_clusters=1000, random_state=0)
tr_clusters=k_means(X_tr, n_clusters=300, random_state=0)

def word2cluster(vocab, clusters):
    # returns a dictionary of each word with its closest cluster
    word2cluster_dict = dict()
    centroids, labels = clusters[0], clusters[1]
    for word_index in range(len(vocab)):
        cluster_index = labels[word_index]
        word2cluster_dict[vocab[word_index]] = centroids[cluster_index]
    return word2cluster_dict

en_word2cluster = word2cluster(y_en, en_clusters)
tr_word2cluster = word2cluster(y_tr, tr_clusters)

pickle.dump(en_word2cluster, open("datasets/en_word2cluster.pickle", "wb"))
pickle.dump(tr_word2cluster,open("datasets/tr_word2cluster.pickle", "wb"))

df_vectorized = df.copy()
df_vectorized["index"] = df_vectorized.index
df_vectorized.set_index("index",inplace=True)
df_vectorized.head(5)



