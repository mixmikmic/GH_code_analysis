import nltk
import pattern.en
import pattern.it
import langdetect as ld
import string
from collections import defaultdict, Counter
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pymongo

class Tokenizer(object):
    
    def __init__(self, preserve_case=True):
        self.tweet = nltk.tokenize.TweetTokenizer(preserve_case=preserve_case)
        self.lang_map = defaultdict(lambda: (SnowballStemmer('english'), pattern.en.parsetree))
        self.lang_map['en'] = (SnowballStemmer('english'), pattern.en.parsetree)
        self.lang_map['it'] = (SnowballStemmer('italian'), pattern.it.parsetree)
    
    @staticmethod
    def lang(doc):
        try:
            lang = ld.detect(doc)
        except Exception:
            lang = 'en'
        return lang
    
    @staticmethod
    def remove_punctuation(tokens, special_chars=None):
        p = string.punctuation
        if special_chars is not None:
            p += "".join(special_chars)
        return [x for x in tokens if x not in p]
    
    def stemming(self, tokens):
        lang = Tokenizer.lang(" ".join(tokens))
        stemmer = self.lang_map[lang][0]
        stems = [stemmer.stem(t) for t in tokens]
        return stems
    
    def tweet_tokenizer(self, doc):
        return self.tweet.tokenize(doc)
    
    def pattern_processing(self, doc, lemmata=False):
        p = self.lang_map[Tokenizer.lang(doc)][1]
        tree = p(doc, lemmata=lemmata)
        tokens, lemmata = [], []
        for sentence in tree:
            for word in sentence.words:
                tokens.append(word.string)
                lemmata.append(word.lemma)
        return tokens, lemmata

class MIndex(defaultdict):
    
    def __init__(self):
        super(MIndex, self).__init__(lambda: [])
        self.docs = set()
    
    def boolean(self, doc_id, tokens):
        self.docs.add(doc_id)
        for token in set(tokens):
            self[token].append(doc_id)
    
    def boolean_to_matrix(self):
        features = list(self.keys())
        docs = list(self.docs)
        M = np.zeros((len(docs), len(features)))
        for token, posting in self.items():
            for doc in posting:
                ti, di = features.index(token), docs.index(doc)
                M[di][ti] = 1
        return M > 0, features, docs
    
class DBIndex(object):
    """
    Indexing with storage on MongoDB
    """
    
    def __init__(self, db_name, db_host='127.0.0.1'):
        self.db = pymongo.MongoClient(host=db_host)[db_name]
    
    def index(self, collection, doc_id, words, tokens):
        data = []
        for i, t in enumerate(tokens):
            data.append({'doc_id': doc_id, 'word': words[i], 
                         'token': t, 'pos': i})
        try:
            self.db[collection + '_pos'].insert_many(data)
        except:
            pass
    
    def stream(self, collection):
        """
        Streams tokens per document in the same order than self.docs
        """
        c = collection + '_pos'
        group = {'$group': {'_id': '$doc_id', 'tokens': {'$push': '$token'}}}
        sort = {'$sort': {'_id': 1}}
        pipeline = [group, sort]
        for record in self.db[c].aggregate(pipeline):
            yield record['tokens']
        
    
    def idf_to_vec(self, collection):
        c = collection + '_idf'
        tokens = self.tokens(collection)
        idf = np.zeros(len(tokens))
        for record in self.db[c].find():
            token, w = record['_id'], record['idf']
            idf[tokens.index(token)] = w
        return idf
    
    def tf_to_matrix(self, collection):
        c = collection + '_tf'
        tokens = self.tokens(collection)
        docs = self.docs(collection)
        mtf = np.zeros((len(docs), len(tokens)))
        for record in self.db[c].find():
            doc_id, token = record['doc_id'], record['token']
            w = record['tfn']
            mtf[docs.index(doc_id), tokens.index(token)] = w
        return mtf
    
    def docs(self, collection):
        c = collection + '_pos'
        return sorted(self.db[c].distinct('doc_id'))
    
    def tokens(self, collection):
        c = collection + '_pos'
        return sorted(self.db[c].distinct('token'))
    
    def tf(self, collection, k=0.5):
        c = collection + '_tf'
        raw = collection + '_raw'
        g = {'$group': {'_id': '$_id.d', 'm': {'$max': '$c'}}}
        pipeline = [g]
        d = {}
        for record in self.db[raw].aggregate(pipeline):
            d[record['_id']] = record['m']
        to_insert = []
        for record in self.db[raw].find():
            token, doc, tf = record['_id']['t'], record['_id']['d'], record['c']
            tfn = k * float(tf) + (1 - k) * float(tf) / d[doc]
            to_insert.append({
                'token': token, 'doc_id': doc, 'tf': tf, 'tfn': tfn
            })
        self.db[c].insert_many(to_insert)
    
    def idf(self, collection):
        c = collection + '_idf'
        self.db[c].drop()
        corpus_size = float(len(self.docs(collection)))
        g = {'$group': {'_id': '$_id.t', 'df': {'$sum': 1} }}
        pipeline = [g]
        to_insert = []
        for record in self.db[collection + '_raw'].aggregate(pipeline):
            token, df = record['_id'], record['df']
            idf = np.log(corpus_size / df)
            to_insert.append({
                '_id': token,
                'df': df,
                'idf': idf
            })
        self.db[c].insert_many(to_insert)

    def aggregate(self, collection):
        g = {
            '$group': {
                '_id': {'t': '$token', 'd': '$doc_id'},
                'c': {'$sum': 1}
            }
        }
        o = {'$out': collection + '_raw'}
        self.db[collection + '_raw'].drop()
        pipeline = [g, o]
        agg_cur = self.db[collection + '_pos'].aggregate(pipeline)        
        

