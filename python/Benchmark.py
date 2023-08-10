import wikipedia
from collections import defaultdict
import json
import codecs
import os
import pymongo
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA

class Wikisearch(object):
    
    def __init__(self, out_folder, results=20):
        self.folder = out_folder
        self.results = results
        self.queries = []
        self.mapping = defaultdict(lambda: set())
    
    def search(self, query):
        qi = len(self.queries)
        self.queries.append(query)
        results = wikipedia.search(query, results=self.results)
        for result in results:
            try:
                page = wikipedia.page(result)
                self.mapping[qi].add((page.pageid, page.url))
            except wikipedia.exceptions.DisambiguationError as e:
                for option in e.options:
                    page = wikipedia.page(option)
                    self.mapping[qi].add((page.pageid, page.url))
    
    def save(self, content=True):
        """
        If content is set to False, gets page summary
        """
        outdict = {}
        for qi, qt in enumerate(self.queries):
            pages = list(self.mapping[qi])
            outdict[qi] = {'query': qt, 'page_ids': [x[0] for x in pages], 
                           'page_urls': [x[1] for x in pages]}
            for page_id, page_url in pages:
                page = wikipedia.page(pageid=page_id)
                if content:
                    text = page.content
                else:
                    text = page.summary
                with codecs.open(os.sep.join([self.folder, page_id + '.txt']), 'wb', encoding='utf-8') as tout:
                    tout.write(text)
        with codecs.open(os.sep.join([self.folder, 'queries.json']), 'wb', encoding='utf-8') as jout:
            json.dump(outdict, jout)           

class ImageTags(object):
    
    def __init__(self, db_name, collection, selection=None,
                category='category', query='query',
                tag='name', weight='value', url='path', doc_id='doc_id',
                host='localhost'):
        self.category, self.query, self.tag = category, query, tag
        self.weight, self.url, self.doc_id = weight, url, doc_id
        self.db = pymongo.MongoClient(host=host)[db_name]
        self.images = self.db[collection]
        if selection is None:
            self.selection = {}
        else:
            self.selection = selection
        group = {'$group': {'_id': None, 
                            'docs': {'$addToSet': '$' + self.url},
                            'tags': {'$addToSet': '$' + self.tag}
                           }}
        match = {'$match': self.selection}
        pipeline = [match, group]
        for record in self.images.aggregate(pipeline):
            self.docs = record['docs']
            self.tags = record['tags']
        self.M = np.zeros((len(self.docs), len(self.tags)))
        for record in self.images.find(self.selection):
            self.M[self.docs.index(record[self.url])][self.tags.index(record[self.tag])] = record[self.weight]
    
    @property
    def idf(self):
        i = np.array([np.log(float(self.M.shape[1]) / (np.count_nonzero(x) + 1)) for x in self.M.T])
        return i
    
    @property
    def tfidf(self):
        return self.M * self.idf
    
    @property
    def probs(self):
        s = np.sum(self.M, axis=1)
        return (self.M.T / s).T
    
    def get_url(self, doc_pos):
        return self.docs[doc_pos]
    
    def tag_stream(self):
        for i, row in enumerate(self.M):
            tags = [self.tags[y] for y in np.where(row > 0)[0]]
            yield tags
        
    def html(self, doc_pos, width=200, local=False):
        url = self.get_url(doc_pos)
        if local:
            prefix = 'file://'
        else:
            prefix = ''
        if url is None:
            return ""
        else:
            return '<img src="{}{}" style="width: {}px;">'.format(prefix, url, width)
        
    def query_vector(self, query, as_probs=False, use_tfidf=False):
        q = np.zeros(self.M.shape[1])
        for t in query:
            try:
                q[self.tags.index(t)] += 1
            except ValueError:
                pass
        if as_probs:
            q /= q.sum()
        elif use_tfidf:
            q *= self.idf
        return q
    
    def vector_search(self, query, distance, use_tfidf=False):
        if use_tfidf:
            candidates = self.tfidf
        else:
            candidates = self.M
        results = []
        for i, v in enumerate(candidates):
            results.append((i, distance(query, v)))
        return sorted(results, key=lambda x: x[1])
    
    def multinomial_search(self, query):
        qw = np.where(query > 0)[0]
        results = []
        candidates = self.probs
        for i, v in enumerate(candidates):
            doc_p = []
            for token in qw:
                doc_p.append(np.log(1 + np.power(v[token], query[token])))
            results.append((i, sum(doc_p)))
        return sorted(results, key=lambda x: -x[1])
    
    def bm25(self, query, k=0.5, b=0.5):
        qw = np.where(query > 0)[0]
        results = []
        candidates = self.M
        lavg = np.array([np.count_nonzero(x) for x in self.M]).mean()
        df = np.array([float(self.M.shape[1]) / (np.count_nonzero(x) + 1) for x in self.M.T])
        for i, v in enumerate(candidates):
            rsv = 0
            for t in qw:
                w = np.log((1+(df[t]*(k+1)*v[t])) / (((k*((1-b) + b*np.count_nonzero(v)/lavg)) + v[t])+1))
                rsv += w
            results.append((i,rsv))
        return sorted(results, key=lambda x: -x[1])
    
    def generate_queries(self, n_queries, threshold=None):
        """
        Exploits LDA to find topics that will be used as queries as follows
        Each topic defines a query
        Topic tags is the query (with their probabilities as relevance)
        Topic docs is the ground truth (with doc probabilities as relevance)
        """
        data = self.tfidf
        lda = LDA(n_components=n_queries, learning_method='batch')
        docs = lda.fit_transform(data)
        queries = []
        tags = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
        for topic in range(0, n_queries):
            tq = sorted([(self.tags[i], p) for i, p in enumerate(tags[topic])], key=lambda x: -x[1])
            dq = sorted([(self.docs[i], p) for i, p in enumerate(docs.T[topic])], key=lambda x: -x[1])
            if threshold is None:
                ts = np.argmax(np.diff([x[1] for x in tq], n=2))
                ds = np.argmax(np.diff([x[1] for x in dq], n=2))
                queries.append((tq[:ts], dq[:ds])) 
            else:
                queries.append((tq[:threshold[0]], 
                              [y for y in dq if y[1] >= threshold[1]]))
        return queries, docs, tags
        

