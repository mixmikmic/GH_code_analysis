from collections import defaultdict
import pattern.en as ptn
import networkx as nx
import numpy as np
import string

class BiGram(defaultdict):
    
    def __init__(self, method=None):
        """
        Must be initialized with a valid method for extracting
        a part of information from the pattern.en word object
        Default will be _get_string
        """
        super(BiGram, self).__init__(lambda: defaultdict(lambda: 0))
        self.start = '<start>'
        self.stop = '<stop>'
        self.punct = '.'
        if method is None:
            self.get = BiGram._get_string
        else:
            self.get = method
    
    def add(self, text, keep_punctuation=True):
        bigrams = self.parse(text, keep_punctuation=keep_punctuation)
        for a, b in bigrams:
            self[a][b] += 1
    
    def parse(self, text, keep_punctuation=True):
        sentences = ptn.parsetree(text, lemmata=True)
        pairs = []
        if keep_punctuation:
            punctuation = ""
        else:
            punctuation = string.punctuation
        for sentence in sentences:
            first = self.get(sentence[0])
            pairs.append((self.start, first))
            for w1, w2 in ptn.ngrams(sentence, n=2, 
                                     punctuation=punctuation):
                pairs.append((self.get(w1), self.get(w2)))
            try:
                pairs.append((self.get(w2), self.punct))
            except UnboundLocalError:
                pairs.append((first, self.punct))
        pairs.append((self.punct, self.stop))
        return pairs
    
    @staticmethod
    def _get_string(word):
        return word.string.lower()
    
    @staticmethod
    def _get_type(word):
        return word.type
    
    @staticmethod
    def _get_lemma(word):
        return word.lemma

class HMM(nx.DiGraph):
    
    def __init__(self, bigram_model, min_occurrences=0):
        super(HMM, self).__init__()
        self.model = bigram_model
        self._create_net(m=min_occurrences)
    
    def _create_net(self, m=0):
        for k, choices in self.model.items():
            s = float(sum([x for x in choices.values() if x > m]))
            for z, w in choices.items():
                if w > m:
                    p = w / s
                    self.add_edge(k, z, probability=p)
    
    def step(self, start):
        try:
            choices = zip(*[(k, p['probability']) for k, p in self[start].items()])
        except KeyError:
            return self.model.stop
        tokens, probabilities = choices[0], np.array(choices[1])
        return np.random.choice(tokens, p=probabilities)
    
    def generate(self, start=None, length=100):
        counter = 0
        if start is None:
            start = self.model.start
        text, last = [], None
        while counter < length:
            last = self.step(start)
            if last == self.model.stop:
                break
            text.append(last)
            counter += 1
            start = last
        return " ".join(text)
    
    def evaluate(self, text, alpha=0.0001, keep_punctuation=True):
        p = []
        pairs = self.model.parse(text, keep_punctuation=keep_punctuation)
        for a, b in pairs:
            try:
                p_pair = self[a][b]['probability']
            except KeyError:
                p_pair = alpha
            p.append(p_pair)
        return np.array(p).prod()

    
class ProbabilisticRetrieval(object):
    
    def __init__(self, method=None, min_occurrences=0):
        self.method = method
        self.min_o = min_occurrences
        self.texts, self.models = [], []
    
    def indexing(self, docs):
        """
        Docs is an iterable of texts
        """
        for text in docs:
            self.texts.append(text)
            b = BiGram(self.method)
            b.add(text, keep_punctuation=False)
            self.models.append(HMM(b, min_occurrences=self.min_o))
    
    def search(self, query):
        results = []
        for i, m in enumerate(self.models):
            results.append((i, m.evaluate(query, keep_punctuation=False)))
        return sorted(results, key=lambda x: -x[1])
    
    def exerpt(self, doc_id, l=None):
        try:
            if l is None:
                return self.texts[doc_id].split('\n')[0]
            else:
                return self.texts[doc_id][:l]
        except IndexError:
            return 'Document not available'

