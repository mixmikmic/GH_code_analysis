import attr
import re
import numpy as np
import spacy

from collections import Counter, defaultdict
from itertools import islice
from boltons.iterutils import windowed
from textblob import TextBlob
from cached_property import cached_property
from tqdm import tqdm_notebook
from multiprocessing import Pool

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

nlp = spacy.load('en')

def worker(ab):
    counts = Counter()
    for sent in ab.sentences:
        for ngram in sent.tokens:
            counts[ngram] += 1
    return counts

class Corpus:
    
    def __init__(self, path, skim=None):
        self.path = path
        self.skim = skim
        
    def lines(self):
        with open(self.path) as fh:
            for line in fh:
                yield line.strip()
    
    def abstract_lines(self):
        lines = []
        for line in self.lines():
            if line:
                lines.append(line)
            else:
                yield lines
                lines = []

    def abstracts(self):
        ab_lines = self.abstract_lines()
        if self.skim:
            ab_lines = islice(ab_lines, self.skim)
        for lines in tqdm_notebook(ab_lines, total=self.skim):
            yield Abstract.from_lines(lines)
            
    def xy(self, vocab):
        for abstract in self.abstracts():
            yield from abstract.xy(vocab)
            
    def ngram_counts(self, n):
        
#         def worker(ab):
#             counts = defaultdict(lambda: 0)
#             for sent in ab.sentences:
#                 for ngram in sent.tokens:
#                     counts[ngram] += 1
#             return counts
        
        counts = defaultdict(lambda: 0)
        
        with Pool() as pool:
            jobs = pool.imap_unordered(worker, self.abstracts())
            for res in jobs:
                for k, v in res.items():
                    counts[k] += v
        
        return counts
            
#         counts = defaultdict(lambda: 0)
#         for ab in self.abstracts():
#             for sent in ab.sentences:
#                 for ngram in sent.tokens:
#                     counts[ngram] += 1
#         return Counter(counts)
            
    def most_common_ngrams(self, n, depth):
        counts = self.ngram_counts(n)
        return set([k for k, _ in counts.most_common(depth)])

@attr.s
class Abstract:
    
    identifier = attr.ib()
    tags = attr.ib()
    sentences = attr.ib()
    
    @classmethod
    def from_lines(cls, lines):
        sentences = list(map(Sentence, lines[2:]))
        return cls(lines[0], lines[1].split(), sentences)
    
    def sentence_tokens(self):
        for sent in self.sentences:
            yield re.findall('[a-z]+', sent.lower())
    
    def xy(self, vocab):
        for i, sent in enumerate(self.sentences):
            x = sent.features(vocab)
            y = i / (len(self.sentences)-1)
            yield x, y

class Sentence:
    
    def __init__(self, text):
        self.text = nlp(text)
        
    @cached_property
    def tokens(self):
        return [t.text.lower() for t in self.text]
    
    def token_ngrams(self, n=1):
        for ng in windowed(self.tokens, n):
            yield '_'.join(ng)
            
    def token_ngram_counts(self, vocab=None, maxn=3):
        for n in range(1, maxn+1):
            counts = Counter(self.token_ngrams(n))
            for k, v in counts.items():
                if not vocab or k in vocab:
                    yield f'_{k}', v
                    
    def word_count(self):
        return len(self.tokens)
                
    def _features(self, vocab=None):
        yield from self.token_ngram_counts(vocab)
        yield 'word_count', self.word_count()
        
    def features(self, vocab=None):
        return dict(self._features(vocab))

s = Sentence('Does this work?')

s.features()

train = Corpus('../data/abstracts/train.txt', 100000)

vocab = (
    train.most_common_ngrams(1, 2000) |
    train.most_common_ngrams(2, 2000) |
    train.most_common_ngrams(3, 2000)
)

dv = DictVectorizer()

train_x, train_y = zip(*train.xy(vocab))

train_x = dv.fit_transform(train_x)

train_x

model = LinearRegression()

fit = model.fit(train_x, train_y)

test = Corpus('../data/abstracts/test.txt', 50000)

test_x, test_y = zip(*test.xy(vocab))

test_x = dv.transform(test_x)

r2_score(test_y, fit.predict(test_x))

names = dv.get_feature_names()

bidx = fit.coef_.argsort()
eidx = np.flip(fit.coef_.argsort(), 0)

for i in bidx[:50]:
    print(fit.coef_[i], names[i])

for i in eidx[:50]:
    print(fit.coef_[i], names[i])

