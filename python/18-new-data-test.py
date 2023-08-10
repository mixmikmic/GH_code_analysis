import matplotlib as mpl
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('bmh')

import attr
import re
import numpy as np
import ujson

from collections import Counter, defaultdict
from itertools import islice
from boltons.iterutils import windowed
from tqdm import tqdm_notebook
from textblob import TextBlob
from glob import glob

from scipy.stats import spearmanr

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class Corpus:
    
    def __init__(self, pattern, skim=None):
        self.pattern = pattern
        self.skim = skim
        
    def lines(self):
        for path in glob(self.pattern):
            with open(path) as fh:
                for line in fh:
                    yield line.strip()

    def abstracts(self):
        lines = self.lines()
        if self.skim:
            lines = islice(lines, self.skim)
        for line in tqdm_notebook(lines, total=self.skim):
            raw = ujson.loads(line)
            yield Abstract.from_raw(raw)
            
    def xy(self, vocab):
        for abstract in self.abstracts():
            yield from abstract.xy(vocab)
            
    def ngram_counts(self, n):
        counts = defaultdict(lambda: 0)
        for ab in self.abstracts():
            for sent in ab.sentences:
                for ngram in sent.ngrams(n):
                    counts[ngram] += 1
        return Counter(counts)
            
    def most_common_ngrams(self, n, depth):
        counts = self.ngram_counts(n)
        return set([k for k, _ in counts.most_common(depth)])

class Abstract:
    
    @classmethod
    def from_raw(cls, raw):
        return cls([Sentence(s['token']) for s in raw['sentences']])
    
    def __init__(self, sentences):
        self.sentences = sentences
    
    def sentence_tokens(self):
        for sent in self.sentences:
            yield re.findall('[a-z]+', sent.lower())
    
    def xy(self, vocab):
        for i, sent in enumerate(self.sentences):
            x = sent.features(vocab)
            y = i / (len(self.sentences)-1)
            yield x, y

class Sentence:
    
    def __init__(self, tokens):
        self.tokens = tokens
    
    def ngrams(self, n=1):
        for ng in windowed(self.tokens, n):
            yield '_'.join(ng)
            
    def ngram_counts(self, vocab, maxn=3):
        for n in range(1, maxn+1):
            counts = Counter(self.ngrams(n))
            for k, v in counts.items():
                if k in vocab:
                    yield f'_{k}', v
                    
    def word_count(self):
        return len(self.tokens)
                
    def _features(self, vocab):
        yield from self.ngram_counts(vocab)
        yield 'word_count', self.word_count()
        
    def features(self, vocab):
        return dict(self._features(vocab))

train = Corpus('/Users/dclure/Projects/sent-order/data/train.json/*.json', 100000)

vocab = (
    train.most_common_ngrams(1, 2000) |
    train.most_common_ngrams(2, 2000) |
    train.most_common_ngrams(3, 2000)
)

len(vocab)

dv = DictVectorizer()

train_x, train_y = zip(*train.xy(vocab))

train_x = dv.fit_transform(train_x)

train_x

model = LinearRegression()

fit = model.fit(train_x, train_y)

test = Corpus('/Users/dclure/Projects/sent-order/data/test.json/*.json', 50000)

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

correct = Counter()
total = Counter()

for ab in test.abstracts():
    
    x, _ = zip(*ab.xy(vocab))
    x = dv.transform(x)
    
    order = list(fit.predict(x).argsort().argsort())
    
    if sorted(order) == order:
        correct[len(order)] += 1
        
    total[len(order)] += 1

for slen in sorted(correct.keys()):
    print(slen, correct[slen] / total[slen])

sum(correct.values()) / sum(total.values())

x = sorted(correct.keys())
y = [correct[slen] / total[slen] for slen in x]

plt.xlabel('Sentence length')
plt.ylabel('Percentage correct')
plt.plot(x, y)



