import nltk
from urllib import request
url = "http://classics.mit.edu/Caesar/gallic.mb.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
type(raw)

tokens = nltk.word_tokenize(raw)
type(tokens)

len(tokens)

tokens[:8]

tokens = nltk.word_tokenize(raw)
tokens = [w.lower() for w in  tokens if w.isalpha()]
tokens[:8]

len(tokens)

fdist1 = nltk.FreqDist(tokens)
print(fdist1)

fdist1.N()

fdist1.hapaxes()

fdist1.max()

from nltk.corpus import stopwords
stopwords=stopwords.words('english')
mynewtokens=[w for w in tokens if w not in stopwords]
Fdist2=nltk.FreqDist(mynewtokens)
print(Fdist2)

Fdist2.most_common(10)

Fdist2.most_common(5)

import pandas as pd
from bokeh.charts import Bar
from bokeh.io import output_notebook, show
output_notebook()
dict = {'frequency': {u'caesar': 481, u'chapter': 402, u'enemy': 359, u'men':314, u'great':299}}
df = pd.DataFrame(dict)
df['word'] = df.index
df
p = Bar(df, values='frequency',label='word')
show(p)

Fdist2.most_common()[-5:]

import pandas as pd
from bokeh.charts import Bar
from bokeh.io import output_notebook, show
output_notebook()
dict = {'frequency': {u'south':1, u'measured':1, u'fugitive':1, u'deeper' :1, u'spend':1}}
df = pd.DataFrame(dict)
df['word'] = df.index
df
p = Bar(df, values='frequency',label='word')
show(p)

