from jupyter_cms.loader import load_notebook

eda = load_notebook('./data_exploration.ipynb')

df, newspapers = eda.load_data()

import pandas as pd

pd.set_option('display.max_columns', 100)

df.head(3)

print('''Rows: {}
Dates: {} ({} - {})
'''.format(
    df.shape[0],
    df.date.nunique(),
    df.date.min(),
    df.date.max()
))

import spacy

nlp = spacy.load('en')

docs = []
for i, doc in enumerate(nlp.pipe(df.text, batch_size=10000, n_threads=7)):
    if i % 5000 == 0:
        print('.', end='')
    docs.append(doc)

def remove_token(t):
    return not t.is_alpha or t.is_stop

lemmas = []

for d in docs:
    d_lemmas = []
    for t in d:
        if not remove_token(t):
            d_lemmas.append(t.lemma_)
    
    lemmas.append(d_lemmas)

import itertools
df['lemmas'] = lemmas

newspaper_text = df.groupby(['date']).lemmas.apply(lambda x: list(itertools.chain(*x)))
newspapers_per_day = df.groupby(['date']).slug.nunique()

import sys
from collections import Counter

newspaper_tfs = []

# tf - number of times word shows up in current document
# doc_freqs - number of documents that has a given word

for i, d in enumerate(newspaper_text):
    if i % 10000 == 0:
        print('.', end='')
        sys.stdout.flush()
    tf = Counter(d)
    newspaper_tfs.append(tf)

get_ipython().magic('matplotlib inline')
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 5))
plot1 = fig.add_subplot(131)
plot2 = fig.add_subplot(132)
plot3 = fig.add_subplot(133)

plot1.plot(range(len(newspaper_text)), [len(text) for text in newspaper_text])
plot1.set_xlabel("Day of scrape")
plot1.set_ylabel("Words")

plot2.plot(range(len(day_vocab_props)), [len(x) for x in day_vocab_props])
plot2.set_xlabel("Day of scrape")
plot2.set_ylabel("Unique words")

plot3.plot(range(len(newspapers_per_day)), newspapers_per_day.values)
plot3.set_xlabel("Day of scrape")
plot3.set_ylabel("Number of newspapers")
plt.tight_layout()

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
X = dv.fit_transform(newspaper_tfs)

from sklearn.feature_selection import chi2

def get_labels_for_day(day, N):
    arr = np.zeros(N)
    arr[day] = 1
    
    return arr

get_labels_for_day(2, 5)

N = len(newspaper_tfs)
words = np.array(dv.get_feature_names())

top_words_by_day = []

for i in range(N):
    print('.', end = '')
    sys.stdout.flush()
    
    keyness, _ = chi2(X, get_labels_for_day(i, N))
    ranking = np.argsort(keyness)[::-1]
    top_words = words[ranking]
    top_words_by_day.append(list(zip(top_words, keyness[ranking])))

sum([sys.getsizeof(x) for x in top_words_by_day])

for date, top_words in zip(newspaper_meta, top_words_by_day):
    print('.', end='')
    sys.stdout.flush()
    
    date_str = pd.to_datetime(str(date)).strftime('%Y-%m-%d')
    
    with open('results/top-words/{}.csv'.format(date_str), 'w') as out:
        out.write('\n'.join([','.join([line[0], str(np.round(line[1], 2))]) for line in top_words]))

newspaper_day_text = df.groupby(['date', 'slug']).lemmas.apply(lambda x: list(itertools.chain(*x)))

newspaper_day_meta = df.groupby(['date', 'slug']).first().reset_index()[['date', 'slug']]

newspaper_day_tf = []

for lemmas in newspaper_day_text:
    newspaper_day_tf.append(Counter([lemma for lemma in lemmas if len(lemma) > 2]))

dv = DictVectorizer()
X = dv.fit_transform(newspaper_day_tf)

newspaper_day_tf = np.array(newspaper_day_tf)

def get_day(day):
    date = newspaper_day_meta.date.unique()[day]
    return newspaper_day_meta[newspaper_day_meta.date == date].index

def get_slug_in_day(slug, day):
    date = newspaper_day_meta.date.unique()[day]
    ndf = newspaper_day_meta[newspaper_day_meta.date == date].reset_index()
    return ndf[ndf.slug == slug].index[0]

def get_day_slugs(day):
    date = newspaper_day_meta.date.unique()[day]
    return newspaper_day_meta[newspaper_day_meta.date == date].slug.values

top_words_by_slug_day = []
words = np.array(dv.get_feature_names())

for i in range(N):
    print('.', end = '')
    sys.stdout.flush()
    
    day_ix = get_day(i)
    X_universe = X[day_ix, ]
    
    day_slugs = get_day_slugs(i)
    M = len(day_slugs)
    for slug in day_slugs:
        j = get_slug_in_day(slug, i)
        
        keyness, _ = chi2(X_universe, get_labels_for_day(j, M))
        ranking = np.argsort(np.nan_to_num(keyness))[::-1]
        top_words = words[ranking[:100]]
        top_words_by_slug_day.append(top_words)

len(top_words_by_slug_day)

1

1

