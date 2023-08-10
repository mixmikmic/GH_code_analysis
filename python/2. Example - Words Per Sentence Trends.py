import nltk

from nltk.corpus import inaugural

inaugural.fileids()

for speech in inaugural.fileids():
    words_total = len(inaugural.words(speech))
    print words_total, speech

speech_len = [(len(inaugural.words(speech)), speech) for speech in inaugural.fileids()]

max(speech_len)

min(speech_len)

for speech in inaugural.fileids():
    words_total = len(inaugural.words(speech))
    sents_total = len(inaugural.sents(speech))
    print words_total/sents_total, speech

import pandas as pd

data = pd.DataFrame([int(speech[:4]), len(inaugural.words(speech))/len(inaugural.sents(speech))] for speech in inaugural.fileids())

data.columns = ['Year', 'Average WPS']

data.head(10)

import matplotlib
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set_style("whitegrid")
data.plot("Year", figsize=(15,5))

