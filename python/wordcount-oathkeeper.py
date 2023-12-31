import pandas as pd

# prepare cleanup function
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# read data
df = pd.read_pickle("nationalOathKeepers")

# clean up
df['post_clean'] = [clean(doc).split() for doc in df.post_content]

# Print top words in cleaned-up input data
import collections
ncount = 500
collections.Counter(" ".join(   df["post_content"]   ).split()).most_common(ncount)



