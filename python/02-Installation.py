get_ipython().system('pip install -U nltk')

import nltk
print('NLTK works! Time to download the data.')

nltk.download('all')

get_ipython().system('pip install -U spacy && python3 -m spacy download en')

import spacy
print('spaCy works!')

import nltk
import spacy
print('NLTK {}'.format(nltk.__version__))
print('spaCy {}'.format(spacy.__version__))

