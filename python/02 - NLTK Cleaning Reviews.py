import pandas as pd
import cPickle as pickle
# Load the yelp review data
#review = pd.read_pickle('../input/yelp_academic_dataset_review.pickle')
review = pd.read_pickle('../input/yelp_academic_dataset_review_SF.pickle')
from spacy.en import English
nlp = English()


#  Adapted, but much improved from  ----   https://github.com/titipata/yelp_dataset_challenge

import time
import collections
#import scipy.sparse as sp
#import nltk.data
from nltk.tokenize import WhitespaceTokenizer
from unidecode import unidecode
from itertools import chain
import numpy as np
#from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import sys
sys.path.append('../vectorsearch/')
from reverse_stemmer import SnowCastleStemmer
import nltk
import pickle
import string


sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
whitespace_tokenizer = WhitespaceTokenizer()
# tb_tokenizer = TreebankWordTokenizer()
stops = set(stopwords.words("english") + stopwords.words("spanish"))
keep_list = ['after', 'during', 'not', 'between', 'other', 'over', 'under', 
             'most', ' without', 'nor', 'no', 'very', 'against','don','aren']
stops = set([word for word in stops if word not in keep_list])



# Multiword tokenizer list taken from: 
# http://www.cs.cmu.edu/~ark/LexSem/
# http://www.cs.cmu.edu/~ark/LexSem/STREUSLE2.1-mwes.tsv

# This parses a list of multiword expressions from # http://www.cs.cmu.edu/~ark/LexSem/STREUSLE2.1-mwes.tsv
# into NLTK format
MWE = [] 
with open('../input/STREUSLE2.1-mwes.tsv') as f:
    for line in f.readlines():
        multiword_expression = line.split('\t')[0].split()[1:]
        MWE.append(multiword_expression)
MWE_tokenizer = MWETokenizer(MWE, separator='-')
# Add whatever additional custom multi-word-expressions.
MWE_tokenizer.add_mwe(('dive', 'bar'))
MWE_tokenizer.add_mwe(('happy','hour'))

# Stemmer
stemmer = SnowCastleStemmer("english")
wnl = WordNetLemmatizer()
table = string.maketrans("","")

def clean_text(text):
    """Clean and lower string
    Parameters
    ----------
        text : in string format
    Returns
    -------
        text_clean : clean text input in string format
    """
    return text.lower().translate(table, string.punctuation.replace('-',''))


def get_noun_chunks(words):
    '''
    Get noun chunks from spacy's library....
    '''
    doc = nlp(words)
    chunks = [u"-".join(chunk.orth_.split()) for chunk in doc.noun_chunks if len(chunk.orth_.split())>1]
    return chunks


def clean_and_tokenize(text):
    """
    1. Divide review into sentences
    2. clean words
    3. tokenize
    4. multiword tokenize
    5. remove stop words
    6. stem words
    Returns
    ------
        text_filtered: list of word in sentence
    """
    # Splits into sentences.
    if type(text) != str:
        try:
            text = str(text)
        except:
            text = str(text.encode('ascii','ignore'))
    sentence = sent_detector.tokenize(unidecode(text.encode('ascii','ignore')))
    # Clean text: (remove) Remove extra puncuations marks...
    text_clean = map(clean_text, sentence)
    
    noun_chunks = map(lambda x: get_noun_chunks(x.decode('unicode-escape')), text_clean)
    noun_chunks = [x for x in noun_chunks if x != []]
    
    # Multiword expression tokenizer
    text_tokenize = map(lambda x: whitespace_tokenizer.tokenize(x), text_clean)
    #text_tokenize = map(lambda x: MWE_tokenizer.tokenize(x), text_tokenize)
    
    # remove stop words
    text_filtered = map(lambda x: [word for word in x if word not in stops], text_tokenize)
    # lemmetize words (stemming removes too much...)
#     text_stemmed = map(lambda x: [wnl.lemmatize(word) 
#                                   if wnl.lemmatize(word).endswith('e') 
#                                   else stemmer.stem(word) 
#                                   for word in x], text_filtered)
    text_stemmed = map(lambda x: [wnl.lemmatize(word) for word in x], text_filtered)
    
    text_stemmed = text_stemmed + noun_chunks
    return text_stemmed


def unstem_text(text_stemmed):
    '''
    Unstem the text with the lowest count real word.  This helps readability.
    '''
    #unstem with the simplest word.  This helps readability of results...
    text_unstemmed = map(lambda x: [stemmer.unstem(word)[0] 
                                  if len(stemmer.unstem(word))>0
                                  else word
                                  for word in x], text_stemmed)
    return text_unstemmed
    
    
def remove_low_occurence_words(texts, threshold=1): 
    '''
    Remove words that appear fewer than "threshold" times.
    '''
    
    frequency = defaultdict(int)
    for text in texts:
        for sentence in text:
            for token in sentence:
                 frequency[token] += 1
    
    texts = [[[token for token in sentence if frequency[token] > threshold]
              for sentence in text] for text in texts]
    return texts
    

# Select reviews that correspond to the list of bars
#bar_ids = pickle.load(open('../output/bar_ids.pickle', 'r'))
#bar_ids = pickle.load(open('../output/bar_restaurant_ids.pickle', 'r'))
bar_ids = pickle.load(open('../output/bar_ids_SF.pickle', 'r'))


bar_reviews = review[review.business_id.isin(bar_ids)][:]
print 'Number of bars (excluding restaurants)', len(bar_ids)
print 'Number of bar reviews', np.sum(review.business_id.isin(bar_ids))

# Clean and tokenize
print 'Cleaning and tokenizing'
review_sentences = map(clean_and_tokenize, bar_reviews.text.iloc[:])
#review_sentences = map(unstem_text, review_sentences)

# This is a list of reviews 
# each review contains a list of sentences
# each sentence contains a list of words (tokens)
review_sentences = remove_low_occurence_words(review_sentences, threshold=3)
# They must be flattened for word2vec. 
# review_flatten = list(chain.from_iterable(review_sentences)) # This is the input to word2vec

# Append to df and save to file
bar_reviews['cleaned_tokenized'] = review_sentences
#bar_reviews.to_pickle('../output/bar_reviews_cleaned_and_tokenized.pickle')

bar_reviews.to_pickle('../output/bar_reviews_cleaned_and_tokenized_SF.pickle')

# bar_reviews.to_pickle('../output/bar_restaurant_reviews_cleaned_and_tokenized.pickle')

# Examine some samples....

print 'Original'
print bar_reviews['text'].iloc[1]
print 

print 'Tokenized'
print bar_reviews['cleaned_tokenized'].iloc[1]

from spacy.en import English
nlp = English()

doc = nlp(u'We checked this place out this past Monday for their wing night. We have heard that their wings are great and decided it was finally time to check it out. Their wings are whole wings and crispy, which is a nice change of pace. I got their wet Cajun sauce and garlic butter wings. The Cajun did not have a bold enough flavor for me and their sauce is too thin. The sauce was also thin for the garlic butter, but that is more expected. They were better than average, but I dont like seeing all the sauce resting at the bottom of the boat. I would definitely come try this place out again to sample some of the other items on the menu, but this will probably not become a regular stop for wings anytime soon.')
noun_chunks = ["-".join(chunk.orth_.split()) for chunk in doc.noun_chunks if len(chunk.orth_.split())>1]
print noun_chunks



