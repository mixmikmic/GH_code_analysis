# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NLP libraries
import re, nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
#nltk.download('stopwords')

# Word2Vec packages
import gensim
from gensim.models import Word2Vec

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
#    level=logging.INFO)

# load the reviews into a pandas dataframe
data_file_name = "./0.datasets/Automotive_5.json"
df = pd.read_json(data_file_name, lines=True)
print("Dataset contains {0:,} reviews".format(df.shape[0]))
df.head()

# Get the ReviewText into a new dataframe
all_reviews = pd.DataFrame(df['reviewText'])
all_reviews.shape

# Select sample set of reviews as appropriate
sample_reviews = all_reviews.head(1000)

# Function to preprocess text - Remove non-letters, lowercase, optionally remove stop words
def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    # 1. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    #
    # 2. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 3. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 4. Return a list of words
    return(words)

# Download the punkt tokenizer for sentence splitting
import nltk.data
nltk.download('punkt')  

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence,remove_stopwords=True ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

# Get the sentences from the review corpus. Sentences are the inputs to Word2Vec algorithm
sentences = []  # Initialize an empty list of sentences

for review in sample_reviews['reviewText']:
    sentences += review_to_sentences(review, tokenizer)
print("We have {0:,} sentences".format(len(sentences)))

## Setting the parameters for Word2Vec

import multiprocessing

# Set values for various parameters for Word2Vec
num_features = 50    # Word vector dimensionality                      
min_word_count = 5   # Minimum word count                        
num_workers = multiprocessing.cpu_count()      # Number of threads to run in parallel
context_size = 3          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize Word2Vec. Since sentences are supplied it will build vocab and then train the model
model_name = "auto2vec_model"
if 1==1:  # Set this to 0==1 if you want to utilize the already trained model
    auto2vec_trained = Word2Vec(sentences, workers=num_workers,sg=1,                 size=num_features, min_count = min_word_count,                 window = context_size, sample = downsampling)


    print("The vocabulary is built")
    print("Auto2Vec vocabulary length: ", len(auto2vec_trained.vocab))

    # If you don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
    auto2vec_trained.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()    
    auto2vec_trained.save(model_name)
    print ("Training model completed...")

# load the finished model from disk
print("Loading model from disk")
auto2vec = Word2Vec.load(model_name)
auto2vec.init_sims()

ordered_vocab = [(term, voc.index, voc.count)
                 for term, voc in auto2vec.vocab.items()]

# sort by the term counts, so the most common terms appear first
ordered_vocab = sorted(ordered_vocab, key=lambda x:(x[2]), reverse=True)

ordered_terms, term_indices, term_counts = zip(*ordered_vocab)

word_vectors = pd.DataFrame(auto2vec.wv.syn0norm[term_indices,:], index=ordered_terms)
#word_vectors[25:75]

#auto2vec.most_similar("auto")

def get_related_terms(token, topn=10):
    """
    look up the topn most similar terms to token
    and print them as a formatted list
    """

    for word, similarity in auto2vec.most_similar(positive=[token], topn=topn):

        print (u'{:20} {}'.format(word, round(similarity, 3)))

get_related_terms(u'jumper',topn=10)

def word_algebra(add=[], subtract=[], topn=1):
    """
    combine the vectors associated with the words provided
    in add= and subtract=, look up the topn most similar
    terms to the combined vector, and print the result(s)
    """
    answers = auto2vec.most_similar(positive=add, negative=subtract, topn=topn)
    
    for term, similarity in answers:
        print (term)

word_algebra(add=[u'vehicle', u'lawn'],topn=5)
#word_algebra(add=[u'vehicle', u'lawn'], subtract=[u'wheels'])

auto2vec.doesnt_match("car truck engine use".split())

import spacy
import pandas as pd
import itertools as it

nlp = spacy.load('en')

# Convert all the review text into a long string and print its length
raw_corpus = u"".join(sample_reviews['reviewText']+" ")
print("Raw Corpus contains {0:,} characters".format(len(raw_corpus)))

if 1==1: # Make this 0==1, if you don't want to run the analysis again
    parsed_review = nlp(raw_corpus)

if 1==1: # Make this 0==1, if you don't want to run the analysis again
    for num, sentence in enumerate(parsed_review.sents):
        print ('Sentence {}:'.format(num + 1))
        print (sentence)
        print ('')

if 0==1: # Make this 0==1, if you don't want to run the analysis again
    for num, entity in enumerate(parsed_review.ents):
        print ('Entity {}:'.format(num + 1), entity, '-', entity.label_)
        print ('')

if 1==1: # Make this 0==1, if you don't want to run the analysis again
    token_text = [token.orth_ for token in parsed_review]
    token_pos = [token.pos_ for token in parsed_review]

    pos_dataframe = pd.DataFrame(list(zip(token_text, token_pos)),columns=['token_text', 'part_of_speech'])

pos_dataframe

if 1==1: # Make this 0==1, if you don't want to run the analysis again
    token_lemma = [token.lemma_ for token in parsed_review]
    token_shape = [token.shape_ for token in parsed_review]

    print(pd.DataFrame(list(zip(token_text, token_lemma, token_shape)),
                 columns=['token_text', 'token_lemma', 'token_shape']))

if 1==1: # Make this 0==1, if you don't want to run the analysis again
    token_entity_type = [token.ent_type_ for token in parsed_review]
    token_entity_iob = [token.ent_iob_ for token in parsed_review]

    print(pd.DataFrame(list(zip(token_text, token_entity_type, token_entity_iob)),
                 columns=['token_text', 'entity_type', 'inside_outside_begin']))

if 1==1: # Make this 0==1, if you don't want to run the analysis again
    token_attributes = [(token.orth_,
                         token.prob,
                         token.is_stop,
                         token.is_punct,
                         token.is_space,
                         token.like_num,
                         token.is_oov)
                        for token in parsed_review]

    df = pd.DataFrame(token_attributes,
                      columns=['text',
                               'log_probability',
                               'stop?',
                               'punctuation?',
                               'whitespace?',
                               'number?',
                               'out of vocab.?'])

    df.loc[:, 'stop?':'out of vocab.?'] = (df.loc[:, 'stop?':'out of vocab.?']
                                           .applymap(lambda x: u'Yes' if x else u''))

    df

df



