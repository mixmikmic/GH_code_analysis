# Here's a brief implementation of what we'll be doing in this session

import torch
import spacy
from torchtext import data, datasets

# 1. create Field objects for preprocessing / tokenizing text
TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), 
                  init_token='<SOS>', eos_token='<EOS>',lower=True)

# 2. create Dataset objects with input text files and Field object(s) which return tokenized and processed text data
train,val,test = datasets.WikiText2.splits(text_field=TEXT)

# 3. create Vocab objects that contain the vocabulary and word embeddings of the words from our data
TEXT.build_vocab(train, wv_type='glove.6B')
vocab = TEXT.vocab

import torch
from torchtext import data, datasets

# 1.0. setting up a default0 Field object
TEXT = data.Field()

# 1.1. applying an external tokenizer

"""
The default tokenizer implemented in torchtext is merely
a .split() function, so we usually have to use our own versions.
Luckily, the most commonly used 'spacy' tokenizer can be
easily called.
Here we call a spacy tokenizer for English, and add it to our Field.
"""

tokenizer = data.get_tokenizer('spacy') # spacy tokenizer function
test_string = 'This is a string to be tokenized...'
print('original string: ',test_string)
print('tokenized string: ',list(tokenizer(test_string)))

TEXT = data.Field(tokenize=tokenizer) # add our tokenizer to our Field object

"""
Or you can do this manually by calling it from the spacy package.
FYI, spacy provides tokenizers of MANY languages.
"""

import spacy
spacy_en = spacy.load('en') # the default English package by Spacy

def tokenizer2(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

test_string = 'This is a string to be tokenized...'
print('original string: ',test_string)
print('tokenized string: ',list(tokenizer(test_string)))

TEXT = data.Field(tokenize=tokenizer2)

# 1.2. converting word tokens to indices

"""
use_vocab is set default to True. By setting it to False, 
we instead take number indices as inputs.
In most cases your inputs will in text and not integer indices, 
so you won't be using this feature that much.
"""

TEXT = data.Field(use_vocab=False)

# 1.3. Adding SOS and EOS tokens to input strings

"""
In seq2seq models, it is (almost) necessary to append 
SOS(start-of-sentence) and EOS(end-of-sentence) tokens to let the model 
determine when to start and stop generating new words. 
These are applied to the dataset created using this Field.
"""

TEXT = data.Field(init_token='<SOS>', eos_token='<EOS>')

# 1.4. converting text to lowercase

"""
Converts all words to lower case
"""

TEXT = data.Field(lower=True)

# 1.5. fixed vs variable lengths

"""
In default, RNNs in Pytorch are invariant of sequence length, 
so it is possible to train models using variable sequence lengths.
However, you may have to fix input lengths 
such as when using CNNs and other models.

Note: Even when set to variable length, when working in batches
the all sentences are padded to match the longest sentence in a batch.
"""

TEXT = data.Field(fix_length=40) # shorter strings will be padded

# Actual Field object to be used for next step

TEXT = data.Field(tokenize=tokenizer, init_token='<SOS>', eos_token='<EOS>',
                 lower=True)
vars(TEXT)

# create a LanguageModelingDataset instance using TEXT as Field

"""
dataset: Cornell Movie-Dialogs Corpus
"""

lang = datasets.LanguageModelingDataset(path='movie.txt',
                                       text_field=TEXT)

# 2.1. print examples from text

"""
You can print examples(=all text) of the given text using Dataset.examples.
When using the basic LanguageModelingDataset, the entire text corpus will
be stored as a long list of word tokens.
"""

examples = lang.examples
print("Number of tokens: ", len(examples[0].text))
print("\n")
print("Print first 100 tokens: ",examples[0].text[:100])
print("\n")
print("Print last 10 tokens: ",examples[0].text[-10:])

# 2.2. Dataset for language modelling

"""
Torchtext provides the Wikitext dataset as downloadable for language modelling.
Through the .splits() function of the WikiText2 class, we can create
training / validation / test datasets.
Note that this class can ONLY read from the WikiText2 data.
"""

# get Field
TEXT_wiki = data.Field(tokenize=tokenizer, init_token='<SOS>', eos_token='<EOS>',
                 lower=True)

# split into train, val, test
train, val, test = datasets.WikiText2.splits(text_field=TEXT_wiki)

# 2.3. Dataset for sentiment analysis

"""
We will use the same .split() function provided by the SST class in torchtext.
The only difference is that we also need a label field to keep a vocabulary of
the labels from 'very negative' to 'very positive'
"""

# get Fields - sentiment analysis also requires a label field
TEXT_sst = data.Field(tokenize=tokenizer, init_token='<SOS>', eos_token='<EOS>',
                 lower=True)
LABEL_sst = data.Field(sequential=False)

# split into train, val, test
train, val, test = datasets.SST.splits(text_field=TEXT_sst, label_field=LABEL_sst)

# 2.4. Dataset for natural language inference

# WIP

# 3.0. building a vocabulary

"""
We build a vocabulary of words included in the dataset.
Note that we can only build on the Field object that was used to create the dataset.
"""

TEXT.build_vocab(lang) # use dataset as input
vocab = TEXT.vocab

# 3.1. vocabulary information (size, frequency of words, etc.)

"""
You can view information of the vocabulary.
Vocab.freqs returns a Counter object that stores the frequency of
all words that appeared in the text.
"""

print("Vocabulary size: ", len(vocab))
print("10 most frequent words: ", vocab.freqs.most_common(10))

# 3.2. string2index (stoi), index2string (itos)

"""
The created Vocab object contains an index mapping for each word.
"""

print("First 10 words: ", vocab.itos[0:10])
print("First 10 words of text data: ", lang.examples[0].text[:10])
print("Index of the first word: ", vocab.stoi[lang.examples[0].text[0]])

# 3.3. create purpose-specific vocabularies (requires a Counter object)

"""
The Vocabulary object created from Field have many parameters set to default.
We can create a new vocabulary using any Counter object 
(e.g. the Counter object from our initial vocabulary). 
Our new vocabulary may
1) only contains word which appear more than N times,
2) be smaller than a given maximum size
"""

counter = vocab.freqs # frequency of the original vocabulary created by Field
vocab2 = data.Vocab(counter=counter,min_freq=10) # discard words appearing less than 10 times
vocab3 = data.Vocab(counter=counter,max_size=100000) # set max number of words for a vocabulary

print(len(vocab))
print(len(vocab2))
print(len(vocab3))

# 3.4. load external word embeddings

"""
We can load external word embeddings such as word2vec or glove with Vocab objects.
Here we build a Vocab object using 'glove.6B'
wv_dir = only if you have a custom word embedding file in .pt, .txt, .zip
wv_dim = 100, 200, 300 (provided by glove) # default=300
wv_type = 'glove.6B', 'glove.840B', 'glove.twitter.27B', 'glove.42B'
"""

########## NOTE: this external word embedding requires 800+ MB space ########## 
# 3.4.1. downloading embedding and loading into Field object 
GLOVE = data.Field()
lang2 = datasets.LanguageModelingDataset(path='movie.txt',
                                       text_field=GLOVE)
GLOVE.build_vocab(lang2, wv_type='glove.6B')

# 3.4.2. loading embedding into specific Vocab object
vocab2.load_vectors(wv_type='glove.6B', wv_dim=100)

# 3.4.3. word embeddings in Vocab objects

"""
Word embedding vectors are accessible via Vocab.vectors.
They are treated as any normal FloatTensor.
"""

print("Word embedding size: ", vocab2.vectors.size())
print(vocab2.vectors[0:10])

# 3.5. easily handle unknown words

"""
While you needed exceptions for dealing with unknown words in normal dictionaries,
when using a Vocab object it automatically assigns <unk> to any unknown word.
"""

unknown_word = "humbahumba"
print("Index for unknown word %s: %d" %(unknown_word, vocab2.stoi[unknown_word]))
print("Token for unknown word: ", vocab2.itos[vocab2.stoi[unknown_word]])

