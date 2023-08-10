import nltk
# Launch the installer to download "gutenberg" and "stop words" corpora.
nltk.download()

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import scipy
import sklearn
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# Utility function for standard text cleaning.
def text_cleaner(text):
    # Visual inspection identifies a form of punctuation spaCy does not
    # recognize: the double dash '--'.  Better get rid of it now!
    text = re.sub(r'--',' ',text)
    text = re.sub("[\[].*?[\]]", "", text)
    text = ' '.join(text.split())
    return text

from nltk.corpus import inaugural

inaugural.fileids()

# Load and clean the data.
p == 0 
for speech in inaugural.fileids():
    speech = inaugural.raw(speech)
    p += 1

reagan81 = text_cleaner(reagan81)
reagan85 = text_cleaner(reagan85) 

#nltk sents
reagan81_sents = nltk.sent_tokenize(reagan81)
reagan85_sents = nltk.sent_tokenize(reagan85)

reagan81_sents = [[sent, "Reagan81"] for sent in reagan81_sents]
reagan85_sents = [[sent, "Reagan85"] for sent in reagan85_sents]


sentences = pd.DataFrame(reagan81_sents + reagan85_sents) 
sentences.tail()

# Utility function to create a list of the 2000 most common words.
def bag_of_words(text):
    
    # Filter out punctuation and stop words.
    allwords = [token.lemma_
                for token in text
                if not token.is_punct
                and not token.is_stop]
    
    # Return the most common words.
    return [item[0] for item in Counter(allwords).most_common(2000)]
    

# Creates a data frame with features for each word in our common word set.
# Each value is the count of the times the word appears in each sentence.
def bow_features(sentences, common_words):
    
    # Scaffold the data frame and initialize counts to zero.
    df = pd.DataFrame(columns=common_words)
    df['text_sentence'] = sentences[0]
    df['text_source'] = sentences[1]
    df.loc[:, common_words] = 0
    
    # Process each row, counting the occurrence of words in each sentence.
    for i, sentence in enumerate(df['text_sentence']):
        
        # Convert the sentence to lemmas, then filter out punctuation,
        # stop words, and uncommon words.
        words = [token.lemma_
                 for token in sentence
                 if (
                     not token.is_punct
                     and not token.is_stop
                     and token.lemma_ in common_words
                 )]
        
        # Populate the row with word counts.
        for word in words:
            df.loc[i, word] += 1
        
        # This counter is just to make sure the kernel didn't hang.
        if i % 500 == 0:
            print("Processing row {}".format(i))
            
    return df


# Set up the bags.
reagan81words = bag_of_words(reagan81)
reagan85words = bag_of_words(reagan85)

# Combine bags to create a set of unique words.
common_words = set(reagan81words + reagan85words)



