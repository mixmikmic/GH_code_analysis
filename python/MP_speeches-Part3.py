import pandas as pd
import bcolz
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models.word2vec import LineSentence
import codecs
import os

import os.path
import time

while True:
    if os.path.isfile("raw_speeches.h5"):
        break
    time.sleep(60)

if True:
    # Load english language model from spacy
    import spacy
    nlp = spacy.load("en")
    # If it complains, you may need to downgrade pip: pip install pip==9.0.1

# Directory to store Phrase models
from config import INTERMEDIATE_DIRECTORY

speeches = bcolz.open("speeches.bcolz")

get_ipython().system('mkdir -p $INTERMEDIATE_DIRECTORY')

# Save speeches to txt file first to make it quicker to process in batches with lower memory
speeches_filepath = os.path.join(INTERMEDIATE_DIRECTORY, "speeches.txt")
# Set to True if you want to run this again
if False:
    with codecs.open(speeches_filepath, "w", encoding="utf_8") as f:
        for speech in speeches["body"]:
            f.write(speech + "\n")

#%%writefile helper_functions.py

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    
    return token.is_punct or token.is_space

def line_speech(filename):
    """
    generator function to read in speeches from the file
    and un-escape the original line breaks in the text
    """
    
    with codecs.open(filename, encoding='utf_8') as f:
        for speech in f:
            yield speech.replace('\\n', '\n')

def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse speeches,
    lemmatize the text, and yield sentences
    """
    
    for parsed_speech in nlp.pipe(line_speech(filename),
                                  batch_size=10000, n_threads=8):
        
        for sent in parsed_speech.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])

get_ipython().run_cell_magic('time', '', "# this is a bit time consuming (takes about 1h) - make the if statement True\n# if you want to execute data prep yourself.\nunigram_sentences_filepath = os.path.join(INTERMEDIATE_DIRECTORY, 'unigram_sentences_all.txt')\nif False:\n    with codecs.open(unigram_sentences_filepath, 'w', encoding='utf_8') as f:\n        for sentence in lemmatized_sentence_corpus(speeches_filepath):\n            f.write(sentence + '\\n')")

get_ipython().system('tail intermediate/unigram_sentences_all.txt')

get_ipython().run_cell_magic('time', '', "# this is a bit time consuming - make the if statement True\n# if you want to execute modeling yourself.\nbigram_model_filepath = os.path.join(INTERMEDIATE_DIRECTORY, 'bigram_model_all')\nif False:\n    # Open unigram sentences as a stream\n    unigram_sentences = LineSentence(unigram_sentences_filepath)\n    bigram_model = Phrases(unigram_sentences)\n    bigram_model.save(bigram_model_filepath)\nelse:\n    # load the finished model from disk\n    bigram_model = Phrases.load(bigram_model_filepath)\n# Phraser class is much faster than Phrases\nbigram_phraser = Phraser(bigram_model)")

get_ipython().run_cell_magic('time', '', "# this is a bit time consuming (takes about 20 mins) - make the if statement True\n# if you want to execute data prep yourself.\nbigram_sentences_filepath = os.path.join(INTERMEDIATE_DIRECTORY, 'bigram_sentences_all.txt')\nif False:\n    with codecs.open(bigram_sentences_filepath, 'w', encoding='utf_8') as f: \n        for unigram_sentence in unigram_sentences:\n            bigram_sentence = u' '.join(bigram_model[unigram_sentence])\n            f.write(bigram_sentence + '\\n')")

get_ipython().system('tail intermediate/bigram_sentences_all.txt')

get_ipython().run_cell_magic('time', '', "## Learn a trigram model from bigrammed speeches\n\n# this is a bit time consuming - make the if statement True\n# if you want to execute modeling yourself.\ntrigram_model_filepath = os.path.join(INTERMEDIATE_DIRECTORY, 'trigram_model_all')\nif False:\n    # Open bigram sentences as a stream\n    bigram_sentences = LineSentence(bigram_sentences_filepath)\n    trigram_model = Phrases(bigram_sentences)\n    trigram_model.save(trigram_model_filepath)\nelse:\n    # load the finished model from disk\n    trigram_model = Phrases.load(trigram_model_filepath)\ntrigram_phraser = Phraser(trigram_model)")

get_ipython().run_cell_magic('time', '', "## Save speeches as trigrams in txt file\n\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\ntrigram_sentences_filepath = os.path.join(INTERMEDIATE_DIRECTORY, 'trigram_sentences_all.txt')\nif False:\n    with codecs.open(trigram_sentences_filepath, 'w', encoding='utf_8') as f:\n        for bigram_sentence in bigram_sentences:\n            trigram_sentence = u' '.join(trigram_model[bigram_sentence])\n            f.write(trigram_sentence + '\\n')\n# Open trigrams file as stream\ntrigram_sentences = LineSentence(trigram_sentences_filepath)")

get_ipython().system('tail intermediate/trigram_sentences_all.txt')

# Load last names and pronouns into stopwords so that they are filtered out
from spacy.en.language_data import STOP_WORDS

for word in ["mr.", "mrs.", "ms.", "``", "sir", "madam", "gentleman", "colleague", "gentlewoman", "speaker", "-PRON-"] + list(pd.read_hdf("list_of_members.h5", "members").last_name.str.lower().unique()):
    STOP_WORDS.add(word)

def clean_text(parsed_speech):
   # lemmatize the text, removing punctuation and whitespace
    unigram_speech = [token.lemma_ for token in parsed_speech
                      if not punct_space(token)]

    # remove any remaining stopwords
    unigram_speech = [term for term in unigram_speech
                      if term not in STOP_WORDS]
    
    # apply the bigram and trigram phrase models
    bigram_speech = bigram_phraser[unigram_speech]
    trigram_speech = trigram_phraser[bigram_speech]

    # write the transformed speech as a line in the new file
    trigram_speech = u' '.join(trigram_speech) 
    
    return trigram_speech

clean_text(nlp("I congratulate the gentlewoman from Maryland (Mrs. Morella), the  gentleman from Tennessee (Mr. Gordon), and the gentleman from Michigan  (Mr. Barcia) for their hard work on this legislation. Also, we would  not be here without the assistance and support of the gentleman from  New York (Chairman Boehlert) and his efforts to bring this bill to the  floor. This a timely piece of legislation, Madam Speaker, and I would  urge my colleagues to support the bill.   Madam Speaker, I reserve the balance of my time.   Mr. HALL of Texas. Madam Speaker, I yield such time as he may consume  to the gentleman from Tennessee (Mr. Gordon), who was ranking member on  the Subcommittee on Environment, Technology, and Standards back when  this legislation first began and wrote the electronic authentication  provisions in it. He is now ranking member on the Subcommittee on Space  and Aeronautics.   Mr. HALL of Texas. Madam Speaker, I have no further requests for  time, and I yield back the balance of my time."))

get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming (takes about 2h) - make the if statement True\n# if you want to execute data prep yourself.\ntrigram_speeches_filepath = os.path.join(INTERMEDIATE_DIRECTORY, 'trigram_transformed_speeches_all.txt')\nif True:\n    with codecs.open(trigram_speeches_filepath, 'w', encoding='utf_8') as f:  \n        for parsed_speech in nlp.pipe(line_speech(speeches_filepath),\n                                      batch_size=10000, n_threads=4):\n            f.write(clean_text(parsed_speech) + '\\n')")

get_ipython().system('tail -n 2 intermediate/trigram_transformed_speeches_all.txt')

