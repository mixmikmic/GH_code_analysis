import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.word2vec import LineSentence
import gensim
import os
import warnings
warnings.filterwarnings('ignore')

from config import INTERMEDIATE_DIRECTORY

get_ipython().run_cell_magic('time', '', "# this is a bit time consuming - make the if statement True\n# if you want to relearn the dictionary.\ntrigram_speeches_filepath = os.path.join(INTERMEDIATE_DIRECTORY, 'trigram_transformed_speeches_all.txt')\ntrigram_dictionary_filepath = os.path.join(INTERMEDIATE_DIRECTORY, 'trigram_dict_all.dict')\nif False:\n    trigram_speeches = LineSentence(trigram_speeches_filepath)\n\n    # learn the dictionary by iterating over all of the speeches\n    trigram_dictionary = Dictionary(trigram_speeches)\n    \n    # filter tokens that are very rare or too common from\n    # the dictionary (filter_extremes) and reassign integer ids (compactify)\n    trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)\n    trigram_dictionary.compactify()\n\n    trigram_dictionary.save(trigram_dictionary_filepath)\nelse: \n    # load the finished dictionary from disk\n    trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)")

def trigram_bow_generator(filepath):
    """
    generator function to read speeches from a file
    and yield a bag-of-words representation
    """
    
    for speech in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(speech)

get_ipython().run_cell_magic('time', '', "trigram_bow_filepath = os.path.join(INTERMEDIATE_DIRECTORY, 'trigram_bow_corpus_all.mm')\n# this is a bit time consuming - make the if statement True\n# if you want to build the bag-of-words corpus yourself.\nif False:\n    # generate bag-of-words representations for\n    # all speches and save them as a matrix\n    MmCorpus.serialize(trigram_bow_filepath,\n                       trigram_bow_generator(trigram_speeches_filepath))")

get_ipython().run_cell_magic('time', '', "## Train the LDA topic model using Gensim\nfrom gensim.models.ldamulticore import LdaMulticore\n\n# this is a bit time consuming (takes about 1h30 mins)- make the if statement True\n# if you want to train the LDA model yourself.\nlda_model_filepath = os.path.join(INTERMEDIATE_DIRECTORY, 'lda_model_all_75')\nif False:\n    # load the finished bag-of-words corpus from disk\n    trigram_bow_corpus = MmCorpus(trigram_bow_filepath)\n    with warnings.catch_warnings():\n        warnings.simplefilter('ignore')\n        \n        # workers => sets the parallelism, and should be\n        # set to your number of physical cores minus one\n        lda = LdaMulticore(trigram_bow_corpus,\n                           num_topics=75,\n                           id2word=trigram_dictionary,\n                           workers=4)\n    \n    lda.save(lda_model_filepath)\nelse:\n    # load the finished LDA model from disk\n    lda = LdaMulticore.load(lda_model_filepath)")

def explore_topic(topic_number, topn=25):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
        
    print('{:20} {}'.format('term', 'frequency'))

    for term, frequency in lda.show_topic(topic_number, topn=topn):
        print('{:20} {:.3f}'.format(term, round(frequency, 3)))

explore_topic(0, topn=10)

get_ipython().run_cell_magic('time', '', 'import pickle\nimport pyLDAvis.gensim\nfrom gensim.corpora import  MmCorpus\nimport pyLDAvis\nimport os\n\nldavis_pickle_path = os.path.join(INTERMEDIATE_DIRECTORY, "pyldavis_75.p")\n# Change to True if you want to recalculate the visualisation (takes about 1h)\nif False:\n    trigram_bow_corpus = MmCorpus(trigram_bow_filepath)\n    LDAvis_prepared = pyLDAvis.gensim.prepare(lda, trigram_bow_corpus,\n                                                  trigram_dictionary, sort_topics=False)\n    pickle.dump(LDAvis_prepared, open(ldavis_pickle_path, "wb"))\nelse:\n    LDAvis_prepared = pickle.load(open(ldavis_pickle_path, "rb"))')

pyLDAvis.display(LDAvis_prepared)

# %load topic_names.py
# Dictionary of topic names
topic_names_100 = {
    0: "congress terminology",
    1: "congress terminology+",
    2: "healthcare",
    3: "",
    4: "random names (?)",
    5: "science",
    6: "social security",
    7: "africa?",
    8: "national security",
    9: "midwest (?)",
    10: "government finances",
    11: "OSHA",
    12: "honors",
    13: "congress terminology++",
    14: "budget",
    15: "consumer protection",
    16: "committees",
    17: "sports",
    18: "arts & culture",
    19: "military equipment (?)",
    20: "coast guard & fishing industry",
    21: "",
    22: "judaism",
    23: "fire fighting",
    24: "",
    25: "NASA",
    26: "homeland security",
    27: "african-american history",
    28: "affordable housing",
    29: "war history (?)",
    30: "",
    31: "",
    32: "",
    33: "china",
    34: "census",
    35: "veteran affairs",
    36: "agriculture",
    37: "FEMA",
    38: "unemployment",
    39: "",
    40: "civil rights",
    41: "transport",
    42: "",
    43: "army corps",
    44: "awards",
    45: "mexico & turkey (?)",
    46: "",
    47: "senior citizens healthcare",
    48: "abortion",
    49: "legislation",
    50: "constitution",
    51: "california",
    52: "texas",
    53: "foreign policy",
    54: "defense",
    55: "medicine + campaign finance reform",
    56: "",
    57: "native americans",
    58: "mental health",
    59: "land management & forestry",
    60: "drug enforcement",
    61: "birth control & women's rights",
    62: "iraq & afghanistan",
    63: "FCC",
    64: "EPA",
    65: "medicare",
    66: "welfare reform",
    67: "",
    68: "",
    69: "employee-employer relations",
    70: "",
    71: "school education",
    72: "",
    73: "trade policy",
    74: ""
}

topic_names_50 = {
    1: "israel & palestine",
    2: "labor relations",
    3: "transportation",
    4: "arts & culture",
    5: "tobacco",
    6: "aviation",
    7: "FDA",
    8: "medicare",
    9: "fiscal policy",
    10: "budget",
    12: "secondary education",
    13: "wildlife conservation",
    14: "veteran affairs",
    15: "nasa & baseball ??",
    16: "health insurance",
    17: "agriculture",
    18: "elections",
    19: "disease prevention",
    20: "medicare+",
    21: "first names (male)",
    22: "young people",
    23: "iraq & afghanistan",
    25: "russia",
    26: "housing",
    27: "homeland security",
    28: "welfare reform",
    29: "honors",
    30: "oil & gas",
    31: "constitional law",
    32: "EPA",
    33: "indian affairs",
    34: "FEMA",
    35: "energy",
    36: "honors+",
    37: "affordable care act (?)",
    38: "border enforcement",
    39: "congressional terminology",
    40: "MURICA",
    41: "china & india",
    42: "business & corporate responsibility (?)",
    43: "congressional terminology+",
    44: "congressional terminology++",
    45: "veterans",
    46: "congressional terminology+++",
    47: "tax",
    48: "african-american civil rights",
    49: "minimum wages"
}

topic_names_75 = {
    0: "healthcare",
    1: "constitution",
    2: "african-american history",
    3: "IRS & trafficking",
    5: "infrastructure & construction",
    7: "african-americans",
    9: "military",
    10: "secondary education",
    11: "border enforcement",
    12: "agriculture",
    13: "cybersecurity",
    14: "budget",
    15: "entrepreneurship",
    16: "transportation",
    17: "worker welfare",
    18: "medical research",
    20: "india & pakistan",
    21: "congressional terminology",
    23: "first names",
    26: "armenian genocide",
    27: "drugs war",
    28: "puerto rico",
    29: "sports",
    30: "community service",
    31: "???",
    32: "tobacco",
    34: "iraq & afghanistan wars",
    35: "affordable housing",
    37: "honors",
    38: "congressional terminology+",
    39: "human rights in china",
    40: "natural disasters",
    42: "financial sector",
    43: "congressional terminology++",
    44: "congressional terminology+++",
    45: "energy",
    48: "veteran affairs",
    49: "child welfare",
    50: "gun safety & abortion",
    51: "honors+",
    55: "veterans",
    56: "unions",
    57: "health insurance & medicare",
    59: "honors++",
    60: "forestry",
    61: "arts",
    62: "nuclear waste",
    63: "trade",
    68: "law",
    70: "terrorism & foreign policy",
    71: "defense",
    72: "voting rights & democracy",
    74: "homeland security"
}

topic_names = topic_names_75
def topic_dict(topic_number):
    """
    return name of topic where identified
    """
    
    try:
        return topic_names[topic_number]
    except KeyError:
        return topic_number
    
# Reverse the topic names so that we can find them easily
reverse_topic_dict = {i[1]:i[0] for i in topic_names.items()}

get_ipython().run_cell_magic('time', '', "if True:\n    # Load the bigram and trigram models so we can apply this to any new text (Takes about 3 mins)\n    bigram_model_filepath = os.path.join(INTERMEDIATE_DIRECTORY, 'bigram_model_all')\n    trigram_model_filepath = os.path.join(INTERMEDIATE_DIRECTORY, 'trigram_model_all')\n    # load the bigram model from disk\n    bigram_model = gensim.models.Phrases.load(bigram_model_filepath)\n    # load the trigram model from disk\n    trigram_model = gensim.models.Phrases.load(trigram_model_filepath)\n    # Phraser class is much faster so use this instead of Phrase\n    bigram_phraser = gensim.models.phrases.Phraser(bigram_model)\n    trigram_phraser = gensim.models.phrases.Phraser(trigram_model)")

# %load helper_functions.py
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
                                  batch_size=10000, n_threads=4):
        
        for sent in parsed_speech.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])

if True:
    # Load english language model from spacy
    import spacy
    nlp = spacy.load("en")

# Load last names and pronouns into stopwords
from spacy.en.language_data import STOP_WORDS

for word in ["mr.", "mrs.", "ms.", "``", "sir", "madam", "gentleman", "colleague", "gentlewoman", "speaker", "-PRON-"] + list(pd.read_hdf("list_of_members.h5", "members").last_name.str.lower().unique()):
    STOP_WORDS.add(word)

def clean_text(speech_text):
    """
    Remove stop words, lemmatize and split into tokens using the trigram parser
    and return a bag-of-words representation
    """
    
    # parse the review text with spaCy
    parsed_speech = nlp(speech_text)
    
    # lemmatize the text, removing punctuation and whitespace
    unigram_speech = [token.lemma_ for token in parsed_speech
                      if not punct_space(token)]

    # remove any remaining stopwords
    unigram_speech = [term for term in unigram_speech
                      if term not in STOP_WORDS]
    
    # apply the bigram and trigram phrase models
    bigram_speech = bigram_phraser[unigram_speech]
    trigram_speech = trigram_phraser[bigram_speech]

    return trigram_speech

def lda_description(speech_text):
    """
    accept the original text of a speech and (1) parse it with spaCy,
    (2) apply text pre-proccessing steps, (3) create a bag-of-words
    representation, (4) create an LDA representation, and
    (5) print a series containing all the topics and their probabilities in the LDA representation
    """
    
    import numpy as np
    
    # Get clean representation of text
    trigram_speech = clean_text(speech_text)

     # create a bag-of-words representation
    speech_bow = trigram_dictionary.doc2bow(trigram_speech)
    
    # create an LDA representation
    speech_lda = lda[speech_bow]
    
    topic_dict = dict(zip(range(75), [0.0]*75))
    
    for topic in speech_lda:
        topic_dict[topic[0]] = topic[1]
    topic_dict["n_words"] = len(trigram_speech)
    print(trigram_speech)
    return pd.Series(topic_dict).astype(np.float16)

lda_description("There is a lot of natural gas in the middle east.").rename(lambda x: topic_dict(x)).sort_values(ascending=False)

get_ipython().run_cell_magic('time', '', '# This takes a while (~5h) so use cached version if available\n# Change to True if you want to recalculate the LDA topics for the speeches\n# MAKE SURE YOU DELETE THE PREVIOUS processed_speeches.h5 FILE FIRST\nif False:\n    import bcolz # For lazy loading speeches\n    from tqdm import tqdm # For a progress bar\n    from multiprocessing import Pool # For spreading out topic modelling over several cores\n    import numpy as np\n\n    # Load speech metadata\n    speeches_meta = pd.read_hdf("speech_metadata.h5", "metadata")\n    # Lazy load array of speeches\n    speeches = bcolz.open("speeches.bcolz")\n    # Remove old file\n    #!rm processed_speeches.h5\n    # Store max string lengths for later\n    max_str_lengths = dict(zip(["doc_title", "id", "speaker"], map(lambda x: speeches_meta[x].str.len().max(), ["doc_title", "id", "speaker"])))\n    # Loop over all the speeches in chunks of 1024 strings. This method keeps memory requirements low. If you have less memory, make CHUNK_SIZE smaller\n    CHUNK_SIZE = 1024\n    for i in tqdm(range(0, len(speeches), CHUNK_SIZE)):\n        # Using multiprocessing, create a pool of 8 workers\n        with Pool(8) as pool:\n            # Apply lda function to each speech in chunk\n            df = pd.DataFrame(list(pool.map(lda_description, speeches[i: i+CHUNK_SIZE])))\\\n                .assign(n_words=lambda x: x.n_words.astype(np.int16))\n            # Align index so that there are no duplicate indices in the final dataframe\n            df.index = df.index+i\n            # Concatenate the speech metadata with the results of the lda topic distribution\n            pd.concat([speeches_meta.iloc[i:i+CHUNK_SIZE], df], axis=1).to_hdf("processed_speeches_75.h5",\n                                                                       "speeches", append=True,\n                                                                       format="table", min_itemsize=max_str_lengths)\n            \n    ## Check that arrays are aligned and that the values stored are the same as computed. Uncomment if you want to do this.\n    assert (pd.read_hdf("processed_speeches_75.h5", "speeches").iloc[60, range(75)] - lda_description(speeches[60]).loc[range(75)]).abs().mean() < 1e-4\nelse:\n    try:\n        del speeches\n    except NameError:\n            pass\n\n    speeches = pd.read_hdf("processed_speeches_50.h5", "speeches")')

