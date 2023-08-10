def fetch_data():
    import glob
    abstract_files = glob.glob("../data/Hulth2003/train/*.abstr")
    full_data = []
    for file in abstract_files:
        f = open(file, 'rb')
        lines = f.readlines()
        file_data = " ".join([str(line.decode("utf-8").strip()) for line in lines])
        full_data.append(file_data)
    return full_data

import nltk
def get_np_chunks(paragraph):
    phrases = []
    sents = nltk.sent_tokenize(paragraph)
    for sent in sents:
        pos_tags = nltk.pos_tag(nltk.word_tokenize(sent))
        grammar = r"""
          NP: {<PP\$>?<JJ>*<NN>+} 
              {<NN>*<NNP>+}                # chunk sequences of proper nouns
              {<NN>*<NNS>+}
              {<NN>+}
        """
        #{<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
        
        chunkParser = nltk.RegexpParser(grammar)
        chunked = chunkParser.parse(pos_tags)
        for subtree in chunked.subtrees(filter=lambda t: t.label() == 'NP'):
            phrase = " ".join([np[0] for np in subtree.leaves()])
            phrases.append(phrase)
    return phrases

import re

def clean_text(text):
    text = re.sub("[^a-z0-9 ]", " ", text.lower())
    return text.strip()

def get_candidate_phrases(full_data):
    
    doc_phrases = []
    for doc in full_data:
        phrases = get_np_chunks(doc)
        
        cleaned_phrases = []
        for phrase in phrases:
            text = clean_text(phrase)
            if len(text) > 1:
                cleaned_phrases.append(text)
        
        doc_phrases.append(cleaned_phrases)
    return doc_phrases

candidates = get_candidate_phrases(full_data)

import math
from collections import Counter, defaultdict


def get_inverse_document_frequency(candidates):
    
    total_docs = len(candidates)
    doc_unique_phrases = []
    for datum in candidates:
        unique_phrases = list(set(datum))
        doc_unique_phrases.extend(unique_phrases)
    
    doc_freq = Counter(doc_unique_phrases)
    
    full_unique_phrases = doc_freq.keys()
    inv_doc_freq = defaultdict()
    for phrase in full_unique_phrases:
        inv_doc_freq[phrase] = math.log(total_docs / (1.0 + doc_freq[phrase]))
    
    return inv_doc_freq
    

inv_doc_freq = get_inverse_document_frequency(candidates)

from operator import itemgetter

def get_keywords(doc_id):
    phrases = candidates[doc_id]
    phrases_tf = Counter(phrases)
    total_tf = len(phrases)
    
    candidate_phrases = []
    for phrase in set(phrases):
        phrase_n_score = []
        tf = phrases_tf[phrase] / (1.0 * total_tf)
        tf_idf = tf * inv_doc_freq[phrase]
        
        phrase_n_score.append(phrase)
        phrase_n_score.append(tf_idf)
        candidate_phrases.append(phrase_n_score)
    
    keywords = sorted(candidate_phrases, key=itemgetter(1), reverse=True)
    return keywords

get_keywords(3)

full_data[3]

