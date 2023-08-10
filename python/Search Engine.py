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

full_data = fetch_data()

import re
import nltk
english_stopwords = nltk.corpus.stopwords.words('english')

def get_preprocessing(text):
    def remove_nonalphanumeric(text):
        text = re.sub("[^a-zA-Z0-9 ]", " ", text)
        text = text.strip()
        return text

    def convert_lower(text):
        text = text.lower()
        return text

    def remove_stopwords(tokens):
        non_stopwords = []
        for t in tokens:
            if t not in english_stopwords:
                non_stopwords.append(t)
        return non_stopwords
    
    def tokenization(text):
        tokens = text.split()
        return tokens
    
    preprocessed = remove_stopwords(tokenization(convert_lower(remove_nonalphanumeric(text))))
    return preprocessed

get_preprocessing(full_data[0])

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
    

full_data = [get_preprocessing(datum) for datum in full_data]
inverse_doc_freq = get_inverse_document_frequency(full_data)

def get_inverted_index(full_data, inverse_doc_freq):
    inverted_index = defaultdict(lambda : defaultdict(lambda : 0))
    for doc_id in range(len(full_data)):
        doc = full_data[doc_id]
        term_freq = Counter(doc)
        doc_len = len(doc)

        terms = term_freq.keys()
        for term in terms:
            tf = (term_freq[term]/(1.0 * doc_len)) 
            idf = inverse_doc_freq[term]
            
            inverted_index[term][doc_id] = tf*idf
    
    return inverted_index
        

inverted_index = get_inverted_index(full_data, inverse_doc_freq)

query = "atomism thesis"

query_prepcsd = get_preprocessing(query)

from operator import itemgetter

def fetch_docs_SUM_model(query_prepcsd):
    doc_scores = defaultdict(lambda: 0)
    
    for query_term in query_prepcsd:
        if query_term in inverted_index:
            posting_list = inverted_index[query_term]
            for doc_id in posting_list.keys():
                doc_scores[doc_id] += posting_list[doc_id]
            
    
    ranked_list = sorted(list(doc_scores.items()), key=itemgetter(1), reverse=True)
    return ranked_list

def fetch_docs_AND_model(query_prepcsd):
    doc_scores = defaultdict(lambda: 0)
    
    and_docs = []
    for query_term in query_prepcsd:
        if query_term in inverted_index:
            posting_list = inverted_index[query_term]
            docs = posting_list.keys()
            if len(and_docs) == 0:
                and_docs = docs
            else:
                and_docs = list(set(and_docs) & set(docs))

            for doc_id in posting_list.keys():
                doc_scores[doc_id] += posting_list[doc_id]
            
    and_doc_scores = [(doc_id, doc_scores[doc_id]) for doc_id in and_docs]
    ranked_list = sorted(and_doc_scores, key=itemgetter(1), reverse=True)
    return ranked_list

fetch_docs_AND_model(query_prepcsd)

