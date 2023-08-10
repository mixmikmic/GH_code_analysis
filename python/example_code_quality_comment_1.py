from bs4 import BeautifulSoup as bsoup
import re
import os
import nltk
from nltk.collocations import *
from itertools import chain
import itertools
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import MWETokenizer

xml_file_path = "./xml_files"

def parsing(t):

    xmlSoup = bsoup(t,"lxml-xml")
    
    pid = xmlSoup.find("publication-reference").find('doc-number').string 
    
    text = ""
    
    #Extract text in "abstract"    
    abt = xmlSoup.find('abstract')
    for p in abt.findAll('p'):
        text = text + p.text + " "
    
    #Extract Claims 
    for tag in xmlSoup.find_all('claim-text'):
        text = text + tag.text
 
    return (pid, text)

patents_raw = {}
for xfile in os.listdir(xml_file_path): 
    xfile = os.path.join(xml_file_path, xfile)
    if os.path.isfile(xfile) and xfile.endswith('.XML'): 
        (pid, text) = parsing(open(xfile))
        patents_raw[pid] = text

tokenizer = RegexpTokenizer(r'[a-zA-Z]{2,}') 

def tokenizePatent(pid):
    """
        the tokenization function is used to tokenize each patent.
        The one argument is patent_id.
        First, normalize the case.
        Then, use the regular expression tokenizer to tokenize the patent with the specified id
    """
    raw_patent = patents_raw[pid].lower() 
    tokenized_patents = tokenizer.tokenize(raw_patent)
    return (pid, tokenized_patents) # return a tupel of patent_id and a list of tokens

patents_tokenized = dict(tokenizePatent(pid) for pid in patents_raw.keys())

all_words = list(chain.from_iterable(patents_tokenized.values()))

bigram_measures = nltk.collocations.BigramAssocMeasures()
bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(all_words)
bigram_finder.apply_freq_filter(20)
bigram_finder.apply_word_filter(lambda w: len(w) < 3)# or w.lower() in ignored_words)
top_100_bigrams = bigram_finder.nbest(bigram_measures.pmi, 100) # Top-100 bigrams
top_100_bigrams

mwetokenizer = MWETokenizer(top_100_bigrams)
colloc_patents =  dict((pid, mwetokenizer.tokenize(patent)) for pid,patent in patents_tokenized.items())
all_words_colloc = list(chain.from_iterable(colloc_patents.values()))
colloc_voc = list(set(all_words_colloc))
print(len(colloc_voc))

pids = []
patent_words = []
for pid, tokens in colloc_patents.items():
    pids.append(pid)
    txt = ' '.join(tokens)
    patent_words.append(txt)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(input = 'content', analyzer = 'word')
tfidf_vectors = tfidf_vectorizer.fit_transform(patent_words)
tfidf_vectors.shape

save_file = open("patent_student.txt", 'w')

vocab = tfidf_vectorizer.get_feature_names()
#########please write the missing code below#######
cx = tfidf_vectors.tocoo() # return the coordinate representation of a sparse matrix
for i,j,v in itertools.zip_longest(cx.row, cx.col, cx.data):
    save_file.write(pids[i] + ',' + vocab[j] + ',' + str(v) + '\n')

save_file.close()



