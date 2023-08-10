import pdfminer as pdf
import os
import glob
import pickle as pkl
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim
import unicodedata
import re, string
from gensim import corpora, models
import gensim

# Get a blob of the pdf filenames, and turn them into a list
files = glob.glob('/Users/sararogis/Dropbox/FoodRecommender/MealRec_LitReview/*.pdf')

files[0:5]

text_files = glob.glob('/Users/sararogis/Dropbox/FoodRecommender/MealRec_LitReview/LitRev_Text/*.txt')

for x in files:
    dest_loc = x.replace('.pdf','.txt').replace('MealRec_LitReview/','MealRec_LitReview/LitRev_Text/')
    ocr_cmd = 'pdf2txt.py -o ' + dest_loc + ' -t text ' + x
    os.system(ocr_cmd)

text_files[0:5]

# Ok, now pull these text files in and lets do some lda

all_texts = []

for textpath in text_files:
    # Set an open string to place document text into
    all_doc_string = ''
    
    # Open the file and save contents to variable 'file_text'
    file_text = open(textpath, 'rb')
    
    # Read the lines of the tester into a variable. 
    file_lines = file_text.readlines()
    
    # Iterate through the lines, and append them to the string to make one big text blob
    for line in file_lines:
        all_doc_string = all_doc_string + line
        
    # Append the new string to the list of all files
    all_texts.append(all_doc_string)
    
    # String will reset next; close the file and flush the file buffer
    file_text.flush()
    file_text.close()

all_texts

len(all_texts)

len(text_files)

# Hmmm. One off, but what the heck. Save it. 
pkl.dump(all_texts, open('all_texts_mealrec_litreview.pkl','wb'))

#####

df = pd.DataFrame(all_texts,columns=['orig_text'])

def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import LineTokenizer
    from nltk.tokenize import WhitespaceTokenizer
    from nltk.stem.porter import PorterStemmer
    
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    
    sw = list(stopwords.words())
    extra_stops = ['R', '', ' ', 'abstract', 'keywords', 'introduction', 'figure','morgan', 'harvey',
                   'david','elsweiler','northumbria','university','newcastle','united','kingdom','university',
                   'regensburg','germany', 'h', 'k', 'f', 'b', 'user', 'g', 'use']
    for word in extra_stops:
        sw.append(word)
    
    # Step 1 - Clean up unicode
    clean_string = ''
    doc = []
    #for x in text:
    #    if ord(x) <= 128:
    #        clean_string += x
    #clean_string = unicodedata.normalize('NFKD', clean_string.encode('utf-8', 'replace')).encode('ascii','replace')
            
    # Tokenize each line to get rid of the line carriages
    lines = LineTokenizer().tokenize(text.lower())
    
    clean_lines = []
    
    for line in lines:
        if line.startswith('e-mail') or line.startswith('doi') or line.startswith('For all other uses, contact') or line.find(' acm. isbn ') > 0:
            pass
        else:
            line_str = ''
            for char in line:
                #if ord(char) <= 127:
                if (char in string.ascii_letters) or char == ' ':
                    line_str += char
        
        # Clean up other undesirable characters
            if line_str != ' ' and line_str.rstrip().lstrip() not in sw:
                clean_lines.append(line_str)
    
    # Tokenize the lines
    for clean_line in clean_lines:
        tokens = WhitespaceTokenizer().tokenize(clean_line)
        stopped_tokens = [i for i in tokens if not i in sw]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        [doc.append(i) for i in stemmed_tokens]
        
    
    return doc

df.insert(df.shape[1], 'clean_text', df.orig_text.apply(preprocess_text))

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(df.clean_text)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in df.clean_text]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)

ldamodel4 = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word = dictionary, passes=20)

ldamodel3 = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=10, num_words=4))

import pyLDAvis

pyLDAvis.enable_notebook()

import pyLDAvis.gensim

pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

pyLDAvis.gensim.prepare(ldamodel3, corpus, dictionary)



