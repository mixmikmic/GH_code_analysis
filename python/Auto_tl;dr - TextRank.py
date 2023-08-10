filename = 'summarytest.txt' #Enter Filename

file = open(filename,'r')
Text = ""
for line in file.readlines():
    Text+=str(line)
    Text+=" "
file.close()

import nltk
from nltk import word_tokenize
import string

def clean(text):
    printable = set(string.printable)
    text = filter(lambda x: x in printable, text) #filter funny characters, if any.
    return text

Cleaned_text = clean(Text)

text = word_tokenize(Cleaned_text)
case_insensitive_text = word_tokenize(Cleaned_text.lower())

# Sentence Segmentation

sentences = []
tokenized_sentences = []
sentence = " "
for word in text:
    if word != '.':
        sentence+=str(word)+" "
    else:
        sentences.append(sentence.strip())
        tokenized_sentences.append(word_tokenize(sentence.lower().strip()))
        sentence = " "
        

from nltk.stem import WordNetLemmatizer

def lemmatize(POS_tagged_text):
    
    wordnet_lemmatizer = WordNetLemmatizer()
    adjective_tags = ['JJ','JJR','JJS']
    lemmatized_text = []
    
    for word in POS_tagged_text:
        if word[1] in adjective_tags:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a")))
        else:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun
    
    return lemmatized_text

#Pre_processing:

POS_tagged_text = nltk.pos_tag(case_insensitive_text)
lemmatized_text = lemmatize(POS_tagged_text)

Processed_text = nltk.pos_tag(lemmatized_text)

def generate_stopwords(POS_tagged_text):
    stopwords = []
    
    wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','FW'] #may be add VBG too
    
    for word in POS_tagged_text:
        if word[1] not in wanted_POS:
            stopwords.append(word[0])
            
    punctuations = list(str(string.punctuation))
    stopwords = stopwords + punctuations
    
    stopword_file = open("long_stopwords.txt", "r")
    #Source = https://www.ranks.nl/stopwords

    for line in stopword_file.readlines():
        stopwords.append(str(line.strip()))

    return set(stopwords)

stopwords = generate_stopwords(Processed_text)

processed_sentences = []

for sentence in tokenized_sentences:
    processed_sentence = []
    
    POS_tagged_sentence = nltk.pos_tag(sentence)
    lemmatized_sentence = lemmatize(POS_tagged_sentence)

    for word in lemmatized_sentence:
        if word not in stopwords:
            processed_sentence.append(word)
    processed_sentences.append(processed_sentence)

import numpy as np
import math
from __future__ import division

sentence_len = len(processed_sentences)
weighted_edge = np.zeros((sentence_len,sentence_len),dtype=np.float32)

score = np.zeros((sentence_len),dtype=np.float32)

for i in xrange(0,sentence_len):
    score[i]=1
    for j in xrange(0,sentence_len):
        if j==i:
            weighted_edge[i][j]=0
        else:
            for word in processed_sentences[i]:
                if word in processed_sentences[j]:
                    weighted_edge[i][j] += processed_sentences[j].count(word)
            if weighted_edge[i][j]!=0:
                len_i = len(processed_sentences[i])
                len_j = len(processed_sentences[j])
                weighted_edge[i][j] = weighted_edge[i][j]/(math.log(len_i)+math.log(len_j))

inout = np.zeros((sentence_len),dtype=np.float32)

for i in xrange(0,sentence_len):
    for j in xrange(0,sentence_len):
        inout[i]+=weighted_edge[i][j]

MAX_ITERATIONS = 50
d=0.85
threshold = 0.0001 #convergence threshold

for iter in xrange(0,MAX_ITERATIONS):
    prev_score = np.copy(score)
    
    for i in xrange(0,sentence_len):
        
        summation = 0
        for j in xrange(0,sentence_len):
            if weighted_edge[i][j] != 0:
                summation += (weighted_edge[i][j]/inout[j])*score[j]
                
        score[i] = (1-d) + d*(summation)
    
    if np.sum(np.fabs(prev_score-score)) <= threshold: #convergence condition
        print "Converging at iteration "+str(iter)+"...."
        break

i=0
for sentence in sentences:
    print "Sentence:\n\n"+str(sentence)+"\nScore: "+str(score[i])+"\n\n"
    i+=1

Reduce_to_percent = 20
summary_size = int(((Reduce_to_percent)/100)*len(sentences))

if summary_size == 0:
    summary_size = 1

sorted_sentence_score_indices = np.flip(np.argsort(score),0)

indices_for_summary_results = sorted_sentence_score_indices[0:summary_size]

summary = "\n"

current_size = 0

if 0 not in indices_for_summary_results and summary_size!=1:
    summary+=sentences[0]
    summary+=".\n\n"
    current_size+=1


for i in xrange(0,len(sentences)):
    if i in indices_for_summary_results:
        summary+=sentences[i]
        summary+=".\n\n"
        current_size += 1
    if current_size == summary_size:
        break

print "\nSUMMARY: "
print summary



