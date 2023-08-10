class McDonald_Word_List:
    def __init__(self, pos_words, neg_words):
        self.pos_words = pos_words
        self.neg_words = neg_words
        self.pos_word_counts = {word:0 for word, val in pos_words.items()}
        self.intersection_pos = {word:0 for word, val in pos_words.items()}
        self.neg_word_counts = {word:0 for word, val in neg_words.items()}
        self.intersection_neg = {word:0 for word,val in neg_words.items()}
        
    def __str__(self):
        print("""Pos words: {0}
                 Neg words: {1}""".format(
                  len(self.pos_words),
                  len(self.neg_words)))

from xlrd import open_workbook
FORMAT = ['Positive', 'Negative']
values = ""

wb = open_workbook('McDonaldDict.xlsx')

values = []
for s in wb.sheets():
    #print 'Sheet:',s.name
    words = []
    pos = {}
    neg = {}
    for row in range(1, s.nrows):
        col_names = s.row(0)[1:]
        col_value = []
        word = s.cell(row, 0).value
        for name, col in zip(col_names, range(1,s.ncols)):
            value = (s.cell(row,col).value)
            if name.value == 'Positive' and int(value) > 0:
                pos[word] = int(value)
            elif name.value == 'Negative' and int(value) > 0:
                neg[word] = int(value)
            col_value.append((name.value, value))
        values.append(col_value)
mcd = McDonald_Word_List(pos, neg)
print(mcd.pos_words)

import re
import sys

def extract_header(text):
    search = re.search('--(.+?)--(.+?)--(.+?)--(.+?)Reuters\)\s-', text, flags=re.DOTALL)
    text = re.sub('--.+?--.+?--.+?--.+?Reuters\)\s-', '', text, flags=re.DOTALL)
    title = search.group(1)
    author = search.group(2)
    date = search.group(3)
    link = search.group(4)
    return title, author, date, link, text

import os

path = os.getcwd() 

reuters_folders = os.listdir('/home/jmkovachi/Documents/jupyter_notebooks/reuters')[0:10]

path += '/reuters'

articles = []

print(reuters_folders)

count = 0
for folder in reuters_folders:
    article_files = os.listdir(path + '/' + folder)
    sys.stdout.flush()
    for file in article_files:
        count += 1
        with open(path + '/' + folder + '/' + file) as f:
            raw_text = f.read()
            title, author, date, link, text = extract_header(raw_text)
            articles.append({'title' : title, 'author' : author, 'date' : date, 'link' : link, 'text' : text})
            #print(articles[len(articles)-1]['text'] if count % 100 == 0 else '')
            


for article in articles:
    print(article)



import nltk

from nltk.chunk import conlltags2tree, tree2conlltags

sentence = "Mark and John are working at Google."

for sent in nltk.sent_tokenize(sentence):
   for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
      if hasattr(chunk, 'label'):
         print(chunk.label(), ' '.join(c[0] for c in chunk))

import math
import nltk

def num_words(sentences):
    l = 0
    pos_count = 0
    neg_count = 0
    for s in sentences:
        l += len(s)
        for word in nltk.word_tokenize(s):
            if word.upper() in mcd.pos_words:
                pos_count += 1
            elif word.upper() in mcd.neg_words:
                neg_count += 1
    return l, pos_count, neg_count

for article in articles:
    print(article)
    l, pos_count, neg_count = num_words(nltk.sent_tokenize(article['text']))
    print('{} positve, {} negative'.format(pos_count,neg_count))
    
    print('\n\n')

def compute_PMI(class1, class2, int_c1c2, overall_count):
    return math.log((int_c1c2+1/overall_count)/((class1+1/overall_count)*(class2+1/overall_count)))
    # +1s added for smoothing

import pandas as pd
import numpy as np

length = 0
overall_pos = 0
overall_neg = 0
overall_org = 0
intersection_pos = 0
intersection_neg = 0

pos_df = pd.DataFrame(0, index=[str(key) for (key,val) in mcd.pos_words.items()], columns=[])
neg_df = pd.DataFrame(0, index=[str(key) for (key,val) in mcd.neg_words.items()], columns=[])

for article in articles[:1000]:
    sentences = nltk.sent_tokenize(article.text)
    tmpL, tmp_pos, tmp_neg = num_words(sentences)
    length += tmpL
    overall_pos += tmp_pos
    overall_neg += tmp_neg
    for sent in sentences:
       org_count = 0
       pos_count = 0
       neg_count = 0
       org_list = []
       chunks = [chunk for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)))]
       for chunk in chunks:
            if hasattr(chunk, 'label') and str(chunk.label()) == 'ORGANIZATION':
                #print(chunk.label())
                org_count += 1
                overall_org += 1
                org_list.append(str(chunk[0]).upper())
                if str(chunk[0]).upper() not in pos_df.columns:
                    print(str(chunk[0]).upper())
                    pos_df[str(chunk[0]).upper()] = np.zeros(len(pos_df.index))
                    neg_df[str(chunk[0]).upper()] = np.zeros(len(neg_df.index))
                
       tmp_org_count = org_count
       for chunk in chunks:
            #print(chunk[0])
            #print(mcd.pos_words)
            if str(chunk[0]).upper() in mcd.pos_words:
                tmp_org_list = org_list
                #print(chunk[0])
                pos_org_count = tmp_org_count
                while len(tmp_org_list) > 0:
                    pos_count += 1
                    pos_df.at[str(chunk[0]).upper(), tmp_org_list[0]] += 1
                    tmp_org_list.pop(0)
                    mcd.intersection_pos[str(chunk[0]).upper()] += 1
                mcd.pos_word_counts[str(chunk[0]).upper()] += 1
            elif str(chunk[0]).upper() in mcd.neg_words:
                #print(chunk[0])
                tmp_org_list = org_list  
                while(len(tmp_org_list) > 0):
                    neg_count += 1
                    neg_df.at[str(chunk[0]).upper(), tmp_org_list[0]] += 1
                    tmp_org_list.pop(0)
                    mcd.intersection_neg[str(chunk[0]).upper()] += 1
                mcd.neg_word_counts[str(chunk[0]).upper()] += 1
       intersection_pos += org_count if org_count < pos_count else pos_count
       intersection_neg += org_count if org_count < neg_count else neg_count
    #print(pos_count)
    #print(overall_org)
    #print(intersection)
    #print(l)
    
print(compute_PMI(overall_pos, overall_org, intersection_pos, l))
print(compute_PMI(overall_neg, overall_org, intersection_neg, l))
#print(mcd.pos_word_counts)
print(pos_df)


            
                
        
              #print(chunk.label(), ' '.join(c[0] for c in chunk))
    #create_co_occurrence_matrix(sentences)
    

import numpy as np
import matplotlib.pyplot as plt

sorted_counts = sorted(mcd.pos_word_counts.items(), key=lambda kv: kv[1], reverse=True)
print(sorted_counts)
print(sorted_counts[50][0])
print(mcd.intersection_pos[sorted_counts[50][0]])
print(compute_PMI(sorted_counts[100][1], overall_org, mcd.intersection_pos[sorted_counts[100][0]], l))

sorted_counts[0:50]

plt.figure(figsize=(20, 3))  # width:20, height:3
# save the names and their respective scores separately
# reverse the tuples to go from most frequent to least frequent 
plt.bar(range(len(sorted_counts[0:20])), [val[1] for val in sorted_counts[0:20]], align='edge', width=.3)
plt.xticks(range(len(sorted_counts[0:20])), [val[0] for val in sorted_counts[:20]])
plt.xticks(rotation=70)
plt.show()


PMIs = [compute_PMI(count[1], overall_org, mcd.intersection_pos[count[0]], l) for count in sorted_counts[0:20]]

plt.figure(figsize=(20, 3))  # width:20, height:3
# save the names and their respective scores separately
# reverse the tuples to go from most frequent to least frequent 
plt.bar(range(len(sorted_counts[0:20])), PMIs, align='edge', width=.3)
plt.xticks(range(len(sorted_counts[0:20])), [val[0] for val in sorted_count[:20]])
plt.xticks(rotation=70)
plt.show()



