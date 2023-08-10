import numpy as np
import re

abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
print('There are %d characters in abc.'%len(abc))

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ",raw)
    words = clean.split()
    return words

book=open('./data/dorian.txt','r')
list_of_words=sentence_to_wordlist(book.read())

list_of_words[1313]

len(list_of_words)

cooc = np.zeros((52,52),np.float64)

def update_cooc(word):
    for char1 in word:
        for char2 in word:
            if char1!=char2:
                one_hot_1 = abc.index(char1)
                one_hot_2 = abc.index(char2)
                cooc[one_hot_1,one_hot_2]+=1

for word in list_of_words:
    update_cooc(word)

cooc

cooc[26,51]

cooc[abc.index('t'),abc.index('h')]

book=open('./data/dorian.txt','r')
book_string= book.read().lower()

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ",raw)
    words = clean.split()
    return words

list_of_words=list(set(sentence_to_wordlist(book_string)))

len(list_of_words)

from nltk.tokenize import sent_tokenize
list_of_sentences=sent_tokenize(book_string)

list_of_sentences[20:22]

cooc = np.zeros((7122,7122),np.float64)

def process_sentence(sentence):
    words_in_sentence =sentence_to_wordlist(sentence)
    list_of_indeces = [list_of_words.index(word) for word in words_in_sentence]
    for index1 in list_of_indeces:
        for index2 in list_of_indeces:
            if index1!=index2:
                cooc[index1,index2]+=1
                    

for sentence in list_of_sentences:
    process_sentence(sentence)

cooc

print('The 16th word is:',list_of_words[15])
for j in range(7122):
    if cooc[15,j]>3 and cooc[15,j]<6:
        print(list_of_words[15],list_of_words[j],cooc[15,j])

from numpy.linalg import norm
def cos_dis(u,v):
    dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
    return dist

sorted(list_of_words, key = lambda word: cos_dis(cooc[15,:],cooc[list_of_words.index(word),:]))

from numpy.linalg import svd

U,S,V=svd(cooc)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(S)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(S[:40])
plt.show()

emb=U[:,:40]

emb[15,:]

sorted(list_of_words, key = lambda word: cos_dis(emb[15,:],emb[list_of_words.index(word),:]))

