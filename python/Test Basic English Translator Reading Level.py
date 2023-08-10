# Loding in my articles, saved in the main running of BasicEnglishTranslator.py
import cPickle as pickle
articles = pickle.load(open('../data/articles.pickle', 'rb'))
keys = articles.keys()

si_entries = []
en_entries = []

bad = 'we could not find the above page on our servers. did you mean'
worse = 'wikipedia does not yet have an article with this name'

not_blank = 0
all_art = 0
for key in articles.keys():
    not_blank += 1
    if (bad in articles[key][2] or worse in articles[key][2] or 'mean:' in articles[key][2] or 
        'wikipedia help' in articles[key][2] or 'refer to' in articles[key][2] or 'may be about' in articles[key][2]):
        pass
    else:
        if key[:2] == 'en':
            en_entries.append((articles[key][0], articles[key][2]))
        elif key[:2] == 'si':
            si_entries.append((articles[key][0], articles[key][2]))

print 'We have {} articles (of {} non-blank).'.format(len(si_entries) + len(en_entries), not_blank)

def count_syllables(word):
    count = 0
    vowels = 'aeiouy'
    word = word.lower().strip(".:;?!")
    try:
        if word[0] in vowels:
            count +=1
        for index in range(1,len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count +=1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count+=1
        if count == 0:
            count +=1
        return count
    except:
        return 0

from nltk.corpus import cmudict
from nltk import pos_tag, word_tokenize
import string
import re

def calc_fk(text):
    '''
    Flesch-Kincaid gives the approximate US School reading level of text:
             0.39 (words/sentences) + 11.8(syllables/words) - 115.59
    '''
    punctuation = ".?!"
    prt = set(string.printable)
    if type(text) ==  list:
        text = ' '.join(text)
    text = filter(lambda x: x in prt, text)
    text = text.encode('utf-8')
    text = re.sub("\xe2\x80\x93", "-", text)
    sentences = 0
    syllables = 0
    n_words = 0
    words = pos_tag(word_tokenize(text))
    text = text
    for word in words:
        if word[1] in punctuation:
            sentences += 1.0
        elif len(word[1]) >= 2:
            syllables += float(count_syllables(word[0]))
            n_words += 1.0
    else:
        if sentences == 0 or sentences == 0.0:
            sentences = 1.0
        try:
            return 0.39 * (n_words/sentences) + 11.8 * (syllables/n_words) - 15.59
        except:
            print sentences, n_words

import numpy as np

Xb = []
yb = []
Xr = []
yr = []
for text in en_entries:
    Xr.append([calc_fk(text[0]), calc_fk([text[1]])])
    yr.append('r')

for text in si_entries:
    Xb.append([calc_fk(text[0]), calc_fk([text[1]])])
    yb.append('b')
    
Xb = np.array(Xb)
Xr = np.array(Xr)

# find most-simplified and average simplification
best = ('a','122')
total = []
for i, x in enumerate(Xb):
    new = x[0] - x[1]
    total.append(new)
    if new < best[1]:
        best = (si_entries[i], new)

blue = np.mean(total)
total = []

for i, x in enumerate(Xr):
    new = x[0] - x[1]
    total.append(new)
    if new < best[1]:
        best = (en_entries[i], new)
        
print best
red = np.mean(total)

import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')

font = {'family' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

plt.figure(figsize=(10, 10))
plt.title("Flesch-Kincaid Reading Levels")
plt.xlabel('Flesch-Kincaid Reading Level of Original Article')
plt.ylabel('Flesch-Kincaid Reading Level of Translated Article')
xx = np.linspace(0.0, 18, 19)
yy = xx
plt.xlim(0, 18)
plt.ylim(0, 18)
plt.plot(xx, yy, c='k', alpha=0.8)
plt.scatter(Xb[:, 1], Xb[:, 0], label='Simple Wikipedia ({:.2f})'.format(blue), c=yb, s=50, alpha=0.25)
plt.scatter(Xr[:, 1], Xr[:, 0], label='Wikipedia ({:.2f})'.format(red), c=yr, s=50, alpha=0.25)
# plt.scatter(Xr[:, 1], Xr[:, 0], c=yr, s=100, alpha=0.25)
#plt.plot(xx, yy+1, c='k', alpha=0.4, label='+/- 1 Grade Level')
plt.plot(xx, yy-1, c='k', alpha=0.4)
plt.plot(xx, yy+1, c='k', alpha=0.4)
# plt.text(1, 2.35, '+1')
# plt.text(1, .25, '-1')
#plt.text(0.25, 17, r'$\Delta$GL$_A$$_V$$_E$ = ${:.2f}$'.format(red))
plt.legend()
plt.show()

plt.figure(figsize=(10, 10))
plt.title(r"$\Delta$Flesch-Kincaid Reading Levels")
plt.xlabel('Original Flesch-Kincaid Reading Level of Article')
plt.ylabel('Change in Flesch-Kincaid Reading Level')
xx = np.linspace(0.0, 18, 19)
yy = xx*0.0
plt.xlim(0, 18)
plt.ylim(-3, 3)

plt.plot(xx, yy, c='k', alpha=0.8)
plt.scatter(Xb[:, 1], Xb[:, 0] - Xb[:, 1], label='Simple Wikipedia ({:.2f})'.format(blue), c=yb, s=50, alpha=0.25)
plt.scatter(Xr[:, 1], Xr[:, 0] - Xr[:, 1], c=yr, s=100, alpha=0.25)
#plt.scatter(Xr[:, 1], Xr[:, 0] - Xr[:, 1], c=yr, s=100, alpha=0.25)
#plt.plot(xx, yy+1, c='k', alpha=0.4, label='+/- 1 Grade Level')
plt.plot(xx, yy-1, c='k', alpha=0.4)
plt.plot(xx, yy+1, c='k', alpha=0.4)
# plt.text(xx[3], yy[-3], r'$\Delta$GL = ${:.2f}$'.format(red))
# plt.text(1, .25, '-1')
plt.legend()
plt.show()





