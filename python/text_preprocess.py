txt_path = 'data/aclImdb/train/neg/10_2.txt'
txt = open(txt_path).read()

txt

from bs4 import BeautifulSoup

example1 = BeautifulSoup(txt, 'lxml')

#print train["review"][0]
print example1.get_text()

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = tokenizer.tokenize(example1.get_text().strip().decode('utf-8'))

sentences

import re

letters_only = re.sub("[^a-zA-Z]", " ", example1.get_text())

letters_only

words = letters_only.lower().split()
words = [w.strip() for w in words]

from nltk.corpus import stopwords

print(len(words))
words = [w for w in words if not w in stopwords.words("english")]
print(len(words))

' '.join(words)

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

clean_words = [wordnet_lemmatizer.lemmatize(w) for w in words]

' '.join(clean_words)

get_ipython().magic('time word_pos = nltk.pos_tag(words)')

def pos_tag_map(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

get_ipython().magic('time clean_words = [wordnet_lemmatizer.lemmatize(w, pos_tag_map(pos)) for (w, pos) in word_pos]')

' '.join(clean_words)

' '.join(words)

