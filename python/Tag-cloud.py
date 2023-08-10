import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from math import floor

words_list= []
with open('Tag-cloud_Passau.txt', 'r') as f:
    for word in f.read().split():
           words_list.append(word)

# Tag-cloud_Passauer-wolf.jpg from: https://de.wikipedia.org/wiki/Passauer_Wolf adapted as mask image

words_string=""
words_string= " ".join(words_list)

k= floor(len(words_list)/3)
mask = np.array(Image.open("Tag-cloud_Passauer-wolf.jpg"))

wordcloud = WordCloud(    stopwords=STOPWORDS,
                          background_color='black',
                          max_words=k,
                          mask=mask,
                         ).generate(words_string)
plt.figure(figsize=(30,25))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

import nltk 
nltk.download('maxent_treebank_pos_tagger')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def processing(string):
    #remove punctuation and special characters
    string = re.sub(r'([^\s\w]|_)+', '', string)
    return string

#Apply POS tagging and generate the list of nouns and adjectives
postag=[]
tag=[]
words=[]
noun_adj= []

words_list = (processing(words_string)).split() 
postag = nltk.pos_tag(words_list)
for i in postag:
     for j in i[::2]:
            words.append(j)
for i in postag:
     for j in i[1::2]:
            tag.append(j)
#Uncomment the line above when using Python 2
#for (i, j) in itertools.izip(words,tag):
#Comment the line above when using Python 2 
for (i, j) in zip(words,tag):
    if (j == "JJ")| (j == "NN") | (j == "NNP") :
        noun_adj.append(i)       

# Generate a tag cloud of nouns and adjectives with the most k frequent terms.

# Tag-cloud_Passauer-wolf.jpg from: https://de.wikipedia.org/wiki/Passauer_Wolf adapted as mask image

noun_adj_string= ""
noun_adj_string= " ".join(noun_adj)

k= floor(len(noun_adj )/3)
mask = np.array(Image.open("Tag-cloud_Passauer-wolf.jpg"))
wordcloud = WordCloud(    stopwords=STOPWORDS,
                          background_color='black',
                          max_words=k,
                          mask=mask,
                         ).generate(noun_adj_string)
plt.figure(figsize=(25,20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

