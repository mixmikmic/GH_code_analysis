import numpy as np
from scipy import spatial # for the cosine distance
import lingualTF as laT # to get our tokenizing function
import sys

# make this the full path to you prototype_python directory
sys.path.append('/Users/Seth/Documents/DSI/Capstone/DSI-Religion-2017/prototype_python/')
import lingual as la

# get 300 dim preset embeddings
filename = 'glove.6B/glove.6B.300d.txt'
#filename = '/Users/Seth/Documents/DSI/Capstone/big-data/glove.6B/glove.6B.300d.txt'

# transform file into a dictionary where key = word, value = embedding array
words = open(filename).read().splitlines()
embedDict = { }
for word in words:
    thisword = word.split(" ")
    embedDict[thisword[0]] = np.array([float(x) for x in thisword[1:]])

queentest = embedDict['king'] - embedDict['man'] + embedDict['woman']

spatial.distance.cosine(embedDict['car'], queentest)

# tokenize sentences
sentence1 = laT.cleanTokens(laT.rawToTokenList('Our sentence is cool.'), stem = False)
sentence2 = laT.cleanTokens(laT.rawToTokenList('Our sentence is radical.'), stem = False)
sentence3 = laT.cleanTokens(laT.rawToTokenList('Quick brown fox jumped over the dog.'), stem = False)

# get sentence vectors
sent1Vect = [embedDict[word] for word in sentence1]
sent2Vect = [embedDict[word] for word in sentence2]
sent3Vect = [embedDict[word] for word in sentence3]

# average the vectors
sent1done = np.mean(sent1Vect, axis = 0)
sent2done = np.mean(sent2Vect, axis = 0)
sent3done = np.mean(sent3Vect, axis = 0)

spatial.distance.cosine(sent1done, sent3done)



self = la.lingualObject(['sampleText/ACLU/raw/ACLU03.txt'])

self.setKeywords()
self.keywords



