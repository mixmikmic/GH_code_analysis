from srs.utilities import Product, AspectPattern

# load training data
product_name = 'Canon'
reviewTrainingFile = product_name + '.txt'
product = Product(name=product_name)
product.loadReviewsFromTrainingFile('../data/trainingFiles/Liu/' + reviewTrainingFile)

aspectPatterns = []
# define an aspect pattern1
pattern_name = 'adj_nn'
pattern_structure ="""
adj_nn:{<JJ><NN.?>}
"""
aspectTagIndices = [1]
aspectPattern = AspectPattern(name='adj_nn', structure=pattern_structure, aspectTagIndices=aspectTagIndices)
aspectPatterns.append(aspectPattern)
# define an aspect pattern2
pattern_name = 'nn_nn'
pattern_structure ="""
nn_nn:{<NN.?><NN.?>}
"""
aspectTagIndices = [0,1]
aspectPattern = AspectPattern(name='nn_nn', structure=pattern_structure, aspectTagIndices=aspectTagIndices)
aspectPatterns.append(aspectPattern)

# pos tagging
for review in product.reviews:
    for sentence in review.sentences:
        sentence.pos_tag()
        sentence.matchDaynamicAspectPatterns(aspectPatterns)

word_dict = {}
for review in product.reviews:
    for sentence in review.sentences:
        for aspect in sentence.dynamic_aspects:
            if aspect in word_dict:
                word_dict[aspect] += 1
            else:
                word_dict[aspect] = 1

word_sorted = sorted(word_dict.items(), key=lambda tup:-tup[1])
word_sorted[:15]

import json
word_output = open('../data/word_list/{0}_wordlist.txt'.format(product_name), 'w')
json.dump(word_sorted[:15], word_output)
word_output.close()

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

# collect word with same stem
stemmedWord_dict = {}
for word in word_dict:
    stemmedWord = stemmer.stem(word)
    if stemmedWord in stemmedWord_dict:
        stemmedWord_dict[stemmedWord] += word_dict[word]
    else:
        stemmedWord_dict[stemmedWord] = word_dict[word]

# frequency ranking
stemmedWord_sorted = sorted(stemmedWord_dict.items(), key=lambda tup:-tup[1])
stemmedWord_sorted[:15]

# save most frequent stemmed words
stemmedWord_output = open('../data/word_list/{0}_stemmedwordlist.txt'.format(product_name), 'w')
json.dump(stemmedWord_sorted[:15], stemmedWord_output)
stemmedWord_output.close()



