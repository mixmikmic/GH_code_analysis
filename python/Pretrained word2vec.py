import gensim


# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)  

dog = model['dog']
print(dog.shape)
print(dog[:10])

# 'Dinesh' -> A persons name
dinesh = model['Dinesh']

print(dinesh.shape)
print(dinesh[:10])

# Dealing with word present in the dictionary
if 'Dinesh' in model:
    print("Hurray,This word is present in the dicitonary!")
    print("Shape of model['Dinesh']:",model['Dinesh'].shape)
else:
    print('{0} is an out of dictionary word'.format('Dinesh'))

# Deal with an out of dictionary word: Михаил (Michail)
if 'Михаил' in model:
    print(model['Михаил'].shape)
else:
    print('{0} is an out of dictionary word'.format('Михаил'))

# Some predefined functions that show content related information for given words
print(model.most_similar(positive=['woman', 'king'], negative=['man']))

print(model.doesnt_match("breakfast cereal dinner lunch".split()))

print(model.similarity('woman', 'man'))

print(model.similarity('dog','cat'))

