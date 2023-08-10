import gensim.models
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec


model = Word2Vec.load('400dm_by_5lac_yelp.model')

word1 = "shake"
word2 = "drinks"
model.most_similar(word1),"  -----------------------------  ",model.most_similar(word2)

word = 'drink'
model[word]



