import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.Word2Vec.load_word2vec_format('Data/GoogleNews-vectors-negative300.bin', binary=True)

model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)

model.most_similar(['girl', 'father'], ['boy'], topn=3)

model.doesnt_match("breakfast cereal dinner lunch".split())

sentences = [['cigarette','smoking','is','injurious', 'to', 'health'],['cigarette','smoking','causes','cancer'],['cigarette','are','not','to','be','sold','to','kids']]

# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1, sg=1, window = 3)

model.most_similar(positive=['cigarette', 'smoking'], negative=['kids'], topn=1)



