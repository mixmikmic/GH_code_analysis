import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = []
with open('../data/corpora/enlang1.txt') as f:
    for line in f.readlines():
        sentences.append(line.strip().split())

model = gensim.models.Word2Vec(sentences, size = 50, min_count=3)

print(model.wv['car'])

model.wv.most_similar(positive=['cars', 'bus'], negative=['car'])

from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('../data/crawl-300.vec', binary=False) 

word_vectors.most_similar(positive=['kings', 'queen'], negative=['king'])

word_vectors.most_similar(positive=['woman', 'husband'], negative=['man'])

word_vectors.most_similar(positive=['Paris', 'Spain'], negative=['France'])

word_vectors.most_similar(positive=['Donald', 'Putin'], negative=['Trump'])

