import gensim, logging 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['first', 'sentence'], ['second', 'sentence']]

model = gensim.models.Word2Vec(sentences, min_count=1)

import os 

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname 
        
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
                
sentences = MySentences('/some/directory')
model = gensim.models.Word2Vec(sentences)

model = gensim.models.Word2Vec(iter=1) # an empty model 
model.build_vocab(some_sentences)
model.train(other_sentences)

