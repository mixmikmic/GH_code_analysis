import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import gensim

# Load
fn = '../../Academic_Work/PROJECTS/corpora/animal/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(fn, binary=True)

df = pd.read_csv('data/handlabeled_vectors_1k.csv.csv')

class RNNSearch:
    def __init__(self):
        pass
    
    def update_vector(self, item, word):
        weight = 400 # still not enough to find new areas
        oldvec = df.iloc[item][5:]
        to_merge = model[word]*weight
        newvec = np.mean(np.vstack((oldvec[0:300].values, to_merge)), axis=0)
        newvec = np.hstack((newvec, oldvec[300:]))
        return newvec.reshape(1,-1)
        
    def find_similarities(self, targetvec):
        simvec = cdist(df[df.columns[5:]].values, targetvec, metric='cosine').reshape(-1)
        return pd.Series(1-simvec)   
    
    def search(self, item_id, modification, k):
        newvec = self.update_vector(item_id, modification)
        simvec = self.find_similarities(newvec)
        answers = df.id[ simvec.sort_values(ascending=False).head(k+1).index[1:]]
        return answers
    

word='casual'
item=7063
k = 10
search_model = RNNSearch()
search_model.search(item, word,k)







