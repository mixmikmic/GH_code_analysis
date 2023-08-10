from IPython.core.debugger import Tracer; debug = Tracer()
import numpy as np
import pandas as pd
trainw2v=pd.read_csv('data/train-w2v.csv',sep="\t")

from nltk.tokenize import word_tokenize
tokenized_sentences=trainw2v.apply(lambda row: word_tokenize(row['Phrase']), axis=1)



import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.Word2Vec(tokenized_sentences, min_count=1)
model.save('w2vmodel')

print("done succesfully")

