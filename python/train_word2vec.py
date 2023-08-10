import numpy as np
import pandas as pd
import matplotlib as plt
get_ipython().magic('matplotlib inline')

from gensim.models import word2vec

with_time = False
if not with_time:
    corpus = pd.read_csv("./cleaned_data/all_events_data_mv.csv", dtype = str)
    corpus_2 = pd.read_csv("./cleaned_data/all_events_data.csv", dtype = str)
else:
    corpus = pd.read_csv("./cleaned_data/all_events_data_w_time_mv.csv", dtype = str)
    corpus_2 = pd.read_csv("./cleaned_data/all_events_data_w_time.csv", dtype = str)

#transfrom it to sentences format
a = corpus.groupby("SUBJECT_ID").apply(lambda x: x.EVE_INDEX.tolist())

a_2 = corpus_2.groupby("SUBJECT_ID").apply(lambda x: x.EVE_INDEX.tolist())

sentence_list = a.values.tolist()
sentence_list2 = a_2.values.tolist()

model = word2vec.Word2Vec(sentence_list, size=100, window=20, min_count=1, workers=2,sg=1)

if not with_time:
    model.save("./word2vec_model/w2vmodel_mv")
else:
    model.save("./word2vec_model/w2vmodel_mv_wt")

model = word2vec.Word2Vec.load("./word2vec_model/w2vmodel_mv")

#10646 is very similar to 10648 and 10649 which are drugs all belong to triamcinolone acetonide familiy
model.most_similar('4000')

model2 = word2vec.Word2Vec(sentence_list2, size=100, window=20, min_count=1, workers=4,sg=1)

if not with_time:
    model2.save("./word2vec_model/w2vmodel")
else:
    model2.save("./word2vec_model/w2vmodel_wt")

model2 = word2vec.Word2Vec.load("./word2vec_model/w2vmodel_wt")

model2.most_similar('4382')



