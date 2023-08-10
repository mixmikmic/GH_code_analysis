from gensim import corpora, models
import pyLDAvis.gensim
import joblib
import pickle

corpus = corpora.MmCorpus('lda/lda_middle_east.mm')
dictionary = corpora.Dictionary.load('lda/lda_middle_east.dict')
lda = models.LdaModel.load('lda/middle_east_100.lda')
print("Loaded Model")
followers_data =  pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.displaym(followers_data)

