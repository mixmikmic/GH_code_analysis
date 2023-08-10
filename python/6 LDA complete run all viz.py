import itertools
import logging
import os
import pickle
import time

from cltk.stop.greek.stops import STOPS_LIST
import gensim
from gensim.corpora.mmcorpus import MmCorpus
from gensim.utils import simple_preprocess
import numpy as np
import pyLDAvis.gensim

pyLDAvis.enable_notebook()

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

user_dir = os.path.expanduser('~/cltk_data/user_data/lda_tlg/')
try:
    os.makedirs(user_dir)
except FileExistsError:
    pass

# load bow dict
bow_name = 'gensim_bow_tlg_nobelow20_noabove0.1_tokmin3_tokmax20_docmin50_deaccentFalse.mm'
bow_path = os.path.join(user_dir, bow_name)
corpus_bow_tlg = gensim.corpora.MmCorpus(bow_path)

PREPROCESS_DEACCENT = False
no_below = 20
no_above = 0.1
NUM_TOPICS_LIST = [5, 10, 20, 40, 60, 120]
PASSES = 100

TOK_MIN = 3  # rm words shorter than
TOK_MAX = 20  # rm words longer than
DOC_MIN = 50  # drop docs shorter than

dict_name = 'gensim_dict_id2word_tlg_nobelow{}_noabove{}_tokmin{}_tokmax{}_docmin{}_deaccent{}.dict'.format(no_below, 
                                                                                                            no_above, 
                                                                                                            TOK_MIN, 
                                                                                                            TOK_MAX, 
                                                                                                            DOC_MIN, 
                                                                                                            PREPROCESS_DEACCENT)
dict_path = os.path.join(user_dir, dict_name)

id2word_tlg = gensim.corpora.dictionary.Dictionary.load(dict_path)

# # Examples of how to use the model
# lda_model.print_topics(-1)  # print a few most important words for each LDA topic
# # transform text into the bag-of-words space
# bow_vector = id2word_tlg.doc2bow(tokenize(doc))
# print([(id2word_tlg[id], count) for id, count in bow_vector])

# # transform into LDA space
# lda_vector = lda_model[bow_vector]
# print(lda_vector)

# # print the document's single most prominent LDA topic
# print(lda_model.print_topic(max(lda_vector, key=lambda item: item[1])[0]))

all_paths = []
for num_topics in NUM_TOPICS_LIST:
    lda_model_name = 'gensim_lda_model_tlg_numtopics{}_numpasses{}_nobelow{}_noabove{}_tokmin{}_tokmax{}_docmin{}_deaccent{}.model'.format(num_topics, 
                                                                                                                                           PASSES, 
                                                                                                                                           no_below, 
                                                                                                                                           no_above, 
                                                                                                                                           TOK_MIN, 
                                                                                                                                           TOK_MAX, 
                                                                                                                                           DOC_MIN, 
                                                                                                                                           PREPROCESS_DEACCENT)
    path_lda = os.path.join(user_dir, lda_model_name)
    all_paths.append(path_lda)

def load_lda_model(path_lda):
    lda_model = gensim.models.LdaMulticore.load(path_lda)
    return lda_model

all_paths

lda_path5 = '/home/kyle/cltk_data/user_data/lda_tlg/gensim_lda_model_tlg_numtopics5_numpasses100_nobelow20_noabove0.1_tokmin3_tokmax20_docmin50_deaccentFalse.model'

lda_model = load_lda_model(lda_path5)
lda_model.show_topics()

pyLDAvis.gensim.prepare(lda_model, corpus_bow_tlg, id2word_tlg)

lda_path10 = '/home/kyle/cltk_data/user_data/lda_tlg/gensim_lda_model_tlg_numtopics10_numpasses100_nobelow20_noabove0.1_tokmin3_tokmax20_docmin50_deaccentFalse.model'
lda_model = load_lda_model(lda_path10)
lda_model.show_topics()

pyLDAvis.gensim.prepare(lda_model, corpus_bow_tlg, id2word_tlg)

lda_path20 = '/home/kyle/cltk_data/user_data/lda_tlg/gensim_lda_model_tlg_numtopics20_numpasses100_nobelow20_noabove0.1_tokmin3_tokmax20_docmin50_deaccentFalse.model'
lda_model = load_lda_model(lda_path20)
lda_model.show_topics()

pyLDAvis.gensim.prepare(lda_model, corpus_bow_tlg, id2word_tlg)

lda_path40 = '/home/kyle/cltk_data/user_data/lda_tlg/gensim_lda_model_tlg_numtopics40_numpasses100_nobelow20_noabove0.1_tokmin3_tokmax20_docmin50_deaccentFalse.model'
lda_model = load_lda_model(lda_path40)
lda_model.show_topics()

pyLDAvis.gensim.prepare(lda_model, corpus_bow_tlg, id2word_tlg)

lda_path60 = '/home/kyle/cltk_data/user_data/lda_tlg/gensim_lda_model_tlg_numtopics60_numpasses100_nobelow20_noabove0.1_tokmin3_tokmax20_docmin50_deaccentFalse.model'
lda_model = load_lda_model(lda_path60)
lda_model.show_topics()

pyLDAvis.gensim.prepare(lda_model, corpus_bow_tlg, id2word_tlg)

lda_path120 = '/home/kyle/cltk_data/user_data/lda_tlg/gensim_lda_model_tlg_numtopics120_numpasses100_nobelow20_noabove0.1_tokmin3_tokmax20_docmin50_deaccentFalse.model'
lda_model = load_lda_model(lda_path120)
lda_model.show_topics()

pyLDAvis.gensim.prepare(lda_model, corpus_bow_tlg, id2word_tlg)

