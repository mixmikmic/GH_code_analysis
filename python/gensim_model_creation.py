from __future__ import division
import logging
import pickle
import pandas as pd
from gensim.models.lsimodel import LsiModel
from gensim.corpora import WikiCorpus, MmCorpus, Dictionary

dict_path = '../../data/wiki/frwiki_wordids.txt.bz2'
corpus_path = '../../data/wiki/frwiki_tfidf.mm'
lsi_model_path = '../../data/wiki/frwiki_lsi'
title2id_path = '../../data/wiki/title2id_mapping.pckl'
skills_wiki_path = '../../data/linkedin_skills_wiki.csv'
skills_corpus_path = '../../wiki/data/skills_corpus.json'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dictionary = Dictionary.load_from_text(dict_path)
tfidf_corpus = MmCorpus(corpus_path)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

lsi_model = LsiModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=400)
lsi_model.save(lsi_model_path)

