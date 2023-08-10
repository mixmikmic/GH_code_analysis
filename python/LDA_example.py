import logging

import numpy as np
from ptm import GibbsLDA
from ptm import vbLDA
from ptm.nltk_corpus import get_reuters_ids_cnt
from ptm.utils import convert_cnt_to_list, get_top_words

n_doc = 1000
voca, doc_ids, doc_cnt = get_reuters_ids_cnt(num_doc=n_doc, max_voca=10000)
docs = convert_cnt_to_list(doc_ids, doc_cnt)
n_voca = len(voca)
print('Vocabulary size:%d' % n_voca)

max_iter=100
n_topic=10

logger = logging.getLogger('GibbsLDA')
logger.propagate = False

model = GibbsLDA(n_doc, len(voca), n_topic)
model.fit(docs, max_iter=max_iter)

for ti in range(n_topic):
    top_words = get_top_words(model.TW, voca, ti, n_words=10)
    print('Topic', ti ,': ', ','.join(top_words))

logger = logging.getLogger('vbLDA')
logger.propagate = False

vbmodel = vbLDA(n_doc, n_voca, n_topic)
vbmodel.fit(doc_ids, doc_cnt, max_iter=max_iter)

for ti in range(n_topic):
    top_words = get_top_words(vbmodel._lambda, voca, ti, n_words=10)
    print('Topic', ti ,': ', ','.join(top_words))



