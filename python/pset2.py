import numpy as np
from collections import defaultdict, Counter

from gtnlplib import preproc, viterbi, most_common, clf_base
from gtnlplib import naive_bayes, scorer, constants, tagger_base, hmm

import matplotlib.pyplot as plt
# this enables you to create inline plots in the notebook 
get_ipython().run_line_magic('matplotlib', 'inline')

## Define the file names
TRAIN_FILE = constants.TRAIN_FILE
print TRAIN_FILE
DEV_FILE = constants.DEV_FILE
TEST_FILE = constants.TEST_FILE # You do not have this for now

## Demo
all_tags = set()
for i,(words, tags) in enumerate(preproc.conll_seq_generator(TRAIN_FILE,max_insts=100000)):
    for tag in tags:
        all_tags.add(tag)
print all_tags

## Demo
all_tags = set()
for i,(words, tags) in enumerate(preproc.conll_seq_generator(TRAIN_FILE,max_insts=100000)):
    for tag in tags:
        all_tags.add(tag)
print all_tags

reload(most_common);

# this block uses your code to find the most common words per tag
counters = most_common.get_tag_word_counts(TRAIN_FILE)
for tag,tag_ctr in counters.iteritems():
    print tag,tag_ctr.most_common(3)

reload(tagger_base);

# here is a tagger that just tags everything as a noun
noun_tagger = lambda words, alltags : ['NOUN' for word in words]
confusion = tagger_base.eval_tagger(noun_tagger,'nouns',all_tags=all_tags)
print scorer.accuracy(confusion)

reload(tagger_base);
reload(clf_base);
reload(most_common);
reload(naive_bayes);

classifier_noun_tagger = tagger_base.make_classifier_tagger(most_common.get_noun_weights())

confusion = tagger_base.eval_tagger(classifier_noun_tagger,'all-nouns.preds',all_tags=all_tags)
print scorer.accuracy(confusion)

theta_mc = most_common.get_most_common_word_weights(constants.TRAIN_FILE)

tagger_mc = tagger_base.make_classifier_tagger(theta_mc)

tags = tagger_mc(['They','can','can','fish'],all_tags)
print tags

tags = tagger_mc(['The','old','man','the','boat','.'],all_tags)
print tags

confusion = tagger_base.eval_tagger(tagger_mc,'most-common.preds',all_tags=all_tags)
print scorer.accuracy(confusion)

sorted_tags = sorted(counters.keys())
print ' '.join(sorted_tags)

nb_weights = naive_bayes.estimate_nb([counters[tag] for tag in sorted_tags],
                                     sorted_tags,
                                     .01)

print nb_weights[('ADJ','bad')], nb_weights[(u'ADJ',u'good')]
print nb_weights[(u'PRON',u'they')], nb_weights[(u'PRON',u'They')], nb_weights[(u'PRON',u'good')]
print nb_weights[(u'PRON',u'.')], nb_weights[(u'NOUN',u'.')], nb_weights[(u'PUNCT',u'.')]

vocab = set([word for tag,word in nb_weights.keys() if word is not constants.OFFSET])

print sum(np.exp(nb_weights[('ADJ',word)]) for word in vocab)
print sum(np.exp(nb_weights[('DET',word)]) for word in vocab)
print sum(np.exp(nb_weights[('PUNCT',word)]) for word in vocab)

print nb_weights[('ADJ','baaaaaaaaad')]

print nb_weights[('VERB'),constants.OFFSET]
print nb_weights[('ADV'),constants.OFFSET]
print nb_weights[('PRON'),constants.OFFSET]

confusion = tagger_base.eval_tagger(tagger_base.make_classifier_tagger(nb_weights),'nb-simple.preds')
dev_acc = scorer.accuracy(confusion)
print dev_acc

reload(naive_bayes);

theta_nb_fixed = naive_bayes.estimate_nb_tagger(counters,.01)

# emission weights still sum to one 
print sum(np.exp(theta_nb_fixed[('ADJ',word)]) for word in vocab)

# emission weights are identical to theta_nb
print nb_weights[('ADJ','okay')],theta_nb_fixed[('ADJ','okay')]

# but the offsets are now correct
print theta_nb_fixed[('VERB'),constants.OFFSET]
print theta_nb_fixed[('ADV'),constants.OFFSET]
print theta_nb_fixed[('PRON'),constants.OFFSET]

sum(np.exp(theta_nb_fixed[(tag,constants.OFFSET)]) for tag in all_tags)

confusion = tagger_base.eval_tagger(tagger_base.make_classifier_tagger(theta_nb_fixed),
                                    'nb-fixed.preds')
dev_acc = scorer.accuracy(confusion)
print dev_acc

START_TAG = constants.START_TAG
TRANS = constants.TRANS
END_TAG = constants.END_TAG
EMIT = constants.EMIT

hand_weights = {('NOUN','they',EMIT):-1,                ('NOUN','can',EMIT):-3,                ('NOUN','fish',EMIT):-3,                ('VERB','they',EMIT):-11,                ('VERB','can',EMIT):-2,                ('VERB','fish',EMIT):-4,                ('NOUN','NOUN',TRANS):-5,                ('VERB','NOUN',TRANS):-2,                (END_TAG,'NOUN',TRANS):-2,                ('NOUN','VERB',TRANS):-1,                ('VERB','VERB',TRANS):-3,                (END_TAG,'VERB',TRANS):-3,                ('NOUN',START_TAG,TRANS):-1,                ('VERB',START_TAG,TRANS):-2}

reload(hmm);

sentence = "they can can fish".split()
print sentence

print hmm.hmm_features(sentence,'Noun','Verb',2)
print hmm.hmm_features(sentence,'Noun',START_TAG,0)
print hmm.hmm_features(sentence,END_TAG,'Verb',4)

reload(viterbi);

print viterbi.viterbi_step('NOUN',0,sentence,hmm.hmm_features,hand_weights,{START_TAG:0})
print viterbi.viterbi_step('VERB',0,sentence,hmm.hmm_features,hand_weights,{START_TAG:0})

print viterbi.viterbi_step('NOUN',1,sentence,
                           hmm.hmm_features,
                           hand_weights,
                           {'NOUN':-2,'VERB':-13})
print viterbi.viterbi_step('VERB',1,sentence,
                           hmm.hmm_features,
                           hand_weights,
                           {'NOUN':-2,'VERB':-13})

reload(viterbi);

all_tags = ['NOUN','VERB']

# let's change the weights a little
hand_weights['NOUN','they',EMIT] = -2
hand_weights['VERB','fish',EMIT] = -5
hand_weights['VERB','VERB',TRANS] = -2

trellis = viterbi.build_trellis(sentence,hmm.hmm_features,hand_weights,all_tags)

for line in trellis:
    print line

reload(viterbi);

viterbi.viterbi_tagger(sentence,hmm.hmm_features,hand_weights,all_tags)

viterbi.viterbi_tagger(['they','can','can','can','can','can','fish'],
                       hmm.hmm_features,hand_weights,all_tags)

tag_trans_counts = most_common.get_tag_trans_counts(TRAIN_FILE)

print tag_trans_counts['DET']
print tag_trans_counts[START_TAG]

reload(hmm);

hmm_trans_weights = hmm.compute_transition_weights(tag_trans_counts,.001)

print tag_trans_counts[START_TAG]['NOUN'], hmm_trans_weights[('NOUN',START_TAG,TRANS)]
print tag_trans_counts[START_TAG]['VERB'], hmm_trans_weights[('VERB',START_TAG,TRANS)]
print tag_trans_counts['DET']['VERB'], hmm_trans_weights[('VERB','DET',TRANS)]
print tag_trans_counts['DET']['INTJ'], hmm_trans_weights[('INTJ','DET',TRANS)]
print tag_trans_counts['DET']['NOUN'], hmm_trans_weights[('NOUN','DET',TRANS)]
print tag_trans_counts['VERB'][START_TAG], hmm_trans_weights[(START_TAG,'VERB',TRANS)]
#print tag_trans_counts[END_TAG]['VERB'] # will throw key error
print hmm_trans_weights[('VERB',END_TAG,TRANS)]

all_tags = tag_trans_counts.keys() + [END_TAG]
print sum(np.exp(hmm_trans_weights[(tag,'NOUN',TRANS)]) for tag in all_tags)
print sum(np.exp(hmm_trans_weights[(tag,'SYM',TRANS)]) for tag in all_tags)

reload(hmm);

theta_hmm,_ = hmm.compute_HMM_weights(TRAIN_FILE,.01)

print theta_hmm['NOUN','right',EMIT], theta_hmm['ADV','right',EMIT]
print theta_hmm['PRON','she',EMIT], theta_hmm['DET','she',EMIT]
print theta_hmm['NOUN','notarealword',EMIT]

len(theta_hmm)

print sum(np.exp(theta_hmm['NOUN',word,EMIT]) for word in vocab)
print sum(np.exp(theta_hmm['DET',word,EMIT]) for word in vocab)

reload(viterbi)
viterbi.viterbi_tagger(['they', 'can', 'can', 'fish'],hmm.hmm_features,theta_hmm,all_tags)

print theta_hmm['NOUN','right',EMIT], theta_hmm['ADV','right',EMIT]
print theta_hmm['PRON','she',EMIT], theta_hmm['DET','she',EMIT]
print theta_hmm['ADJ','thisworddoesnotappear',EMIT]

# this is just for fun
for i,(words,_) in enumerate(preproc.conll_seq_generator(DEV_FILE)):
    print i, 
    pred_tags = viterbi.viterbi_tagger(words,hmm.hmm_features,theta_hmm,all_tags)[0]
    for word,pred_tag in zip(words,pred_tags):
        print "%s/%s"%(word,pred_tag),
    print
    if i >= 2: break

tagger = lambda words, all_tags : viterbi.viterbi_tagger(words,
                                                         hmm.hmm_features,
                                                         theta_hmm,
                                                         all_tags)[0]
confusion = tagger_base.eval_tagger(tagger,'hmm-dev-en.preds')

print scorer.accuracy(confusion)

tagger_base.apply_tagger(tagger,'hmm-te-en.preds',testfile=constants.TEST_FILE_UNLABELED)

# you don't have en-ud-test.conllu, so you can't run this
te_confusion = scorer.get_confusion('data/en-ud-test.conllu','hmm-te-en.preds')
print scorer.accuracy(te_confusion)

from gtnlplib.constants import JA_TRAIN_FILE, JA_DEV_FILE, JA_TEST_FILE_HIDDEN, JA_TEST_FILE
from gtnlplib.constants import TRAIN_FILE, START_TAG, END_TAG

tag_trans_counts_en = most_common.get_tag_trans_counts(TRAIN_FILE)

tag_trans_counts_ja = most_common.get_tag_trans_counts(JA_TRAIN_FILE)

print tag_trans_counts_en['VERB'].most_common(3)
print tag_trans_counts_ja['VERB'].most_common(3)
print
print tag_trans_counts_en['NOUN'].most_common(3)
print tag_trans_counts_ja['NOUN'].most_common(3)

theta_hmm_ja, all_tags_ja = hmm.compute_HMM_weights(JA_TRAIN_FILE,.01)

tagger = lambda words, all_tags : viterbi.viterbi_tagger(words,
                                                         hmm.hmm_features,
                                                         theta_hmm_ja,
                                                         all_tags_ja)[0]

confusion = tagger_base.eval_tagger(tagger,'hmm-dev-ja.preds',testfile=JA_DEV_FILE)

print scorer.accuracy(confusion)

tagger_base.apply_tagger(tagger,'hmm-test-ja.preds',testfile=JA_TEST_FILE_HIDDEN)

# you don't have the test file, so you can't run this
confusion_te_ja = scorer.get_confusion(JA_TEST_FILE,'hmm-test-ja.preds')
print scorer.accuracy(confusion_te_ja)

