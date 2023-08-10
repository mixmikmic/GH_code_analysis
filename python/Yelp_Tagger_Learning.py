get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import cPickle, os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from ddlite import *
np.random.seed(seed=1701)

get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (18,6)

E = Entities('yelp_tag_saved_entities_v5.pkl')

feats = None

pkl_f = 'yelp_tag_feats_v1.pkl'
try:
    with open(pkl_f, 'rb') as f:
        feats = cPickle.load(f)
except:
    get_ipython().magic('time E.extract_features()')
    with open(pkl_f, 'w+') as f:
        cPickle.dump(E.feats, f)

DDL = DDLiteModel(E, feats)
print "Extracted {} features for each of {} mentions".format(DDL.num_feats(), DDL.num_candidates())

uids = []
for i in range(198):
    uids.append(E[i].uid)
    #print E[i].mention(attribute='words')

#uid example:
uids[0]

gt = np.array([1,1,1,-1,1,1,1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,1,1,1,1,1,1,1,-1,-1,-1,1,
               1,1,1,1,1,1,1,-1,1,-1,1,1,-1,1,1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,1,1,1,
               -1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,
               1,1,1,1,1,-1,1,1,1,-1,1,  1,1,-1,1,1,1,1,-1,1,1,1,1,1,1,-1,-1,1,1,1,
               -1,1,1,1,1,1,1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,-1,-1,-1,1,-1,-1,1,1,
               1,1,1,1,1,1,1,1,1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,1,1,1,1,1,1,-1,1,1,1,
               1,1,1,-1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1])

DDL.update_gt(gt[:100], uids=uids[:100])
DDL.set_holdout(validation_frac=0.5)

DDL.open_mindtagger(num_sample=25, width='100%', height=1200)

DDL.add_mindtagger_tags()

DDL.update_gt(gt[100:], uids=uids[100:])

#Negative LFs

def LF_like(m):
    return 1 if ('like' in m.post_window('lemmas')) or ('like' in m.pre_window('lemmas')) else 0

def LF_love(m):
    return 1 if ('love' in m.post_window('lemmas')) or ('love' in m.pre_window('lemmas')) else 0

def LF_usually(m):
    return 1 if ('usually' in m.post_window('lemmas')) or ('usually' in m.pre_window('lemmas')) else 0

def LF_favorite(m):
    return 1 if ('favorite' in m.post_window('lemmas')) or ('favorite' in m.pre_window('lemmas')) else 0

peprsausage = ["pepperoni", "sausage", "cheese"]
def LF_pepsau(m):
    for p in peprsausage:
        if (p in m.mention('words')):
            return 1
    return 0

order = ["ordered", "had", "tried", "ate", "have", "has", "eat", "order"]
def LF_order(m):
    for p in order:
        if (p in m.pre_window('lemmas')):
            return 1
    return 0

#Positive LFs

def LF_notLike(m):
    return -1 if ('don\'t like' in m.post_window('lemmas')) or ('don\'t like' in m.pre_window('lemmas')) else 0

def LF_bad(m):
    return -1 if ('bad' in m.post_window('lemmas')) or ('bad' in m.pre_window('lemmas')) else 0

POS = ["VB", "VBD", "VBN", "VBP", "VBZ", "PRP", "RB", "RBR", "RBS"] 
def LF_pos(m):
    for p in POS:
        if (p in m.mention('poses')):
            return -1
    return 0

def LF_prep(m):
    for prep in ["IN"]:
        if (prep in m.mention('poses')):
            return -1
    return 0

miscneg = ["or"] 
def LF_miscneg(m):
    for p in miscneg:
        if (p in m.mention('words')):
            return -1
    return 0

otherFood = ["burger", "salad", "sandwich", "wings", "beer"]
def LF_otherFood(m):
    for food in otherFood:
        if (food in m.mention(attribute='words')):
            return -1
    return 0


LFs = [LF_like, LF_love, LF_usually, LF_favorite, LF_notLike, LF_bad, LF_prep, LF_otherFood, LF_pos, 
       LF_miscneg, LF_pepsau, LF_order]

DDL.apply_lfs(LFs, clear=True)

DDL.print_lf_stats()

DDL.plot_lf_stats()

DDL.top_conflict_lfs(n=3)

DDL.lowest_coverage_lfs(n=3)

#Show the n LFs with the lowest empirical accuracy against ground truth for candidates in the devset
DDL.lowest_empirical_accuracy_lfs(n=10) 

DDL.lf_summary_table()

matplotlib.rcParams['figure.figsize'] = (12,4)
mu_seq = np.ravel([1e-5, 1e-3, 1e-2, 1e-1])
DDL.set_use_lfs(True)
get_ipython().magic('time DDL.learn_weights(sample=False, n_iter=500, alpha=0.5, mu=mu_seq,                        bias=True, verbose=True, log=True)')

DDL.plot_calibration()

DDL.show_log()

DDL.get_predicted_probability()

print len(DDL.get_predicted())
print DDL.get_predicted()
neg_count = 0
pos_count = 0
for c in DDL.get_predicted():
    if c == -1.:
        neg_count += 1
    else:
        pos_count += 1
print "neg count:", neg_count
print "pos_count:", pos_count

import collections

names = []

print "POSITIVE CANDIDATES:\n"
for i in range(len(DDL.get_predicted())):
    if DDL.get_predicted()[i] == 1.:
        name = " ".join(DDL.C[i].mention('words'))
        print " ", name
        names.append(name)

print "\n\nNEGATIVE CANDIDATES:\n"
for i in range(len(DDL.get_predicted())):
    if DDL.get_predicted()[i] == -1.:
        name = " ".join(DDL.C[i].mention('words'))
        print " ", name

print "\n\nMost Common pizza types:\n"
collections.Counter(names).most_common(15)

