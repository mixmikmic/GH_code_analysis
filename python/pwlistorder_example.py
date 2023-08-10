import numpy as np, pandas as pd
from random import shuffle

nusers=4
nitems=6

np.random.seed(10)
items=[i for i in range(nitems)]
item_probs=np.random.beta(a=1,b=1,size=nitems)
item_probs=item_probs/np.sum(item_probs)
item_goodness=np.random.beta(a=2,b=2,size=nitems)
user_preferences=[np.random.beta(a=3,b=3,size=nitems)*item_goodness for u in range(nusers)]
agg_item_goodness=np.zeros((nitems))
for u in range(nusers):
    agg_item_goodness+=user_preferences[u]

preferences=list()
for iteration in range(100):
    chosen_user=np.random.randint(low=0,high=nusers)
    for sample in range(3):
        chosen_items=np.random.choice(items,size=2,replace=False,p=item_probs)
        if chosen_items[0]==chosen_items[1]:
            continue
        goodness_A=user_preferences[chosen_user][chosen_items[0]]
        goodness_B=user_preferences[chosen_user][chosen_items[1]]
        if goodness_A>goodness_B:
            preferences.append((chosen_user,chosen_items[0],chosen_items[1]))
        else:
            preferences.append((chosen_user,chosen_items[1],chosen_items[0]))
            
shuffle(preferences)
pd.DataFrame(preferences, columns=['User','Prefers This Item','Over This Item']).head()

from collections import defaultdict

aggregated_preferences=defaultdict(lambda: 0)
for pref in preferences:
    if pref[1]<pref[2]:
        aggregated_preferences[(pref[1],pref[2])]+=1
    else:
        aggregated_preferences[(pref[2],pref[1])]-=1

# some preferences are hidden, thus some will be deleted at random here
for iteration in range(3):
    chosen_items=np.random.randint(low=0,high=nitems,size=2)
    if chosen_items[0]==chosen_items[1]:
        continue
    if chosen_items[0]<chosen_items[1]:
        del aggregated_preferences[(chosen_items[0],chosen_items[1])]
    else:
        aggregated_preferences[(chosen_items[1],chosen_items[0])]
aggregated_preferences

from itertools import permutations
from copy import deepcopy
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def eval_ranking(order,aggregated_preferences):
    score=0
    for ind in range(len(order)-1):
        el1=order[ind]
        el2=order[ind+1]
        if el1<el2:
            score+=aggregated_preferences[(el1,el2)]
        else:
            score-=aggregated_preferences[(el2,el1)]
    return score

best_score=-100
worst_score=100
scores=list()
for order in permutations([i for i in range(nitems)]):
    score=eval_ranking(order,aggregated_preferences)
    scores.append(score)
    if score>best_score:
        best_score=deepcopy(score)
        best_order=order
        
print('Best order:',best_order)
print("Best score:",best_score)
print('Theoretical maximum (perhaps not feasible):',np.sum([np.abs(v) for v in aggregated_preferences.values()]))
print()
print('Item goodness defined in simulation:',[(el,eval("{0:.2f}".format(item_goodness[el]))) for el in np.argsort(-item_goodness)])
print('Item goodness across all users in simulation:',[(el,eval("{0:.2f}".format(agg_item_goodness[el]))) for el in np.argsort(-agg_item_goodness)])
_=plt.hist(np.array(scores))
_=plt.title('Histogram of scores for all permutations',size=14)
_=plt.xlabel('Score')

nusers=500
nitems=30
aggregated_preferences=defaultdict(lambda: 0)

def add_preferences(el1,el2):
    global aggregated_preferences
    
    if el1<el2:
        aggregated_preferences[(el1,el2)]+=1
    else:
        aggregated_preferences[(el2,el1)]-=1

np.random.seed(100)
items=[i for i in range(nitems)]
item_probs=np.random.beta(a=2,b=2,size=nitems)
item_probs=item_probs/np.sum(item_probs)
item_goodness=np.random.beta(a=3,b=3,size=nitems)
user_preferences=[np.random.beta(a=4,b=4,size=nitems)*item_goodness for u in range(nusers)]
agg_item_goodness=np.zeros((nitems))
for u in range(nusers):
    agg_item_goodness+=user_preferences[u]

for iteration in range(300):
    chosen_user=np.random.randint(low=0,high=nusers)
    for sample in range(4):
        chosen_items=np.random.choice(items,size=2,replace=False,p=item_probs)
        if chosen_items[0]==chosen_items[1]:
            continue
        goodness_A=user_preferences[chosen_user][chosen_items[0]]
        goodness_B=user_preferences[chosen_user][chosen_items[1]]
        if goodness_A>goodness_B:
            add_preferences(chosen_items[0],chosen_items[1])
        else:
            add_preferences(chosen_items[1],chosen_items[0])
            
print('Theoretical maximum score:',np.sum([np.abs(v) for v in aggregated_preferences.values()]))
print('Number of pairwise preferences observed:',len([v for v in aggregated_preferences.values() if v!=0]))
print('Number of pairs in the list:',int(nitems*(nitems-1)/2))

import time, random
from pwlistorder import (eval_ordering, greedy_order, kwiksort, pagerank,
                         cvx_relaxation, minconflict, random_swaps, metropolis_hastings)

#Generating a random ordering
np.random.seed(1)
random_ordering=deepcopy(items)
np.random.shuffle(random_ordering)

start_time = time.time()
greedy_rank=greedy_order(aggregated_preferences, random_ordering)
time_greedy_rank=time.time() - start_time

start_time = time.time()
ks_rank=kwiksort(aggregated_preferences, random_ordering, runs=100, random_seed=1)
time_kwiksort=time.time() - start_time

start_time = time.time()
pr_rank=pagerank(aggregated_preferences, len(random_ordering))
time_pagerank=time.time() - start_time

start_time = time.time()
cvxrelax_rank=cvx_relaxation(aggregated_preferences, len(random_ordering))
time_cvxrelax=time.time() - start_time

start_time = time.time()
mc_rank=minconflict(aggregated_preferences, random_ordering)
time_minconflict=time.time() - start_time

start_time = time.time()
rs_rank=random_swaps(aggregated_preferences, random_ordering, iterations=50000, random_seed=1)
time_random_swaps=time.time() - start_time

start_time = time.time()
mh_rank=metropolis_hastings(aggregated_preferences, random_ordering, iterations=50000, explore_fact=8, random_seed=1)
time_metropolis=time.time() - start_time

lst_scores={'Greedy Order':eval_ordering(greedy_rank,aggregated_preferences),
            'Kwik-Sort (100 trials)':eval_ordering(ks_rank,aggregated_preferences),
            'PageRank (tuned epsilon)':eval_ordering(pr_rank,aggregated_preferences),
            'Convex relaxation':eval_ordering(cvxrelax_rank,aggregated_preferences),
            'Min-Conflict':eval_ordering(mc_rank,aggregated_preferences),
            'Random Swaps':eval_ordering(rs_rank,aggregated_preferences),
            'Metropolis-Hastings Swapping':eval_ordering(mh_rank,aggregated_preferences)
           }
lst_times={'Greedy Order':time_greedy_rank,
            'Kwik-Sort (100 trials)':time_kwiksort,
            'PageRank (tuned epsilon)':time_pagerank,
            'Convex relaxation':time_cvxrelax,
            'Min-Conflict':time_minconflict,
            'Random Swaps':time_random_swaps,
            'Metropolis-Hastings Swapping':time_metropolis
           }

eval_df=pd.DataFrame.from_dict(lst_scores,orient='index').rename(columns={0:'Score'}).sort_values('Score',ascending=False)
runtimes=pd.DataFrame.from_dict(lst_times,orient='index').rename(columns={0:'Time (seconds)'})
eval_df.join(runtimes)

#implementing useful metrics
def fraction_conc_pairs(list1,list2):
    pairs_list1=set()
    for i in range(len(list1)-1):
        for j in range (i+1,len(list1)):
            pairs_list1.add((list1[i],list1[j]))
            
    p=0
    q=0
    for i in range(len(list2)-1):
        for j in range (i+1,len(list2)):
            if (list2[i],list2[j]) in pairs_list1:
                p+=1
            else:
                q+=1
    return p/(p+q)

def ap_cor(list1,list2):
    pairs_list2=set()
    p_prime=0
    for i in range(len(list2)-1):
        for j in range (i+1,len(list2)):
            pairs_list2.add((list2[i],list2[j]))
    
    for i in range(1,len(list1)):
        for j in range(i):
            if (list1[j],list1[i]) in pairs_list2:
                c=1
            else:
                c=0
            p_prime+=c/i
    p_prime=p_prime/(len(list1)-1)
    return p_prime-(1-p_prime)

def sym_ap_cor(list1,list2):
    return (ap_cor(list1,list2)+ap_cor(list2,list1))/2

best_theoretical_order=list(np.argsort(-item_goodness))

lst_conc_pairs={'Greedy Order':'{:.2%}'.format(fraction_conc_pairs(greedy_rank,best_theoretical_order)),
            'Kwik-Sort (100 trials)':'{:.2%}'.format(fraction_conc_pairs(ks_rank,best_theoretical_order)),
            'PageRank (tuned epsilon)':'{:.2%}'.format(fraction_conc_pairs(pr_rank,best_theoretical_order)),
            'Convex relaxation':'{:.2%}'.format(fraction_conc_pairs(cvxrelax_rank,best_theoretical_order)),
            'Min-Conflict':'{:.2%}'.format(fraction_conc_pairs(mc_rank,best_theoretical_order)),
            'Random Swaps':'{:.2%}'.format(fraction_conc_pairs(rs_rank,best_theoretical_order)),
            'Metropolis-Hastings Swapping':'{:.2%}'.format(fraction_conc_pairs(mh_rank,best_theoretical_order))
           }
lst_sym_ap_cor={'Greedy Order':'{:.2%}'.format(sym_ap_cor(greedy_rank,best_theoretical_order)),
            'Kwik-Sort (100 trials)':'{:.2%}'.format(sym_ap_cor(ks_rank,best_theoretical_order)),
            'PageRank (tuned epsilon)':'{:.2%}'.format(sym_ap_cor(pr_rank,best_theoretical_order)),
            'Convex relaxation':'{:.2%}'.format(sym_ap_cor(cvxrelax_rank,best_theoretical_order)),
            'Min-Conflict':'{:.2%}'.format(sym_ap_cor(mc_rank,best_theoretical_order)),
            'Random Swaps':'{:.2%}'.format(sym_ap_cor(rs_rank,best_theoretical_order)),
            'Metropolis-Hastings Swapping':'{:.2%}'.format(sym_ap_cor(mh_rank,best_theoretical_order))
           }

eval_df=pd.DataFrame.from_dict(lst_scores,orient='index').rename(columns={0:'Score'}).sort_values('Score',ascending=False)
runtimes=pd.DataFrame.from_dict(lst_times,orient='index').rename(columns={0:'Time (seconds)'})
fcp=pd.DataFrame.from_dict(lst_conc_pairs,orient='index').rename(columns={0:'% concordant pairs w/generator'})
sapc=pd.DataFrame.from_dict(lst_sym_ap_cor,orient='index').rename(columns={0:'Symmetrical AP correlation w/generator'})
eval_df.join(runtimes).join(fcp).join(sapc)

pd.DataFrame({"Simulation's order":best_theoretical_order,'Metropolis-Hastings Swapping':mh_rank,
              'PageRank':pr_rank, 'Convex relaxation':cvxrelax_rank, 'Min-Conflict':mc_rank,
              'Random Swaps':rs_rank,'Kwik-Sort (100 trials)':ks_rank,'Greedy Order':greedy_rank}).head(6)

pd.DataFrame({"Simulation's order":best_theoretical_order,'Metropolis-Hastings Swapping':mh_rank,
              'PageRank':pr_rank, 'Convex relaxation':cvxrelax_rank, 'Min-Conflict':mc_rank,
              'Random Swaps':rs_rank,'Kwik-Sort (100 trials)':ks_rank,'Greedy Order':greedy_rank}).tail(6)

nitems=250
aggregated_preferences=defaultdict(lambda: 0)

np.random.seed(123)
items=[i for i in range(nitems)]
item_probs=np.random.beta(a=2,b=2,size=nitems)
item_probs=item_probs/np.sum(item_probs)
item_goodness=np.random.beta(a=3,b=3,size=nitems)
user_preferences=[np.random.beta(a=4,b=4,size=nitems)*item_goodness for u in range(nusers)]
agg_item_goodness=np.zeros((nitems))
for u in range(nusers):
    agg_item_goodness+=user_preferences[u]

for iteration in range(3000):
    prefs_user=np.random.beta(a=4,b=4,size=nitems)*item_goodness
    for sample in range(5):
        chosen_items=np.random.choice(items,size=2,replace=False,p=item_probs)
        if chosen_items[0]==chosen_items[1]:
            continue
        goodness_A=prefs_user[chosen_items[0]]
        goodness_B=prefs_user[chosen_items[1]]
        if goodness_A>goodness_B:
            add_preferences(chosen_items[0],chosen_items[1])
        else:
            add_preferences(chosen_items[1],chosen_items[0])
            
print('Theoretical maximum score:',np.sum([np.abs(v) for v in aggregated_preferences.values()]))
print('Number of pairwise preferences observed:',len([v for v in aggregated_preferences.values() if v!=0]))
print('Number of pairs in the list:',int(nitems*(nitems-1)/2))

#Generating a random ordering
np.random.seed(1)
random_ordering=deepcopy(items)
np.random.shuffle(random_ordering)

start_time = time.time()
greedy_rank=greedy_order(aggregated_preferences, random_ordering)
time_greedy_rank=time.time() - start_time

start_time = time.time()
ks_rank=kwiksort(aggregated_preferences, random_ordering, runs=100, random_seed=1)
time_kwiksort=time.time() - start_time

start_time = time.time()
pr_rank=pagerank(aggregated_preferences, len(random_ordering))
time_pagerank=time.time() - start_time

start_time = time.time()
cvxrelax_rank=cvx_relaxation(aggregated_preferences, len(random_ordering))
time_cvxrelax=time.time() - start_time

start_time = time.time()
rs_rank=random_swaps(aggregated_preferences, random_ordering, iterations=50000, random_seed=1)
time_random_swaps=time.time() - start_time

start_time = time.time()
mh_rank=metropolis_hastings(aggregated_preferences, random_ordering, iterations=50000, explore_fact=8, random_seed=1)
time_metropolis=time.time() - start_time

best_theoretical_order=list(np.argsort(-item_goodness))

lst_scores={'Greedy Order':eval_ordering(greedy_rank,aggregated_preferences),
            'Kwik-Sort (100 trials)':eval_ordering(ks_rank,aggregated_preferences),
            'PageRank (tuned epsilon)':eval_ordering(pr_rank,aggregated_preferences),
            'Convex relaxation':eval_ordering(cvxrelax_rank,aggregated_preferences),
            'Random Swaps':eval_ordering(rs_rank,aggregated_preferences),
            'Metropolis-Hastings Swapping':eval_ordering(mh_rank,aggregated_preferences)
           }
lst_times={'Greedy Order':time_greedy_rank,
            'Kwik-Sort (100 trials)':time_kwiksort,
            'PageRank (tuned epsilon)':time_pagerank,
            'Convex relaxation':time_cvxrelax,
            'Random Swaps':time_random_swaps,
            'Metropolis-Hastings Swapping':time_metropolis
           }

lst_conc_pairs={'Greedy Order':'{:.2%}'.format(fraction_conc_pairs(greedy_rank,best_theoretical_order)),
            'Kwik-Sort (100 trials)':'{:.2%}'.format(fraction_conc_pairs(ks_rank,best_theoretical_order)),
            'PageRank (tuned epsilon)':'{:.2%}'.format(fraction_conc_pairs(pr_rank,best_theoretical_order)),
            'Convex relaxation':'{:.2%}'.format(fraction_conc_pairs(cvxrelax_rank,best_theoretical_order)),
            'Random Swaps':'{:.2%}'.format(fraction_conc_pairs(rs_rank,best_theoretical_order)),
            'Metropolis-Hastings Swapping':'{:.2%}'.format(fraction_conc_pairs(mh_rank,best_theoretical_order))
           }
lst_sym_ap_cor={'Greedy Order':'{:.2%}'.format(sym_ap_cor(greedy_rank,best_theoretical_order)),
            'Kwik-Sort (100 trials)':'{:.2%}'.format(sym_ap_cor(ks_rank,best_theoretical_order)),
            'PageRank (tuned epsilon)':'{:.2%}'.format(sym_ap_cor(pr_rank,best_theoretical_order)),
            'Convex relaxation':'{:.2%}'.format(sym_ap_cor(cvxrelax_rank,best_theoretical_order)),
            'Random Swaps':'{:.2%}'.format(sym_ap_cor(rs_rank,best_theoretical_order)),
            'Metropolis-Hastings Swapping':'{:.2%}'.format(sym_ap_cor(mh_rank,best_theoretical_order))
           }

eval_df=pd.DataFrame.from_dict(lst_scores,orient='index').rename(columns={0:'Score'}).sort_values('Score',ascending=False)
runtimes=pd.DataFrame.from_dict(lst_times,orient='index').rename(columns={0:'Time (seconds)'})
fcp=pd.DataFrame.from_dict(lst_conc_pairs,orient='index').rename(columns={0:'% concordant pairs w/generator'})
sapc=pd.DataFrame.from_dict(lst_sym_ap_cor,orient='index').rename(columns={0:'Symmetrical AP correlation w/generator'})
eval_df.join(runtimes).join(fcp).join(sapc)

pd.DataFrame({"Simulation's order":best_theoretical_order,'Metropolis-Hastings Swapping':mh_rank,
              'PageRank':pr_rank, 'Convex relaxation':cvxrelax_rank,
              'Random Swaps':rs_rank,'Kwik-Sort (100 trials)':ks_rank,'Greedy Order':greedy_rank}).head(15)

pd.DataFrame({"Simulation's order":best_theoretical_order,'Metropolis-Hastings Swapping':mh_rank,
              'PageRank':pr_rank, 'Convex relaxation':cvxrelax_rank,
              'Random Swaps':rs_rank,'Kwik-Sort (100 trials)':ks_rank,'Greedy Order':greedy_rank}).tail(15)

