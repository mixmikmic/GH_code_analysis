import pandas as pd
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

import numpy as np

with open("./results/classification.json","r") as fp:
    data = jsonpickle.loads(fp.read())

def get_std(x):
    ret_dict = {}
    for s, data in x.iteritems():
        for measure, v in data.iteritems():
            if measure not in ret_dict:
                ret_dict[measure]={}
            for k, d in v.iteritems():
                #ret_dict[(k,'mean')] = np.round(np.mean(d), 2)
                # ret_dict[(k,'std')] =  np.round(np.std(d),3)
                ret_dict[measure][(s,k)] = np.round(np.std(d),3)
    return ret_dict

df = {m: pd.DataFrame(get_std(d)) for m,d in data.iteritems()}

df['br'].apply(lambda x: x.groupby(level=0).rank(ascending=True).groupby(level=1).mean(), axis=0)

df['lp'].apply(lambda x: x.groupby(level=0).rank(ascending=True).groupby(level=1).mean(), axis=0)

df['FG'].apply(lambda x: x.groupby(level=0).rank(ascending=True).groupby(level=1).mean(), axis=0)

df['FGW'].apply(lambda x: x.groupby(level=0).rank(ascending=True).groupby(level=1).mean(), axis=0)

with open("./results/networks.json","r") as fp:
    networks = jsonpickle.loads(fp.read())

for s in networks:
    for v in networks[s]:
        for m in networks[s][v]:
            if 'SOIS-N' in networks[s][v][m]:
                del networks[s][v][m]['SOIS-N']

def get_network_std(x, var, nm):
    ret_dict = {}
    for s, data in x.iteritems():
        for k, d in data[var][nm].iteritems():
            #ret_dict[(k,'mean')] = np.round(np.mean(d), 2)
            # ret_dict[(k,'std')] =  np.round(np.std(d),3)
            ret_dict[(s,k)] = np.round(np.std(d),3)
    return ret_dict

def get_network_mean(x, var, nm):
    ret_dict = {}
    for s, data in x.iteritems():
        for k, d in data[var][nm].iteritems():
            #ret_dict[(k,'mean')] = np.round(np.mean(d), 2)
            # ret_dict[(k,'std')] =  np.round(np.std(d),3)
            ret_dict[(s,k)] = np.round(np.mean(d),3)
    return ret_dict

def jaccard_score(a,b):
    a_s = set(map(str,a))
    b_s = set(map(str,b))
    
    nominator = len(a_s.intersection(b_s))
    denominator = len(a_s.union(b_s))
    
    return float(nominator)/denominator

def get_unique(x):
    unique_x = []
    for i in x:
        if i not in unique_x:
            unique_x.append(i)
    return unique_x

def get_jaccard(y):
    scores = {}
    for n, x1 in y.iteritems():
        scores[n]=[]
        x=get_unique(x1)
        for i in xrange(len(x)):
            for j in xrange(i+1, len(x)):
                scores[n].append(jaccard_score(x[i][0], x[j][0]))
    return {k: np.mean(v) if len(v)>0 else 1.0 for k,v in scores.iteritems()}

def get_unique_count(y):
    scores = {}
    for n, x1 in y.iteritems():
        scores[n]=len(get_unique(x1))
    return scores

def get_sizes(y):
    scores = {}
    for n, x1 in y.iteritems():
        scores[n]=np.round(np.std(map(len,x1)),2)
    return scores

sets = networks.keys()
variants = ['train', 'test']

fold_methods = networks['scene']['test_communities']['FG'].keys()

jaccard_score(networks[s]['train_communities'][m][f][2], networks[s]['test_communities'][m][f][2])

def are_same(a,b):
    return int(sorted(map(sorted,a))==sorted(map(sorted,b)))

inter_fold_jaccards = {m:{} for m in network_methods}
for s in sets:
    for m in network_methods:
        for f in fold_methods:
            inter_fold_jaccards[m][(s,f)]=sum([are_same(networks[s]['train_communities'][m][f][i],networks[s]['test_communities'][m][f][i]) for i in range(10)])

modularity_diff_means = {m:{} for m in network_methods}
for s in sets:
    for m in network_methods:
        for f in fold_methods:
            modularity_diff_means[m][(s,f)]=np.mean([abs(networks[s]['train_modularities'][m][f][i]-networks[s]['test_modularities'][m][f][i]) for i in range(10)])
pd.DataFrame(modularity_diff_means).apply(lambda x: x.groupby(level=0).rank(ascending=True).groupby(level=1).mean(), axis=0)            

modularity_diff_means = {m:{} for m in network_methods}
for s in sets:
    for m in network_methods:
        for f in fold_methods:
            modularity_diff_means[m][(s,f)]=np.std([abs(networks[s]['train_modularities'][m][f][i]-networks[s]['test_modularities'][m][f][i]) for i in range(10)])
pd.DataFrame(modularity_diff_means).apply(lambda x: x.groupby(level=0).rank(ascending=True).groupby(level=1).mean(), axis=0)            



variant_names = [ '# unique communities ({})', 'Community sizes std ({})', 'Modularity std ({})', 'Modularity mean ({})']

network_frames = {m: {} for m in network_methods}
for s in sets:
    print s
    for v in variants:
        for m in network_methods:
            for v_name in variant_names:
                v_name_formatted=v_name.format(v)
                if v_name_formatted not in network_frames[m]:
                    network_frames[m][v_name_formatted]={}
            
#            for k,v1 in get_jaccard(networks[s][v+'_communities'][m]).iteritems():
#                network_frames[m]['Mean Jaccard Score ({})'.format(v)][(s,k)] = v1
            for k,v1 in get_unique_count(networks[s][v+'_communities'][m]).iteritems():
                network_frames[m]['# unique communities ({})'.format(v)][(s,k)]= v1
            for k,v1 in get_sizes(networks[s][v+'_communities'][m]).iteritems():
                network_frames[m]['Community sizes std ({})'.format(v)][(s,k)]=v1
            
            network_frames[m]['Modularity std ({})'.format(v)] = get_network_std(networks,v+'_modularities',m)
            network_frames[m]['Modularity mean ({})'.format(v)] = get_network_mean(networks,v+'_modularities',m)
            network_frames[m]['# matched partitions in folds']=inter_fold_jaccards[m]

network_dfs = {k: pd.DataFrame(network_frames[k]) for k in network_methods}
for k in network_dfs:
    for c in network_dfs[k].columns:
        if 'mean' in c.lower() or 'matched' in c.lower():
            network_dfs[k][c] = -network_dfs[k][c]

network_dfs['FG'].apply(lambda x: x.groupby(level=0).rank(ascending=True).groupby(level=1).mean(), axis=0)

network_dfs['FGW'].apply(lambda x: x.groupby(level=0).rank(ascending=True).groupby(level=1).mean(), axis=0)

len(sets)

print sets



