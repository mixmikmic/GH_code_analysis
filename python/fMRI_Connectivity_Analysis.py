## Uses Python 3
## Import statements and load data
from __future__ import division
import numpy as np
import scipy
from scipy import io
from scipy import special as special
import scipy.stats as stats
from itertools import chain,combinations
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Loading Data 
dataC = scipy.io.loadmat("data/controls.mat")['controls'] #17, 116, 116
dataD = scipy.io.loadmat("data/depressed.mat")['depressed'] #16, 116, 116

# Let's visualize these first
C_avgd = np.mean(dataC, 0)   #collapse across subjects
D_avgd = np.mean(dataD, 0)

fig, axs = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)

axs[0].imshow(C_avgd, interpolation='nearest')
axs[0].set_title('Control')
#plt.show()
axs[1].set_title('Depressed')
axs[1].imshow(D_avgd, interpolation='nearest')
plt.show()

# Calculate characteristic path length

import scipy.sparse.csgraph as graphs

CPL_C = np.zeros(dataC.shape[0])
CPL_D = np.zeros(dataD.shape[0])

for subj in range(0, dataC.shape[0]):
    dist = graphs.shortest_path(np.abs(dataC[subj,:,:]), method = 'D')  #our connectivity measure can be negative
    CPL_C[subj] = np.nanmean(np.nanmean(np.where(dist!=0, dist, np.nan),1))
    
for subj in range(0, dataD.shape[0]):
    dist = graphs.shortest_path(np.abs(dataD[subj,:,:]), method = 'D')  
    CPL_D[subj] = np.nanmean(np.nanmean(np.where(dist!=0, dist, np.nan),1))

# Visualize our CPL measures between groups
plt.plot(CPL_D, len(CPL_D) * [0.5], "x", label = 'Controls')
plt.plot(CPL_C, len(CPL_C) * [1], "+", label = 'Depressed')
plt.axis([0, 1,0,2])
plt.legend()
plt.show()
# We'll test the group difference significance later...

# Connected Components: Thresholding

bin_C = np.ones(dataC.shape)   # we'll have to store our adjacency matrices following binarization
bin_D = np.ones(dataD.shape)

thresh = 1.4                # our threshold; all values < epsilon will be assigned a 0
                             # abs(values) range from 0 to 2.9; mess with epsilon accordingly 

sub_threshold_indices = np.abs(dataC) < thresh
bin_C[sub_threshold_indices] = 0  

sub_threshold_indices = np.abs(dataD) < thresh
bin_D[sub_threshold_indices] = 0    

# Connected Components: Computation 

comp_C = np.zeros([dataC.shape[0], 2])  #column 1: DFS result; column 2: Laplacian result 
comp_D = np.zeros([dataD.shape[0], 2])

eps = 1e-7                              #numerical stability stuff 

for subj in range(0, dataC.shape[0]):
    comp_C[subj,0] = graphs.connected_components(bin_C[subj,:,:])[0]   #DFS
    
    lap       = graphs.laplacian(bin_C[subj,:,:])                      #Create Laplacian
    u, s, vh  = np.linalg.svd(lap)                                     #under the hood: find nullspace
    null_mask = (s <= eps)
    null_space     = np.compress(null_mask, vh, axis=0)
    comp_C[subj,1] = null_space.shape[0]
    
    
for subj in range(0, dataD.shape[0]):
    comp_D[subj,0] = graphs.connected_components(bin_D[subj,:,:])[0]
    
    lap = graphs.laplacian(bin_D[subj,:,:])
    u, s, vh  = np.linalg.svd(lap)
    null_mask = (s <= eps)
    null_space     = np.compress(null_mask, vh, axis=0)
    comp_D[subj,1] = null_space.shape[0]

# Let's check out our results

#all values positive --> if 0 then both methods agree 100%
print('Sum of method differences for controls is:', np.mean(np.abs(comp_D[:,0] - comp_D[:,1]))) 
print('Sum of method differences for depressed is:', np.mean(np.abs(comp_C[:,0] - comp_C[:,1])))

# now plot group differences
plt.plot(comp_C[:,0], len(comp_C[:,0]) * [0.5], "x", label = 'Controls')
plt.plot(comp_D[:,0], len(comp_D[:,0]) * [1], "+", label = 'Depressed')
plt.axis([0, 116,0,2])
plt.legend()
plt.show()

# Robust against threshold choice? Generate 100 thresholds, look at avg connected component number for each

bin_C = np.ones(dataC.shape)   
bin_D = np.ones(dataD.shape)

ind_comp_C = np.zeros(dataC.shape[0])
ind_comp_D = np.zeros(dataD.shape[0])   
avg_comp   = np.zeros([100, 2])            # column 1: controls; Column 2: depressed

thresh = 0.025                             # initial threshold; we'll move it in increments

for i in range(0, 100): 
    sub_threshold_indices_C = np.abs(dataC) < (thresh * i)  
    sub_threshold_indices_D = np.abs(dataD) < (thresh * i)
    bin_C[sub_threshold_indices_C] = 0  
    bin_D[sub_threshold_indices_D] = 0 
    
    for subj in range(0, dataC.shape[0]):
        ind_comp_C[subj] = graphs.connected_components(bin_C[subj,:,:])[0] 
    for subj in range(0, dataD.shape[0]):
        ind_comp_D[subj] = graphs.connected_components(bin_D[subj,:,:])[0] 
    avg_comp[i,:] = [np.mean(ind_comp_C), np.mean(ind_comp_D)]
      

plt.plot(np.linspace(0.025, 2.5, 100), avg_comp[:,0])
plt.plot(np.linspace(0.025, 2.5, 100), avg_comp[:,1])
plt.ylabel('# Connected Components')
plt.xlabel('Threshold Value')
plt.show()

from modularity import modularity_louvain_und_sign
#loop over each individual's data, and put modularity value into a list
modularity_C = [modularity_louvain_und_sign(dataC[i])[1] for i in range(len(dataC))]
modularity_D = [modularity_louvain_und_sign(dataD[i])[1] for i in range(len(dataD))]

# Visualize our modularity measures between groups
plt.plot(modularity_D, len(modularity_D) * [0.5], "x", label = 'Controls')
plt.plot(modularity_C, len(modularity_C) * [1], "+", label = 'Depressed')
plt.axis([0, 0.25, 0,2])
plt.legend()
plt.show()
#Note: the y-axis here does not mean anything. 
#It is merely introduced to separate out and visualize the controls and depressed modularity data.

## Visualize data 
# depressed = np.random.uniform(2,10,16)
# happy = np.random.uniform(5,15,17)

CPL = {"control": CPL_C, "depressed": CPL_D}
NumConnectedComponents = {"control": comp_C[:, 0], "depressed": comp_D[:, 0]}
modularity = {"control": modularity_C, "depressed": modularity_D}

for name, metric in [("Characteristic Path Length", CPL), ("Number of Connected Components", NumConnectedComponents), ("Modularity", modularity)]:
    control = metric["control"]
    depressed = metric["depressed"]
    plt.plot(control, len(control) * [0.5], "x", label = 'Controls')
    plt.plot(depressed, len(depressed) * [1], "+", label = 'Depressed')
    plt.xlabel(name)
    plt.ylim(0, 2)
    plt.legend()
    plt.show()

# Wilcoxon rank sum test
# Reference https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.wilcoxon.html

for name, metric in [("Characteristic Path Length", CPL), ("Number of Connected Components", NumConnectedComponents), ("Modularity", modularity)]:
    control = metric["control"]
    depressed = metric["depressed"]
    test_statistic, p_value= stats.ranksums(depressed, control)
    print("-------------------------------------------------------------------------------------------------------------")
    print("metric: " + name)
    print("The probability we observe a rank sum this extreme given the brains were drawn from the same distribution is:")
    print(p_value)

def get_rank_distribution(num_small ,num_total):
    """
    Returns function that evaluates p-value given sum_statisitc
    from num_small,num_total distribution
    """
    assert (num_small < num_total),"num small must be less than num total"
    num_sums = int(round(special.binom(num_total,num_small)))
    vals = np.zeros(num_sums)
    i = 0
    # Note even though we are working in python, the rankings are starting with 1
    for combination in combinations(np.arange(num_total),num_small):
        vals[i] = sum(combination)
        i += 1
    vals = np.sort(vals)         
    return vals


def get_p_val_func(rank_distribution):
    """
    Returns function that computes the p-value of the observed sum from distribution
    of vals
    """
    def p_val_func(obs_sum):
        return
    return 

def compute_rank_sum_stat(shorter_subset,total_set):
    """
    Compute rank sum of smaller population
    """
    return 

def compute_rank_of(x,sorted_lst):
    """
    Given x that lives in sorted list. This function finds its ranking
    """
    ix =  np.isin(sorted_lst,x)
    loc = np.where(ix)*1
    tuple_to_num = lambda x : x[0] 
    # Note : ranke is currently zero indexed
    rank = tuple_to_num(loc) 
    return rank[0]


def compute_p_value(shorter_subset,total_set):
    """ 
    Compute p-value using rank Wilcoxon Rank Sum statistic to see it their is
    a difference between the connectivity levels for depressed and happy brains 
    """
    rank_distribution = get_rank_distribution(shorter_subset ,total_set)
    p_val_func = get_p_val_func(rank_distribution)
    obs_sum = compute_rank_sum_stat(shorter_subset,total_set)
    return p_val_func(obs_sum)

assert compute_rank_of(2,np.arange(4)) == 2
assert np.allclose(get_rank_distribution(1,3), np.array([0,1,2]) )



