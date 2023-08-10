os.system('python get_files.py')

# Directory with SRA files
SRA_dir='./SRA_files/'

# Path to our version of kallisto that outputs transcript compatibility counts
modified_kallisto_path='/data/SS_RNA_seq/Code/kalliPso_pair'

# Path to transcriptome
transcriptome_path='/data/SS_RNA_seq/Trapnell/reference_transcriptome/Homo_sapiens.GRCh38.rel79.cdna.all.fa'

num_processes=32

import os
cmd=('python Trapnell_wrapper.py -i '+SRA_dir+' -n ' 
     + str(num_processes)+' -k '+modified_kallisto_path
     +' -t '+transcriptome_path)
os.system(cmd)

import pickle
import numpy as np
with open('./Trapnell_TCC_distribution.dat', 'rb') as infile:
    X = pickle.load(infile)
with open('./Trapnell_TCC_pairwise_distance.dat','rb') as infile:
    D = pickle.load(infile)
Trap_labels=np.loadtxt('./Trapnells_data/Trapnell_labels.txt',dtype=str)

assert np.all(np.isclose(D,D.T))
assert np.all(np.isclose(np.diag(D),np.zeros(np.diag(D).shape)))

# Clustering is done using scikit-learn's implementation of affinity propagation

# D is a symmetric N-by-N distance matrix where N is the number of cells
from sklearn import cluster
def AffinityProp(D,pref,damp):
    aff= cluster.AffinityPropagation(affinity='precomputed',
                                     preference=pref,damping=damp, verbose=True)
    labels=aff.fit_predict(D)
    return labels

# Jensen-shannon metric used to compute distances. 
# This code is used in get_pairwise_distances.py and is repeated here for convenience. 
from scipy.stats import entropy
def jensen_shannon(p, q):
    m=0.5*p+0.5*q
    p = np.transpose(p[p > 0])
    q = np.transpose(q[q > 0])
    m = np.transpose(m[m > 0])
    return np.sqrt(entropy(m)-0.5*entropy(q)-0.5*entropy(p))

from sklearn.metrics.pairwise import pairwise_distances

pref = -1.3*np.ones(271)
labels3=AffinityProp(-D,pref,0.95)

# Clustering
pref = -.6*np.ones(271)
labels8=AffinityProp(-D,pref,0.95)

# First find the clusters with less than 5 cells
from collections import Counter
num_cells_in_cluster=Counter(labels8)
clusters_to_collapse=[x for x in np.unique(labels8) if num_cells_in_cluster[x] < 5]

# For each of those clusters, find the cluster it should be merged with
X_separated=[]
for labl in np.unique(labels8):
    features=X.todense()[np.ix_(np.flatnonzero(labels8==labl),xrange(X.shape[1]))]
    X_separated.append(features)
Xcentroid=np.zeros((len(np.unique(labels8)), (X_separated[0].shape)[1]))
for labl in np.unique(labels8):
    Xq=np.sum(X_separated[labl],axis=0)/float((X_separated[labl].shape)[0])
    Xcentroid[labl, :]=Xq
Dcentroid = pairwise_distances(Xcentroid,metric=jensen_shannon)
cluster_to_collapse_into=[np.argsort(Dcentroid[clusters_to_collapse[0],:])[1]]

# Finally, merge the clusters
labels7=np.array(labels8)
labels7[labels8==clusters_to_collapse[0]]=cluster_to_collapse_into[0]
labels7[labels8>clusters_to_collapse[0]]-=1

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

Tr0 = Counter(labels3[:69])
Tr24 = Counter(labels3[69:143])
Tr48 = Counter(labels3[143:213])
Tr72 = Counter(labels3[213:])

ind=np.arange(4)
width = 0.35

Cr0=np.array([Tr0[0], Tr24[0], Tr48[0], Tr72[0]])
Cr1=np.array([Tr0[1], Tr24[1], Tr48[1], Tr72[1]])
Cr2=np.array([Tr0[2], Tr24[2], Tr48[2], Tr72[2]])

plt.figure(figsize=(5,5))
p1 = plt.bar(ind, Cr1, width, color="#ff008a")
p3 = plt.bar(ind, Cr0, width, color="#628cf2",bottom=Cr1)
p2 = plt.bar(ind, Cr2, width, color="#01db2e",bottom=Cr0+Cr1)

plt.xticks(ind + width/2., ('T0', 'T24', 'T48', 'T72'))
plt.legend((p1[0], p3[0],p2[0]), 
           ('TCC cluster 1', 'TCC cluster 2','TCC cluster 3'),bbox_to_anchor=(1.3, 1.05))
plt.ylabel('Number of cells')
plt.xlabel('Time in low-serum medium')

import networkx as nx

X_separated=[]
for labl in np.unique(labels7):
    features=X.todense()[np.ix_(np.flatnonzero(labels7==labl),xrange(X.shape[1]))]
    X_separated.append(features)
    
Xcentroid=np.zeros((len(np.unique(labels7)), (X_separated[0].shape)[1]))
for labl in np.unique(labels7):
    Xq=np.sum(X_separated[labl],axis=0)/float((X_separated[labl].shape)[0])
    Xcentroid[labl, :]=Xq
    
Dcentroid = pairwise_distances(Xcentroid,metric=jensen_shannon)

fig=plt.figure(figsize=(8,8))

G = nx.complete_graph(len(np.unique(labels7)))
for u,v in G.edges():
    G[u][v]["weight"]=Dcentroid[u,v]
    
T=nx.minimum_spanning_tree(G)
pos=nx.spring_layout(T,scale=10000)
nx.draw_networkx(T,pos)
edge_labels=dict([((u,v,),round(d['weight'],2))
             for u,v,d in T.edges(data=True)])
plt.axis('off')

def draw_pie_MST(T,Ltrue,Labels,piesize=0.05,c= ["#628cf2","#ff008a","#01db2e"],
                 Ltrue_unique=['Proliferating cell','Differentiating myoblast',
                               'Interstitial mesenchymal cell'],leg=True,show_clust_names=False):
    pos=nx.spring_layout(T)
    fig=plt.figure(figsize=(8,8))
    ax=plt.subplot(111)
    ax.set_aspect('equal')
    nx.draw_networkx_edges(T,pos,ax=ax)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.axis('off')
    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform
    p2=piesize/2.0
    max_p = 0
    for n in T:
        xx,yy=trans(pos[n]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        a.set_aspect('equal')
        Ltemp = Ltrue[Labels == n]
        # Save proportion of each label
        fracs = []
        tot_size = float(len(Ltemp))
        for label in np.unique(Ltrue):
            fracs.append(np.sum(Ltemp == label)/tot_size)
        fracs = np.array(fracs)
        # Account for case where pie should only contain one color 
        if np.sum(fracs != 0) == 1: 
            ind = np.where(fracs != 0)[0]
            fracs = [fracs[ind]]
            ctemp = [c[ind]]
        else: 
            ctemp = c
        p,_ = a.pie(fracs,colors=ctemp,shadow=True)
        if len(p) > max_p: 
            patches = p
            max_p = len(p)
        if show_clust_names: plt.title(n)
    if leg: plt.legend(patches, Ltrue_unique, bbox_to_anchor=(14, 1))
    plt.show()

draw_pie_MST(T,Trap_labels,labels7,c=['r','b','g'],leg=True,show_clust_names=False)

draw_pie_MST(T,labels3,labels7,Ltrue_unique=['TCC cluster 1','TCC cluster 2','TCC cluster 3'],
             leg=True,show_clust_names=False)

Ghd = nx.complete_graph(271)
for u,v in Ghd.edges():
    Ghd[u][v]["weight"]=D[u,v]
Thd=nx.minimum_spanning_tree(Ghd)
colour={'1': 'red', '2': 'blue', '3':'green'}
vals=map(lambda x: colour[x],Trap_labels )
plt.figure(3,figsize=(12,12)) 
pos=nx.spring_layout(Thd,scale=10)
nx.draw(Thd,node_size=50,node_color=vals,pos=pos)
plt.show()

# Dimensionality reduction done using a diffusion map
from diffusion_maps import *
xdm,ydm = plotDiffusionMap(X.todense())

from matplotlib import gridspec

# For a label vector and a gene expression vector (both length-M where M = # cells), 
# compute the proportion of cells in each cluster that expresses the gene. 
def percent_of_each_type(g,labels):
    percents = []
    for label in np.unique(labels):
        v_label = g[np.squeeze(labels==label)]
        v_label_nz = v_label[np.squeeze(v_label>5)]
        percents.append(100*len(v_label_nz)/float(len(v_label)))
    return np.array(percents)

markergenes=["CDK1","MYOG","PDGFRA","SPHK1","MYF5","NCAM1","MSTN","HES1","HES6"]
colors = ['cornflowerblue','deeppink','limegreen']
for gn in markergenes:
    g = np.loadtxt('./Trapnells_data/'+gn+'_TPM.txt')

    fig = plt.figure(figsize=(17,3))
    gs = gridspec.GridSpec(1, 3)

    # Diffusion map with each point sized proportionally to log(expression level)
    ax1 = fig.add_subplot(gs[0])
    vec = np.log(np.ones_like(g) + g)
    vec=30*(vec/max(vec))
    for i in range(271):
        ax1.plot(xdm[i],ydm[i],'x',c=colors[labels3[i]],markersize=6,
                 linewidth=0.0,markeredgewidth=1.8)
        ax1.plot(xdm[i],ydm[i],'o',color=colors[labels3[i]],markersize=vec[i],
                 alpha=0.33,markeredgewidth=0.5,fillstyle='full')
    ax1.set_title(gn+' diffusion map')
    ax1.axis('off')

    # Bar plot of the percentage of cells in each cluster expressing the gene 
    ax2 = fig.add_subplot(gs[1])
    percents = percent_of_each_type(g,labels3)
    percents[0],percents[1] = percents[1],percents[0]
    ind = np.arange(3)
    width = 0.75
    ax2.bar(ind, percents, width,color=['deeppink','cornflowerblue','limegreen'],zorder=3)
    ax2.set_xticks(ind + width)
    ax2.set_xticklabels([1,2,3])
    ax2.set_xticks(ind+width/2)
    ax2.set_xlim(-0.25,3)
    ax2.set_xlabel('TCC Cluster')
    ax2.set_yticks([0,25,50,75,100])
    ax2.set_yticklabels([i for i in ['0%','25%','50%','75%','100%']])
    ax2.set_ylabel('Cells')
    ax2.set_title('Percentage of cells expressing '+gn)
    ax2.grid()
    
    # Bar plot visualizing the mean expression of a gene in each cluster
    ax3 = fig.add_subplot(gs[2])
    g7 = [np.average(g[labels7==i]) for i in np.unique(labels7)]
    ax3.bar(range(1,8),g7,zorder=3,align='center')
    ax3.set_xlabel('TCC Cluster')
    ax3.set_ylabel('TPM')
    ax3.set_title(gn+' expression across the 7 clusters')
    ax3.grid()

get_ipython().magic('matplotlib inline')
labelsT = Trap_labels.astype(int)-1
colors = ['red','blue','green']
plt.figure()
for i in np.unique(labelsT): 
    plt.plot(xdm[labelsT==i],ydm[labelsT==i],'o',c=colors[i])
plt.title('Diffusion map using Trapnell et al. labels')
plt.legend(['Proliferating cell','Differentiating myoblast','Interstitial mesenchymal cell'], 
           bbox_to_anchor=(1.8, 1))
plt.axis('off')
plt.show()

markergenes=["CCNB2","CCNA2","CDK1","MCM4","ACTA2","NCAM1",
             "MYH3","MYOG","MEF2C","ENO3","TNNT3","HES6"]

# Compute means and standard deviations for proliferating cells and differentiating myoblasts
T48_CT_G10,red_mean,blu_mean,red_sd,blu_sd = [],[],[],[],[]
for gn in markergenes:
    g = np.loadtxt('./Trapnells_data/'+gn+'_FPKM.txt') 
    red_mean.append(np.mean(g[Trap_labels=='1']))
    blu_mean.append(np.mean(g[Trap_labels=='2']))
    red_sd.append(np.std(g[Trap_labels=='1'])/np.sqrt(np.sum(Trap_labels=='1')))
    blu_sd.append(np.std(g[Trap_labels=='2'])/np.sqrt(np.sum(Trap_labels=='2')))
    T48_CT_G10.append(g[210])

# Plot the mean expressions of all genes of interest along with the corresponding expressions in T48_CT_G10
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ind = np.arange(len(markergenes))
width = 0.25                    
rects1 = ax.bar(ind,red_mean,width,yerr=red_sd,color='red',zorder=3)
rects2 = ax.bar(ind+width,blu_mean,width,yerr=blu_sd,color='blue',zorder=3)
rects3 = ax.bar(ind+2*width,T48_CT_G10,width,color='lavender',zorder=3)
ax.set_xlim(-width,len(ind)+width)
xTickMarks = [i for i in genes_of_interest]
ax.set_xticks(ind+width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45)
ax.set_yscale('log')
ax.set_ylabel('FPKM')
plt.legend( (rects1[0], rects2[0], rects3[0]), ('Proliferating Cells', 'Differentiating Myoblasts',"Cell T48_CT_G10"))
ax.grid(zorder=0)
plt.show()

get_ipython().run_cell_magic('time', '', "\nimport os\nimport multiprocessing as mp\n\n# Map all cells within a cluster to Trapnell's original cell file names and run\n# kallisto on pooled reads of all cells.\ndef run_kallisto_on_pooled(ftuple):\n    flnames,clusters,cluster,idxpath,outpath = ftuple[0],ftuple[1],ftuple[2],ftuple[3],ftuple[4]\n    os.system('mkdir -p '+outpath)\n    s = 'kallisto quant -i ' + idxpath + ' -o ' + outpath + '/cluster_' + str(cluster) + '/ '\n    for cell in np.where(clusters == cluster)[0]:\n        s += './reads/' + str(flnames[cell]) + '_1.fastq.gz ' + './reads/' + \\\n            str(flnames[cell]) + '_2.fastq.gz '\n    os.system(s)\n\nfilenames = np.loadtxt('./Trapnells_data/Trapnell_filenames.txt',dtype=str)\nidxpath = './kallisto_index/Trapnell_index.idx'\n\nftuples7 = [(filenames,labels7,cluster,idxpath,'./labels7') for cluster in np.unique(labels7)]\nftuples3 = [(filenames,labels3,cluster,idxpath,'./labels3') for cluster in np.unique(labels3)]\nftuples = ftuples7+ftuples3\npool = mp.Pool(processes = 50)\npool.map(run_kallisto_on_pooled,ftuples)")

from matplotlib import gridspec

# Grab Ensembl transcript names of important genes
gNames = ['CDK1','MYOG','PDGFRA']
markergenes=['CDK1','MYOG','PDGFRA']
gTrans = [0 for i in range(len(markergenes))]
for i in range(len(markergenes)):
    gTrans[i] = np.loadtxt('./Trapnells_data/'+markergenes[i]+'_transcripts.txt',dtype=str)

# Function to get transcript abundances from kallisto output
def get_abun(labelname,numclust,gTrans):
    abun = [np.zeros((numclust,1)) for i in range(len(gTrans))]
    for label in range(numclust):
        flnm='./'+labelname+'/cluster_'+str(label)+'/abundance.tsv'
        with open(flnm) as f:
            for line in f:
                line = line.split()
                for i in range(len(gTrans)):
                    if line[0] in gTrans[i]:
                        abun[i][label]+=float(line[4])
    return abun

gAbun_L7 = get_abun('labels7',7,gTrans)
gAbun_L3 = get_abun('labels3',3,gTrans)

# Generate plots
for i in range(len(gAbun_L3)):
    fig = plt.figure(figsize=(17,4.2))
    gs = gridspec.GridSpec(1, 2)
    
    # Bar plot of TPM of gene in each cluster
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(range(1,4),gAbun_L3[i],width=0.5,zorder=3,color=['deeppink','cornflowerblue','limegreen'],align='center')
    ax1.set_xlabel('TCC Cluster')
    ax1.set_ylabel('TPM')
    ax1.set_title(gNames[i]+' across 3 clusters computed after quantifying cluster centers')
    ax1.grid()
    
    # Bar plot visualizing the mean expression of a gene in each cluster
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(range(1,8),gAbun_L7[i],zorder=3,align='center')
    ax2.set_xlabel('TCC Cluster')
    ax2.set_ylabel('TPM')
    ax2.set_title(gNames[i]+' expression across the 7 clusters')
    ax2.grid()

import itertools,os

nclust = 3
nsamps = 20
seed = 100

# Get names of cells within each cluster
flnames_clust = []
for i in range(nclust):
    flnames_clust.append([filenames[i] for i in range(271) if labels3[i]==0])
    
# Sample cells
fls_clusters = []
for i in range(nclust):
    np.random.seed(100)
    fls_clusters.append(np.random.choice(flnames_clust[i],nsamps,replace=False))

# Perform quantification on subsampled cells 
def run_kallisto(fltuple):
    flname=fltuple[0]
    outpath=fltuple[2]+'/'+flname+'/'
    #print outpath
    flname_1='./reads/'+fltuple[0]+'_1.fastq.gz'
    flname_2='./reads/'+fltuple[0]+'_2.fastq.gz'
    idxpath=fltuple[1]
    os.system('mkdir -p '+outpath)
    cmd='kallisto quant -i ' + idxpath + ' -o ' + outpath + ' ' + flname_1+ ' '+flname_2
    os.system(cmd)
    
fltuples = []
for i in range(nclust):
    os.system('mkdir -p ./labels'+str(nclust)+'_cluster'+str(i))
    fltuples+=(itertools.product(fls_clusters[i],[idxpath],
                                      ['./labels'+str(nclust)+'_cluster'+str(i)]))
pool = mp.Pool(processes = 20)
pool.map(run_kallisto,fltuples)

gNames = ['CDK1','MYOG','PDGFRA']
gAbun = [[np.zeros((nsamps,1)) for i in range(nclust)] for j in range(len(gNames))]

# Load transcripts from file
gTrans = []
for i in range(len(gNames)):
    gTrans.append(np.loadtxt('./Trapnells_data/'+gNames[i]+'_transcripts.txt',dtype=str))
    
# Get TPMs for each cluster
for i in range(nclust):
    lfile='./labels'+str(nclust)+'_cluster'+str(i)+'/'
    index = 0
    for drname in os.listdir(lfile):
        with open(lfile+'/'+drname+'/abundance.tsv') as f:
            for line in f:
                line = line.split()
                for j in range(len(gTrans)):
                    if line[0] in gTrans[j]:
                        gAbun[j][i][index]+=float(line[4])
        index += 1

# Generate plots
for j in range(len(gNames)):
    percents = [np.count_nonzero(gAbun[j][i]>5)*5 for i in range(nclust)]
    
    fig = plt.figure(figsize=(14,4))
    gs = gridspec.GridSpec(1, 2)
    
    # Bar plot of TPM of gene in each cluster
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(range(1,4),percents,width=0.5,zorder=3,
            color=['deeppink','cornflowerblue','limegreen'],align='center')
    ax1.set_xlabel('TCC Cluster')
    ax1.set_ylabel('Cells')
    ax1.set_yticks([0,25,50,75,100])
    ax1.set_yticklabels([i for i in ['0%','25%','50%','75%',]])
    ax1.set_ylim(0,100)
    ax1.set_title('Percentage of sampled cells expressing '+gNames[j]+' across 3 clusters')
    ax1.grid()

    save_name='subsamp_'+gNames[j]
    if save_name is not None: plt.savefig('/data/SS_RNA_seq/Figures/'+save_name+'2.eps', format='eps', dpi=900)

# Load Trapnell's expression matrix and the related distance matrix
X_Trap = np.loadtxt('/data/SS_RNA_seq/Code/metadata/Trapnell_metadata/HSMM/HSMM_expressions.txt').T
D_Trap = pickle.load(file('/data/SS_RNA_seq/Code/pickled/Trapnell_expression_SJ.dat','rb'))

# Perform the parameter sweep
def pref_sweep(D,damping):
    n = 200
    xax = [float(i)/100 for i in range(-n,0)]
    max_cells = np.zeros(len(range(-n,0)))
    min_cells = np.zeros(len(range(-n,0)))
    num_clust = np.zeros(len(range(-n,0)))
    ii = 0
    for x in xax:
        labels=AffinityProp(-D,x,damping)
        z = np.bincount(labels)
        max_cells[ii] = np.max(z)
        min_cells[ii] = np.min(z)
        num_clust[ii] = len(z)
        ii += 1
    return (max_cells,min_cells,num_clust)

Exp_ap_curves = []
damping_params = [0.5,0.7,0.95]
for d in damping_params:
    Exp_ap_curves.append(pref_sweep(D_Trap,d))

# Visualize the results from the parameter sweeps
def plot_param_sweep(curve_set,damping_params,ttl,leg=True):
    N = len(curve_set)
    HSV_tuples = [(x*1.0/N, 0.7, 0.9) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    
    n = len(curve_set[0][0])
    xax = [float(i)/100 for i in range(-n,0)]
    
    fig = plt.figure(figsize=(17,4.2))
    gs = gridspec.GridSpec(1, 2)
    
    # First plot shows # number of cells in the largest and smallest clusters
    ax1 = fig.add_subplot(gs[0])
    for i in range(len(curve_set)):
        ax1.plot(xax,curve_set[i][0],label='damping = '+str(damping_params[i]),c=RGB_tuples[i],lw=2)
        ax1.plot(xax,curve_set[i][1],c=RGB_tuples[i],lw=2,linestyle='--')
    ax1.grid()
    ax1.set_xlabel('preference parameter')
    ax1.set_ylabel('number of cells')
    ax1.set_title(ttl)

    # Second plot shows # clusters
    ax2 = fig.add_subplot(gs[1])
    for i in range(len(curve_set)):
        ax2.plot(xax,curve_set[i][2],label='damping = '+str(damping_params[i]),c=RGB_tuples[i],lw=2)
    ax2.grid()
    if leg: ax2.legend(loc=2)
    ax2.set_ylim([0,40])
    ax2.set_xlabel('preference parameter')
    ax2.set_ylabel('number of clusters')
    ax2.set_title(ttl)
    
plot_param_sweep(Exp_ap_curves,damping_params,ttl='Gene expressions')

# Perform affinity propagation
labels_Trap=AffinityProp(-D_Trap,-.51,0.95)

# Run same MST-drawing approach as above
X_separated=[]
for labl in np.unique(labels_Trap):
    features=X.todense()[np.ix_(np.flatnonzero(labels_Trap==labl),xrange(X_Trap.shape[1]))]
    X_separated.append(features)
Xcentroid=np.zeros((len(np.unique(labels_Trap)), (X_separated[0].shape)[1]))
for labl in np.unique(labels_Trap):
    Xq=np.sum(X_separated[labl],axis=0)/float((X_separated[labl].shape)[0])
    Xcentroid[labl, :]=Xq
Dcentroid = pairwise_distances(Xcentroid,metric=jensen_shannon)
G = nx.complete_graph(len(np.unique(labels_Trap)))
for u,v in G.edges():
    G[u][v]["weight"]=Dcentroid[u,v]
T=nx.minimum_spanning_tree(G)

draw_pie_MST(T,Trap_labels,labels_Trap,c=['r','b','g'],leg=True,show_clust_names=False)

