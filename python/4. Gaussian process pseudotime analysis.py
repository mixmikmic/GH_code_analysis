get_ipython().magic('pylab inline')
import pandas as pd
import seaborn as sns

genes = pd.read_csv('gene_expression_s.csv', index_col=0).sort_index(0).sort_index(1)
sample_data = pd.read_csv('sample_info_qc_pt.csv', index_col=0).sort_index(0).sort_index(1)

GFP = genes.ix['GFP']

sample_data = sample_data.ix[sample_data["Pass QC"]]
sample_data = sample_data.query('cluster != "x"')

genes = genes[sample_data.index]

ercc_idx = filter(lambda i: 'ERCC' in i, genes.index)
egenes = genes.drop(ercc_idx)
egenes = egenes.drop('GFP')
egenes = (egenes / egenes.sum()) * 1e6

gene_annotation = pd.read_csv('zv9_gene_annotation.txt', sep='\t', index_col=0)
gene_annotation = gene_annotation.ix[egenes.index]

from ast import literal_eval
sample_data['cluster_color'] = sample_data['cluster_color'].apply(literal_eval)

sample_data['ranktime'] = sample_data['pseudotime'].rank()

logexp = np.log10(egenes + 1)

sample_data = sample_data.sort('ranktime')
logexp = logexp[sample_data.index]

gene_id = 'ENSDARG00000013477'  # gata1a

from sklearn.preprocessing import scale

import GPy

from scipy import stats



# Sort cells bys pseudotime

ranktime = sample_data.ranktime.copy()
ranktime.sort()
logexp = logexp[ranktime.index]

from glob import iglob

# Read in results from parallell GP training.
rbf_dfs = []
const_dfs = []
for gp_f in iglob('fit_gps/gp_*.csv'):
    if 'rbf' in gp_f:
        rbf_dfs.append(pd.read_csv(gp_f, index_col=0))
    else:
        const_dfs.append(pd.read_csv(gp_f, index_col=0))
    
rbf_gps = pd.concat(rbf_dfs)
const_gps = pd.concat(const_dfs)

rbf_gps = rbf_gps.sort_index()
const_gps = const_gps.sort_index()

# D-statistic of the different models for each gene
D = -2 * const_gps['log_likelihood'] + 2 * rbf_gps['log_likelihood']

figsize(8, 6)
plt.scatter(const_gps.ix[const_gps.index, 'log_likelihood'],
            D.ix[const_gps.index],
            c=rbf_gps.ix[:, 4:204].std(1),
            edgecolor='grey')

sns.axlabel('Bias kernel log likelihood', 'D statistic')
sns.despine();

sns.set_style('white')

figsize(8, 6)
plt.scatter(const_gps.ix[const_gps.index, 'log_likelihood'],
            D.ix[const_gps.index],
            c=rbf_gps.ix[:, 4:204].std(1),
            edgecolor='grey')

plt.xlim(-1000, 500)
plt.ylim(-50, 600)

plt.colorbar(label='STD of posterior mean');
sns.axlabel('Bias kernel log likelihood', 'D statistic')
sns.despine();

# Collect features of the models in a DataFrame

gp_info = pd.DataFrame({'D': D})

gp_info['stde'] = rbf_gps.ix[:, 4:204].std(1)

xx = np.linspace(0, 361, 200)[:,None]
peaktime = xx[rbf_gps.ix[:, 4:204].as_matrix().argmax(1)]
gp_info['pktm'] = peaktime

std_threshold = 0.75

figsize(8, 6)
plt.scatter(const_gps.ix[const_gps.index, 'log_likelihood'],
            D.ix[const_gps.index],
            c=gp_info['stde'] > std_threshold,
            edgecolor='grey')

plt.xlim(-1000, 500)
plt.ylim(-50, 600)

plt.colorbar(label='STD of posterior mean');
sns.axlabel('Bias kernel log likelihood', 'D statistic')
sns.despine();

print((gp_info['stde'] > std_threshold).sum())

# Index of significantly varying genes sorted by peak time
idx = gp_info.query('stde > {}'.format(std_threshold)).sort('pktm', ascending=True).index

from sklearn import preprocessing

M = preprocessing.scale(rbf_gps.ix[idx, 4:204], 1)

figsize(10, 10)
sns.heatmap(M, lw=0,
            yticklabels=False,
            xticklabels=True,
            cbar_kws={'label': 'Expression (z-score scaled)', 'shrink':.75});

loc, lab = plt.xticks()
plt.xticks(loc[::15][1:-1], np.round(xx[::15, 0])[1:-1])
plt.xlabel('Pseudotime')
plt.ylabel('Genes (sorted by peak time)');

sys.path.append('/nfs/research2/teichmann/valentine/GPclust/')

import GPclust

Y_ts = logexp.ix[idx].as_matrix()
X_ts = np.atleast_2d(ranktime).T

k_underlying = GPy.kern.RBF(input_dim=1, variance=0.5, lengthscale=150)
k_corruption = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=50) + GPy.kern.White(input_dim=1, variance=0.01)

m = GPclust.MOHGP(X_ts, k_underlying, k_corruption, Y_ts, K=4, prior_Z='DP', alpha=1.0)

figsize(8, 8)
m.plot(on_subplots=True, colour=True, newfig=False, joined=False, data_in_grey=True, errorbars=True)

m.optimize()

figsize(10, 4)
m.plot(on_subplots=True, colour=True, newfig=False, joined=False, data_in_grey=True, errorbars=True)

for i in range(4):
    plt.plot(xx, m.predict_components(xx)[0][i], lw=2);

cluster_membership = pd.DataFrame({'cluster': np.argmax(m.phi,1)}, index=idx)

cluster_membership.head()

cluster_membership.cluster.value_counts()

from gprofiler import gprofiler

query = list(cluster_membership.query('cluster == 0').index)
result = gprofiler(query, organism='drerio')
result.sort('p.value')[['p.value', 'term.name']].to_csv('cluster_I_enrichment.csv')
result.sort('p.value')[['p.value', 'term.name']].head(10)

query = list(cluster_membership.query('cluster == 1').index)
result = gprofiler(query, organism='drerio')
result.sort('p.value')[['p.value', 'term.name']].to_csv('cluster_II_enrichment.csv')
result.sort('p.value')[['p.value', 'term.name']].head(10)

query = list(cluster_membership.query('cluster == 2').index)
result = gprofiler(query, organism='drerio')
result.sort('p.value')[['p.value', 'term.name']].to_csv('cluster_III_enrichment.csv')
result.sort('p.value')[['p.value', 'term.name']].head(10)

figsize(5, 3)

cluster = 0

fixed_inputs = []
fixed_dims = np.array([i for i,v in fixed_inputs])
free_dims = np.setdiff1d(np.arange(m.X.shape[1]),fixed_dims)

col='k'

X = m.X[:,free_dims]
xmin, xmax = X.min(), X.max()

Xgrid = np.empty((300, m.X.shape[1]))
Xgrid[:,free_dims] = np.linspace(xmin,xmax,300)[:,None]
for i,v in fixed_inputs:
    Xgrid[:,i] = v
    
mu  = m.predict_components(Xgrid)[0][cluster]
var = m.predict_components(Xgrid)[1][cluster]

cluster_intervals = zip(sample_data.groupby('cluster').first()['ranktime'] - 1,
                        sample_data.groupby('cluster').last()['ranktime'] + 1)
cluster_intervals_color = sample_data.groupby('cluster').first()['cluster_color']

for interval, colr in zip(cluster_intervals, cluster_intervals_color):
    plt.axvspan(*interval, facecolor=colr, zorder=-1, edgecolor='none', alpha=0.2)


plt.fill_between(Xgrid[:,free_dims].flatten(),
                 mu - 2. * np.sqrt(np.diag(var)),
                 mu + 2. * np.sqrt(np.diag(var)),
                 alpha=0.33,
                 color='grey',
                 zorder=1,)

plt.plot(Xgrid[:,free_dims].flatten(), mu.flatten(), lw=2, zorder=2, color='k')

plt.ylim(-0.1, 5)
plt.xlim(0, 370)

sns.despine()
plt.title('Group I')

plt.tight_layout();
plt.savefig('figures/pseudotime_cluster_I.pdf')

figsize(5, 3)
g = 'ENSDARG00000054155'

plt.scatter(sample_data.ranktime, logexp.ix[g], color=sample_data.cluster_color, edgecolor='k', s=75);
plt.title(gene_annotation.ix[g, 'Associated Gene Name'])

plt.ylim(-0.1, 5);
plt.xlim(0, 370)
sns.despine();

plt.tight_layout();
plt.savefig('figures/pseudotime_cluster_I_example.pdf')

figsize(5, 3)

cluster = 1

fixed_inputs = []
fixed_dims = np.array([i for i,v in fixed_inputs])
free_dims = np.setdiff1d(np.arange(m.X.shape[1]),fixed_dims)

col='k'

X = m.X[:,free_dims]
xmin, xmax = X.min(), X.max()

Xgrid = np.empty((300, m.X.shape[1]))
Xgrid[:,free_dims] = np.linspace(xmin,xmax,300)[:,None]
for i,v in fixed_inputs:
    Xgrid[:,i] = v
    
mu  = m.predict_components(Xgrid)[0][cluster]
var = m.predict_components(Xgrid)[1][cluster]

cluster_intervals = zip(sample_data.groupby('cluster').first()['ranktime'] - 1,
                        sample_data.groupby('cluster').last()['ranktime'] + 1)
cluster_intervals_color = sample_data.groupby('cluster').first()['cluster_color']

for interval, colr in zip(cluster_intervals, cluster_intervals_color):
    plt.axvspan(*interval, facecolor=colr, zorder=-1, edgecolor='none', alpha=0.2)


plt.fill_between(Xgrid[:,free_dims].flatten(),
                 mu - 2. * np.sqrt(np.diag(var)),
                 mu + 2. * np.sqrt(np.diag(var)),
                 alpha=0.33,
                 color='grey',
                 zorder=1,)

plt.plot(Xgrid[:,free_dims].flatten(), mu.flatten(), lw=2, zorder=2, color='k')

plt.ylim(-0.1, 5)
plt.xlim(0, 370)

sns.despine()
plt.title('Cluster II')

plt.tight_layout();
plt.savefig('figures/pseudotime_cluster_II.pdf')

figsize(5, 3)
g = 'ENSDARG00000036629'
plt.scatter(sample_data.ranktime, logexp.ix[g], color=sample_data.cluster_color, edgecolor='k', s=75);
plt.title(gene_annotation.ix[g, 'Associated Gene Name'])

plt.ylim(-0.1, 5);
plt.xlim(0, 370)
sns.despine();

plt.tight_layout();
plt.savefig('figures/pseudotime_cluster_II_example.pdf')

figsize(5, 3)

cluster = 2

fixed_inputs = []
fixed_dims = np.array([i for i,v in fixed_inputs])
free_dims = np.setdiff1d(np.arange(m.X.shape[1]),fixed_dims)

col='k'

X = m.X[:,free_dims]
xmin, xmax = X.min(), X.max()

Xgrid = np.empty((300, m.X.shape[1]))
Xgrid[:,free_dims] = np.linspace(xmin,xmax,300)[:,None]
for i,v in fixed_inputs:
    Xgrid[:,i] = v
    
mu  = m.predict_components(Xgrid)[0][cluster]
var = m.predict_components(Xgrid)[1][cluster]

cluster_intervals = zip(sample_data.groupby('cluster').first()['ranktime'] - 1,
                        sample_data.groupby('cluster').last()['ranktime'] + 1)
cluster_intervals_color = sample_data.groupby('cluster').first()['cluster_color']

for interval, colr in zip(cluster_intervals, cluster_intervals_color):
    plt.axvspan(*interval, facecolor=colr, zorder=-1, edgecolor='none', alpha=0.2)


plt.fill_between(Xgrid[:,free_dims].flatten(),
                 mu - 2. * np.sqrt(np.diag(var)),
                 mu + 2. * np.sqrt(np.diag(var)),
                 alpha=0.33,
                 color='grey',
                 zorder=1,)

plt.plot(Xgrid[:,free_dims].flatten(), mu.flatten(), lw=2, zorder=2, color='k')

plt.ylim(-0.1, 5)
plt.xlim(0, 370)

sns.despine()
plt.title('Cluster III')

plt.tight_layout();
plt.savefig('figures/pseudotime_cluster_III.pdf')

figsize(5, 3)
g = 'ENSDARG00000010785'

plt.scatter(sample_data.ranktime, logexp.ix[g], color=sample_data.cluster_color, edgecolor='k', s=75);
plt.title(gene_annotation.ix[g, 'Associated Gene Name'])

plt.ylim(-0.1, 5);
plt.xlim(0, 370)
sns.despine();

plt.tight_layout();
plt.savefig('figures/pseudotime_cluster_III_example.pdf')

from matplotlib import gridspec

figsize(6, 10)

M = preprocessing.scale(rbf_gps.ix[cluster_membership.sort('cluster').index, 4:204], 1)

breaks = np.where((cluster_membership.sort('cluster').diff() != 0).as_matrix())[0]

shift = 3
Ms = np.zeros((M.shape[0] + shift * (len(breaks) - 1), M.shape[1]))
for i in range(len(breaks) - 1):
    Ms[breaks[i] + shift * i : breaks[i+1] + shift * i] = M[breaks[i]:breaks[i+1]]
    
Ms[breaks[i+1] + shift * (i+1) : ] = M[breaks[i+1]: ]

gs = gridspec.GridSpec(2, 1, hspace=0.05, height_ratios=(0.95, 0.05))

plt.subplot(gs[0, 0])

sns.heatmap(Ms,
            lw=0,
            square=False,
            cbar=False,
            yticklabels=False,
            xticklabels=False,
           );

loc, lab = plt.xticks()
plt.ylabel('Genes');

plt.subplot(gs[1, 0])

for interval, colr in zip(cluster_intervals, cluster_intervals_color):
    plt.axvspan(*interval, facecolor=colr, zorder=-1, edgecolor='none', alpha=1)
    
plt.yticks([])
x_ticks = np.linspace(sample_data.ranktime.min(), sample_data.ranktime.max(), 10)
plt.xlim(x_ticks.min(), x_ticks.max())
plt.xticks(x_ticks, map(int, np.round(x_ticks)))
plt.xlabel('Pseudotime')

sns.despine(left=True, bottom=True)

plt.savefig('figures/pseudotime_clusters_heatmap.pdf');



# Note that the name of the cluster need to be identified manually,
# they change order when you run the clustering algorithm
cluster = 1

mu_2 = m.predict_components(np.atleast_2d(sample_data.ranktime).T)[0][cluster]

mRNA_content = 1e6 - sample_data['ERCC Content']

stats.spearmanr(mRNA_content, mu_2)







cluster_membership.query('cluster == 2').index

genes = pd.read_csv('gene_expression_s.csv', index_col=0).sort_index(0).sort_index(1)
sample_data = pd.read_csv('sample_info_qc.csv', index_col=0).sort_index(0).sort_index(1)

GFP = genes.ix['GFP']

sample_data = sample_data.ix[sample_data["Pass QC"]]

genes = genes[sample_data.index]

ercc_idx = filter(lambda i: 'ERCC' in i, genes.index)
egenes = genes.drop(ercc_idx)
egenes = egenes.drop('GFP')
egenes = (egenes / egenes.sum()) * 1e6

sample_data.query('cluster == "x"').index

y_2m = egenes.ix[cluster_membership.query('cluster == 2').index, sample_data.query('cluster == "x"').index].mean(1)
y_2m

x_2m = egenes.ix[cluster_membership.query('cluster == 2').index, sample_data.query('cluster == "4"').index].mean(1)
x_2m

figsize(4, 4)
plt.scatter(x_2m + 1, y_2m + 1);
plt.plot([1, 1e4+1], [1, 1e4+1]);
plt.loglog();
sns.axlabel('Cluster 4', 'Cluster x');



membership_table = pd.DataFrame({'group': cluster_membership.cluster.map(lambda s: 'I' * (s + 1)).sort(inplace=False)})

membership_table['Associated Gene Name'] = gene_annotation.ix[membership_table.index, 'Associated Gene Name']

membership_table [['Associated Gene Name', 'group']] .sort(['group', 'Associated Gene Name']) .to_csv('pseudotime_gene_groups_membership.csv')



