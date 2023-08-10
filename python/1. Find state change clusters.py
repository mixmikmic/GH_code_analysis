get_ipython().magic('pylab inline')
import pandas as pd
import seaborn as sns
from scipy import linalg

genes = pd.read_csv('gene_expression_s.csv', index_col=0).sort_index(0).sort_index(1)
sample_data = pd.read_csv('sample_info_qc.csv', index_col=0).sort_index(0).sort_index(1)

genes = genes.ix[:, sample_data[sample_data["Pass QC"]].index]
sample_data = sample_data[sample_data["Pass QC"]]

ercc_idx = filter(lambda i: 'ERCC' in i, genes.index)
egenes = genes.drop(ercc_idx)
egenes = egenes.drop('GFP')

egenes = (egenes / egenes.sum()) * 1e6

mask = (egenes > 1).sum(1) > 2
egenes = egenes.ix[mask]

gene_annotation = pd.read_csv('zv9_gene_annotation.txt', sep='\t', index_col=0)
gene_annotation = gene_annotation.ix[egenes.index]

from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE

n = 4
ica = FastICA(n, random_state=3984)
Y = ica.fit_transform(np.log10(egenes.T + 1).copy())

XX = pd.DataFrame(Y, index=egenes.columns)
XX.columns = ['difference_component', 'within_small_component', 'outlier_component', 'within_large_component']
g = sns.PairGrid(XX)
# g.map(plt.scatter, c=np.log10(sample_data['488']))
g.map(plt.scatter, c='k')

clm = sns.clustermap(XX, method='ward', lw=0, col_cluster=False);

XX.head()

# Put hidden components in sample data
for component in XX.columns:
    sample_data[component] = XX.ix[sample_data.index, component]

from scipy.cluster.hierarchy import dendrogram

from collections import defaultdict
from matplotlib.colors import rgb2hex, colorConverter 

class Clusters(dict):
    def _repr_html_(self):
        html = '<table style="border: 0;">'
        for c in self:
            hx = rgb2hex(colorConverter.to_rgb(c))
            html += '<tr style="border: 0;">'             '<td style="background-color: {0}; '                        'border: 0;">'             '<code style="background-color: {0};">'.format(hx)
            html += c + '</code></td>'
            html += '<td style="border: 0"><code>' 
            html += repr(self[c]) + '</code>'
            html += '</td></tr>'

        html += '</table>'

        return html

def get_cluster_classes(den, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l

    return cluster_classes

def get_cluster_limits(den):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    cluster_limits = Clusters()
    for c in cluster_idxs:
        cluster_limits[c] = (min(cluster_idxs[c]), max(cluster_idxs[c]))
    
    return cluster_limits

figsize(10, 3)
thr = 0.8
cden = dendrogram(clm.dendrogram_row.linkage, color_threshold=thr, labels=XX.index);
plt.axhline(thr, color='k');
plt.xticks(rotation=90, fontsize=4);

figsize(10, 3)
thr = 0.442
finer_den = dendrogram(clm.dendrogram_row.linkage, color_threshold=thr, labels=XX.index);
plt.axhline(thr, color='k');
plt.xticks(rotation=90, fontsize=4);

clusters = get_cluster_classes(cden)
clusters

finer_clusters = get_cluster_classes(finer_den)
finer_clusters

cell_color = []
for cell in XX.index:
    for color in clusters:
        if cell in clusters[color]:
            cell_color.append(color)
            break

XX = pd.DataFrame(Y, index=egenes.columns)
g = sns.PairGrid(XX)
g.map(plt.scatter, color=cell_color)

figsize(6, 6)

sm_ica = TSNE(n_components=2, perplexity=75, random_state=254)

XX2 = sm_ica.fit_transform(XX.copy())
XX2 = pd.DataFrame(XX2, index=XX.index)
plt.scatter(XX2[0], XX2[1], c=cell_color);

finer_cell_color = []
for cell in XX.index:
    for color in finer_clusters:
        if cell in finer_clusters[color]:
            finer_cell_color.append(color)
            break

figsize(6, 6)

sm_ica = TSNE(n_components=2, perplexity=75, random_state=254)

XX2 = sm_ica.fit_transform(XX.copy())
XX2 = pd.DataFrame(XX2, index=XX.index)
plt.scatter(XX2[0], XX2[1], c=finer_cell_color);

named_clusters = {}

named_clusters['1a'] = finer_clusters['c']
named_clusters['1b'] = finer_clusters['m']
named_clusters['2'] = clusters['y']
named_clusters['3'] = clusters['m']
named_clusters['4'] = clusters['g']
named_clusters['x'] = clusters['r']

cl_plt = sns.color_palette("Set2", 5)
named_cluster_colors = {'1a' : cl_plt[0],
                        '1b' : cl_plt[1],
                        '2' : cl_plt[2],
                        '3' : cl_plt[3],
                        '4' : cl_plt[4],
                        'x' : (0.8, 0.8, 0.8)}

cell_cluster = []
for cell in sample_data.index:
    for cluster in named_clusters:
        if cell in named_clusters[cluster]:
            cell_cluster.append(cluster)
            break

sample_data['cluster'] = cell_cluster

sample_data.groupby('cluster').size()

sample_data['cluster_color'] = sample_data['cluster'].map(named_cluster_colors)

sample_data['tsne_0'] = XX2[0][sample_data.index]
sample_data['tsne_1'] = XX2[1][sample_data.index]

s_d_mean = sample_data.query('cluster != "x"').groupby('cluster').mean()

s_d_first = sample_data.query('cluster != "x"').groupby('cluster').first()



sample_data.condition.value_counts()

sample_data['condition_color'] = ['#31a354' if c == 'HIGH' else '#e5f5e0' for c in sample_data['condition']]

sns.set_style('white')
sns.set_context('talk')

plt.scatter(sample_data['tsne_0'], sample_data['tsne_1'],
            color=sample_data['condition_color'],
            s=100, edgecolor='k');

plt.axis('off');
plt.tight_layout();
plt.savefig('figures/tsne_dim_bright.pdf')

sns.set_style('white')
sns.set_context('talk')

plt.scatter(sample_data['tsne_0'], sample_data['tsne_1'],
            color=sample_data['cluster_color'],
            s=100, edgecolor='k');

# plt.plot(s_d_mean['tsne_0'], s_d_mean['tsne_1'],
#          alpha=0.075,
#          color='k',
#          zorder=0,
#          lw=20,
#          )

plt.text(s_d_mean['tsne_0']['1a'] - 3,
         s_d_mean['tsne_1']['1a'] - 2.5,
         '1a',
         color=s_d_first.cluster_color['1a'],
         size=16
         )

plt.text(s_d_mean['tsne_0']['1b'] - 3,
         s_d_mean['tsne_1']['1b'] - 0,
         '1b',
         color=s_d_first.cluster_color['1b'],
         size=16
         )

plt.text(s_d_mean['tsne_0']['2'] - 0.5,
         s_d_mean['tsne_1']['2'] + 3.5,
         '2',
         color=s_d_first.cluster_color['2'],
         size=16
         )

plt.text(s_d_mean['tsne_0']['3'] + 5.3,
         s_d_mean['tsne_1']['3'] + 2.5,
         '3',
         color=s_d_first.cluster_color['3'],
         size=16
         )

plt.text(s_d_mean['tsne_0']['4'] + 7,
         s_d_mean['tsne_1']['4'],
         '4',
         color=s_d_first.cluster_color['4'],
         size=16
         )

plt.axis('off');
plt.tight_layout();
plt.savefig('figures/tsne_clusters.pdf')



comp_cols = filter(lambda n: '_component' in n, sample_data.columns)

g = sns.PairGrid(sample_data[comp_cols])
g.map(plt.scatter, color=sample_data['condition_color'], edgecolor='k')
for ax in g.axes.flatten():
    ax.set_xticks([]);
    ax.set_yticks([]);
    
plt.tight_layout()
plt.savefig('figures/ica_condition.pdf');

g = sns.PairGrid(sample_data[comp_cols])
g.map(plt.scatter, color=sample_data['cluster_color'], edgecolor='k')
for ax in g.axes.flatten():
    ax.set_xticks([]);
    ax.set_yticks([]);
    
plt.tight_layout()
plt.savefig('figures/ica_clusters.pdf');

sample_data.groupby('cluster').mean()['within_small_component']

sample_data.groupby('cluster').mean()['488']

for clid in sorted(sample_data['cluster'].unique()):
    gfp_flourenscence = sample_data.query('cluster == "{}"'.format(clid))['488']
    sns.distplot(np.log10(gfp_flourenscence),
                 color=sample_data['cluster_color'][gfp_flourenscence.index[0]],
                 hist=False,
                 label=clid)

plt.legend();

figsize(12, 4)

plt.subplot(1, 2, 1)
for clid in sorted(sample_data['cluster'].unique()):
    gfp_flourenscence = sample_data.query('cluster == "{}"'.format(clid))['FSC Horizontal']
    sns.distplot(gfp_flourenscence,
                 color=sample_data['cluster_color'][gfp_flourenscence.index[0]],
                 kde_kws={'shade': True},
                 hist=False,
                 label=clid)

plt.xlabel('FSC Horizontal (cell size)')
plt.ylabel('Density')
    
plt.subplot(1, 2, 2)
for clid in sorted(sample_data['cluster'].unique()):
    gfp_flourenscence = sample_data.query('cluster == "{}"'.format(clid))['SSC']
    sns.distplot(gfp_flourenscence,
                 color=sample_data['cluster_color'][gfp_flourenscence.index[0]],
                 kde_kws={'shade': True},
                 hist=False,
                 label=clid)
    
plt.xlabel('SSC (cell granularity)')

sns.despine()
plt.legend();

plt.tight_layout();
plt.savefig('figures/clusters_FSC_SSC.pdf')

sample_data['log_488'] = np.log10(sample_data['488'])
sample_data['log_SSC'] = np.log10(sample_data['SSC'])
g = sns.PairGrid(sample_data[['log_488', 'FSC Horizontal', 'log_SSC']])
g.map(plt.scatter, color=sample_data['cluster_color'], edgecolor='k')

sample_data.to_csv('sample_info_qc.csv')

# fli1a
egenes.ix['ENSDARG00000054632', sample_data.query('cluster == "x"').index]

# gata1a
egenes.ix['ENSDARG00000013477', sample_data.query('cluster == "x"').index]

# gf1b
egenes.ix['ENSDARG00000079947', sample_data.query('cluster == "x"').index]

# gp1bb
egenes.ix['ENSDARG00000074441', sample_data.query('cluster == "x"').index]

# itga3b
egenes.ix['ENSDARG00000012824', sample_data.query('cluster == "x"').index]





