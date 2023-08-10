get_ipython().magic('pylab inline')
import pandas as pd
import seaborn as sns
from scipy import linalg

# Load data
genes = pd.read_csv('gene_expression_s.csv', index_col=0).sort_index(0).sort_index(1)

sample_data = pd.read_csv('sample_info_qc.csv', index_col=0).sort_index(0).sort_index(1)
sample_data = sample_data.ix[sample_data["Pass QC"]]

genes = genes.ix[:, sample_data.index]

# Rescale data without ERCC's
ercc_idx = filter(lambda i: 'ERCC' in i, genes.index)

egenes = genes.drop(ercc_idx)
egenes = egenes.drop('GFP')
egenes = (egenes / egenes.sum()) * 1e6

# Load gene annotations
gene_annotation = pd.read_csv('zv9_gene_annotation.txt', sep='\t', index_col=0)
gene_annotation = gene_annotation.ix[egenes.index]

# Load Ohnologs
ohnopairs = pd.read_table('ohnopairs_zebrafish_ensembl63.txt')

# Focus on only Ohnolog _pairs_
mask1 = ohnopairs['ohnolog_1'].map(lambda s: len(s.split(' '))) == 1
mask2 = ohnopairs['ohnolog_2'].map(lambda s: len(s.split(' '))) == 1

mask = mask1 & mask2

ohnopairs = ohnopairs[mask]

un_annot = set(ohnopairs.ohnolog_1).difference(gene_annotation.index)
ohnopairs = ohnopairs[(~ohnopairs.ohnolog_1.map(un_annot.__contains__))]

un_annot = set(ohnopairs.ohnolog_2).difference(gene_annotation.index)
ohnopairs = ohnopairs[(~ohnopairs.ohnolog_2.map(un_annot.__contains__))]

ohnopairs.index = ohnopairs.ohnolog_1

# Let's use log transformed expression throughout
logexp = np.log10(egenes + 1)

# Look at binary expression patterns in Ohnolog pairs

ohno_bin = pd.DataFrame(index=ohnopairs.index)

ohno_bin['c00'] = 0
ohno_bin['c01'] = 0
ohno_bin['c10'] = 0
ohno_bin['c11'] = 0

for g in ohnopairs.index:
    pair = ohnopairs.ix[g]
    oh1 = logexp.ix[pair['ohnolog_1']] > np.log10(1 + 1)
    oh2 = logexp.ix[pair['ohnolog_2']] > np.log10(1 + 1)
    
    c11 = (oh1 & oh2).sum()
    ohno_bin.ix[g, 'c11'] = c11
    ohno_bin.ix[g, 'c10'] = oh1.sum() - c11
    ohno_bin.ix[g, 'c01'] = oh2.sum() - c11
    ohno_bin.ix[g, 'c00'] = len(oh1) - (oh1 ^ oh2).sum()

# Find a value for smallest of uniquely expressed gene in a given Ohnolog pair
ohno_bin['min_c01c10'] = ohno_bin[['c01', 'c10']].min(1)

# We want to compare the number of cells where both genes in
# a pair are expressed (c11) to the number of cells where 
# only one if either gene is used exclusively.

ohno_bin['both_min_diff'] = ohno_bin['min_c01c10'] - ohno_bin['c11']
ohno_bin['both_min_diff'].sort(inplace=False, ascending=False).head()

# A high value of this measure means very XOR like beahvior,
# while a low value means both genes in a pair are expressed 
# most of the time.

# For a pair we want to look at the difference between a
# gene uniquely expressed in many cells to the other gene,
# being expressed in fewer cells.

ohno_bin['exclusive'] = list(ohno_bin[['c01', 'c10']].max(1) - ohno_bin[['c01', 'c10']].min(1))

# A high value of this indicates only one gene in a pair is
# predominately used.

# For convenience, put in gene names in the binary expression pattern table.

ohno_bin['name'] = gene_annotation.ix[ohno_bin.index, 'Associated Gene Name']

# Now we formulate the taxnonomy of the Ohnolog pairs
# in to categories by a decision tree on the binary
# expression features

ohno_bin['category'] = 'Mixed Ohnolog'

for g in ohno_bin.index:
    if ohno_bin.ix[g, 'c00'] > 300:
        ohno_bin.ix[g, 'category'] = 'Not expressed'
        continue
    
    if ohno_bin.ix[g, 'both_min_diff'] > 15:
        ohno_bin.ix[g, 'category'] = 'XOR Ohnolog'
        continue

    if ohno_bin.ix[g, 'exclusive'] > 60:
        ohno_bin.ix[g, 'category'] = 'Single Ohnolog'
        continue

ohno_bin['category'].value_counts()

# Give some colors to the classes
colorer = {
    'Not expressed': (1,1,1,1),
    'XOR Ohnolog': cm.Set2(0),
    'Single Ohnolog': cm.Set2(0.33),
    'Mixed Ohnolog': cm.Set2(0.99)
    }

# For convenience, put  the color in the table
ohno_bin['color'] = ohno_bin['category'].map(colorer.__getitem__)

figsize(12, 6)

plt.subplot(1, 2, 1)

# Plot unexpressed before expressed so we don't
# cover the expressed pairs with unexpressed pairs.

idx = ohno_bin.query('category == "Not expressed"').index
plt.scatter(ohno_bin.ix[idx, 'exclusive'],
            ohno_bin.ix[idx, 'both_min_diff'],
            color=ohno_bin.ix[idx, 'color'], edgecolor='none')

idx = ohno_bin.query('category != "Not expressed"').index
plt.scatter(ohno_bin.ix[idx, 'exclusive'],
            ohno_bin.ix[idx, 'both_min_diff'],
            color=ohno_bin.ix[idx, 'color'], edgecolor='k')

plt.axvline(60, c='k', lw=0.5);
plt.axhline(15, c='k', lw=0.5);
sns.axlabel('exclusive', 'both_min_diff');

# Make a legend
plt.subplot(1, 2, 2)

labs = []
locs = []
for i, cat in enumerate(colorer):
    plt.plot([0, 1], [i, i], color=colorer[cat], lw=15)
    locs.append(i)
    labs.append(cat)

plt.ylim(-1, 5);
plt.yticks(locs, labs);
plt.xticks([]);

plt.tight_layout();

# Plot example pairs from the different categories

sns.set_style('white');

sample_data.head()

from ast import literal_eval

sample_data.cluster_color = sample_data.cluster_color.apply(literal_eval)

def plot_pair(xg='ENSDARG00000077760', yg='ENSDARG00000073971'):
    
    xe = np.log10(egenes[sample_data.index].ix[xg] + 1)
    ye = np.log10(egenes[sample_data.index].ix[yg] + 1)
    
    mask = xe.where(xe < 0.001).index
    xe[mask] += np.random.uniform(-0.4, -0.1, size=mask.shape)

    mask = ye.where(ye < 0.001).index
    ye[mask] += np.random.uniform(-0.4, -0.1, size=mask.shape)
    
    plt.scatter(xe,
                ye,
                color=sample_data['cluster_color'],
                s=100,
                edgecolor='k')
    
    plt.axhline(0, c='grey', linestyle='--')
    plt.axvline(0, c='grey', linestyle='--')

    sns.axlabel(gene_annotation['Associated Gene Name'][xg] + ' (log10 TPM)',
                gene_annotation['Associated Gene Name'][yg] + ' (log10 TPM)')
    
    plt.xlim(-0.5, 4.1);
    plt.ylim(-0.5, 4.1);

figsize(9, 3)
example_pairs = []
np.random.seed(1221)
for i, example in enumerate(['ENSDARG00000026540', 'ENSDARG00000067570', 'ENSDARG00000044573']):
    
    plt.subplot(1, 3, i + 1)
    pair = ohnopairs.ix[example]
    example_pairs.append(pair)
    plot_pair(pair.ohnolog_1, pair.ohnolog_2)

    plt.title(ohno_bin.ix[example, 'category'])
    plt.xlabel(gene_annotation['Associated Gene Name'][pair['ohnolog_1']])
    plt.ylabel(gene_annotation['Associated Gene Name'][pair['ohnolog_2']])

    plt.xlim(-0.6, 4)
    plt.ylim(-0.6, 4)
    
sns.despine()
plt.tight_layout();
plt.savefig('figures/ohnolog_taxa_examples.pdf');

ohno_bin.columns

[['name', 'category', 'c00', 'c01', 'c10', 'c11', 'both_min_diff', 'exclusive']]

combined = ohno_bin.query('category != "Not expressed"')         .sort('category')         .rename(columns={'name': 'name_1'})

combined['id_2'] = ohnopairs.ix[combined.index, 'ohnolog_2']
combined['name_2'] = list(gene_annotation.ix[combined['id_2'], 'Associated Gene Name'])

combined[['name_1', 'id_2', 'name_2',
          'category', 'c00', 'c01', 'c10', 'c11',
          'both_min_diff', 'exclusive']] \
        .to_csv('table3-ohnolog-classes.csv')

pwd







combined.head()





ohno_bin.query('category != "Not expressed"')         .sort('category')[['name', 'category', 'c00', 'c01', 'c10', 'c11', 'both_min_diff', 'exclusive']]         .head()









# Just check some functional enrichment

from gprofiler import gprofiler

query = list(ohno_bin.query("category == 'XOR Ohnolog'").index)
result = gprofiler(query, organism='drerio')
result.sort('p.value')[['term.name', 'p.value']]

query = list(ohno_bin.query("category == 'Mixed Ohnolog'").index)
result = gprofiler(query, organism='drerio')
result.sort('p.value')[['term.name', 'p.value']]

query = list(ohno_bin.query("category == 'Single Ohnolog'").index)
result = gprofiler(query, organism='drerio')
result.sort('p.value')[['term.name', 'p.value']]







# Examine how this relates to blood related genes

blood_related = pd.read_csv('BloodLIST.csv', index_col='ID')

blood_related.shape

L1 = set(ohnopairs.ohnolog_1).intersection(blood_related.index)
L2 = set(ohnopairs.ohnolog_2).intersection(blood_related.index)

blood_ohno = (ohnopairs.ohnolog_1.apply(L1.union(L2).__contains__) |               ohnopairs.ohnolog_2.apply(L2.union(L1).__contains__))

blood_ohno_idx = ohnopairs[blood_ohno].index

blood_ohno_idx.shape

ohno_bin.ix[blood_ohno_idx]['category'].value_counts()

ohno_bin['name2'] = list(gene_annotation.ix[ohnopairs.ix[ohno_bin.index, 'ohnolog_2'], 'Associated Gene Name'])

ohno_bin.drop('color', 1).sort_index(1).sort('category').to_csv('ohnolog_taxonomy.csv')









# Make a graphic of the relation between which chromosomes the expressed Ohnologs are located at

from matplotlib.patches import Wedge, Rectangle

sns.set_style('dark');
figsize(16, 10)
ax = plt.subplot(111, polar=False)

for g in ohnopairs.index:
    try:
        p1, p2 = int(ohnopairs.ix[g].ohnolog_1_chrom), int(ohnopairs.ix[g].ohnolog_2_chrom)
    except ValueError:
        pass
    else:
        p1 += np.random.uniform(-0.25, 0.25)
        p2 += np.random.uniform(-0.25, 0.25)
        center = np.mean((p1, p2))
        radius = np.max((p1, p2)) - center
        w = Wedge((center, 12.5), radius, 180, 360, width=0.05,
                  edgecolor='none',
                  fc=ohno_bin.ix[g, 'color'],
                  lw=1,
                  zorder=(ohno_bin.ix[g, 'category'] != 'Not expressed'),
                 )
        ax.add_patch(w)
        
for i in range(1, 25 + 1):
        wdt = 0.66
        hgt = 0.33
        r = Rectangle((i - wdt / 2, 12.5 - hgt / 2), wdt, hgt, zorder=2, fc='k')
        ax.add_patch(r)
        
#         plt.text(i, 12.5 + hgt, str(i), horizontalalignment='center', verticalalignment='center')

        
plt.xlim(0, 25 + 1)
plt.ylim(0, 13 + 1 + 1);
plt.xticks([])
plt.yticks([]);

plt.tight_layout()
plt.savefig('figures/ohnolog_chromosomes.png')

# Make a legend
figsize(2, 3)
plt.subplot(1, 1, 1)

labs = []
locs = []
for i, cat in enumerate(colorer):
    plt.plot([0, 1], [i, i], color=colorer[cat], lw=15)
    locs.append(i)
    labs.append(cat)

plt.ylim(-1, 5);
plt.yticks(locs, labs);
plt.xticks([]);











# Saying something about sequence similarites

import requests

server = "http://rest.ensembl.org"
ext = "/sequence/id/ENSDARG00000075183?"

r = requests.get(server+ext, headers={ "Content-Type" : "text/plain"})

r.headers

import Levenshtein



def leven_pair(p):
    seqs = []
    for g in p:
        q = "/sequence/id/{}?".format(g)
        r = requests.get(server + q, headers={ "Content-Type" : "text/plain"})
        seqs.append(r.text)
        
    d = Levenshtein.distance(*seqs)
    return (p, d)

leven_pair(('ENSDARG00000035869', 'ENSDARG00000052789'))

from joblib import Parallel, delayed

get_ipython().system('head -n 10 similiarites.txt')

similarities = []
with open('similiarites.txt') as fh:
    for l in fh.readlines():
        similarities.append(eval(l))

similarities[:10]



['ENSDARG00000009087', 'ENSDARG00000036628'] 
g = 'ENSDARG00000009087'
q = "/sequence/id/{}?expand_5prime=1000".format(g)
r = requests.get(server + q, headers={ "Content-Type" : "text/plain"})

p1 = r.text[:1000]

g = 'ENSDARG00000036628'
q = "/sequence/id/{}?expand_5prime=1000".format(g)
r = requests.get(server + q, headers={ "Content-Type" : "text/plain"})

p2 = r.text[:1000]

p1

p2

Levenshtein.distance(p1, p2)



sim_dict = {}
for t in similarities:
    sim_dict[t[0][0]] = t[1]

sim_series = pd.Series(sim_dict)

ohno_sim = sim_series[ohno_corr.index]

idx = ohno_corr[ohno_corr > 0.1].index

figsize(8, 6)
plt.scatter(ohno_sim.ix[idx] + 1, ohno_corr.ix[idx], s=50, alpha=0.5, label='Ohnolog gene pairs');
plt.xscale('log');
sns.axlabel('Ohnolog pair Levenshtein distance', 'Ohnolog pair expression correlation')
plt.legend(scatterpoints=3);

stats.spearmanr(ohno_sim.ix[idx], ohno_corr.ix[idx])

gene_annotation[gene_annotation['Associated Gene Name'] == 'znf292b']

ohnopairs.ix['ENSDARG00000043973']

ohno_bin.ix['ENSDARG00000043973']

sim_series.ix['ENSDARG00000043973']





['ENSDARG00000043973', 'ENSDARG00000016763'] 
g = 'ENSDARG00000043973'
q = "/sequence/id/{}?expand_5prime=1000".format(g)
r = requests.get(server + q, headers={ "Content-Type" : "text/plain"})

p1 = r.text[:1000]

g = 'ENSDARG00000016763'
q = "/sequence/id/{}?expand_5prime=1000".format(g)
r = requests.get(server + q, headers={ "Content-Type" : "text/plain"})

p2 = r.text[:1000]

p1

p2

Levenshtein.distance(p1, p2)

r = requests.get(server + q, headers={ "Content-Type" : "text/plain"})

r.headers

def leven_pair_promotor(p):
    seqs = []
    for g in p:
        q = "/sequence/id/{}?expand_5prime=1000".format(g)
        r = requests.get(server + q, headers={ "Content-Type" : "text/plain"})
        print r.ok, r.reason
        seqs.append(r.text[:1000])
        
    d = Levenshtein.distance(*seqs)
    return (p, d)

leven_pair_promotor(['ENSDARG00000043973', 'ENSDARG00000016763'])

from multiprocessing import Pool

pool = Pool(processes=16)

leven_pair_promotor(('ENSDARG00000021140', 'ENSDARG00000017219'))

map(leven_pair_promotor, zip(ohnopairs.ohnolog_1, ohnopairs.ohnolog_2)[:40])

get_ipython().run_cell_magic('time', '', 'pool.map(leven_pair_promotor, zip(ohnopairs.ohnolog_1, ohnopairs.ohnolog_2)[:40])')

promotor_similarities = Parallel(n_jobs=2)(delayed(leven_pair_promotor)(p) for p in zip(ohnopairs.ohnolog_1, ohnopairs.ohnolog_2)[:40])





