get_ipython().magic('pylab inline')

from SuchTree import SuchTree, SuchLinkedTrees, pearson
import pandas as pd
import numpy as np
import seaborn

T1 = SuchTree( '../fishpoo/mcgee_trimmed.tree' )
T2 = SuchTree( 'http://edhar.genomecenter.ucdavis.edu/~russell/fishpoo/fishpoo2_p200_c2_unique_2_clustalo_fasttree.tree' )
links = pd.read_csv( 'http://edhar.genomecenter.ucdavis.edu/~russell/fishpoo/fishpoo2_p200_c2_host_count_table.tsv', 
                    sep='\t', index_col='Host')
links.index = map( lambda x : x.replace(' ','_'), links.index )

from ete2 import Tree, TreeStyle, NodeStyle, TextFace
from numpy import linspace

ts = TreeStyle()
ts.mode = 'r'
ts.show_leaf_name = True
ts.branch_vertical_margin = 2
ts.scale = 1000
ts.show_leaf_name = False
ts.show_scale = False

nstyle = NodeStyle()
nstyle['size'] = 0

ete_tree = Tree( '../fishpoo/mcgee_trimmed.tree' )

for node in ete_tree.traverse() :
    node.set_style(nstyle)
    if node.is_leaf :
        tf = TextFace( node.name.replace('_',' ').replace('\'','') )
        tf.fsize = 10
        tf.hz_align = 100
        node.add_face( tf, 0 )

ete_tree.render("%%inline", w=120, units="mm", tree_style=ts)

D1 = zeros( ( len(T1.leafs),len(T1.leafs) ) )
for i,a in enumerate(T1.leafs.values()) :
    for j,b in enumerate( T1.leafs.values() ) :
        D1[i,j] = T1.distance( a, b )
        
seaborn.clustermap(D1)

D2_list = []
for i,a in enumerate(T1.leafs.values()) :
    for j,b in enumerate( T1.leafs.values() ) :
        D2_list.append( ( a, b ) )
D2_array = array( D2_list )
D2 = T1.distances( D2_array )
D2 = D2.reshape( ( len(T1.leafs), len(T1.leafs) ) )
seaborn.clustermap(D2)

get_ipython().run_cell_magic('time', '', 'SLT = SuchLinkedTrees( T1, T2, links )')

SLT.TreeB.get_leafs( 7027 )

SLT.subset_b( 7027 )
print 'subset size, a :', SLT.subset_a_size
print 'subset size, b :', SLT.subset_b_size
print 'subset links   :', SLT.subset_n_links
print 'link pairs     :', ( SLT.subset_n_links * ( SLT.subset_n_links -1 ) ) / 2

print SLT.col_ids
print SLT.subset_columns
print SLT.subset_b_leafs

result = SLT.linked_distances()
seaborn.jointplot( result['TreeA'], result['TreeB'] )

SLT.subset_a( 1 )
print 'subset size, a :', SLT.subset_a_size
print 'subset size, b :', SLT.subset_b_size
print 'subset links   :', SLT.subset_n_links
print 'link pairs     :', ( SLT.subset_n_links * ( SLT.subset_n_links -1 ) ) / 2

result = SLT.linked_distances()
seaborn.jointplot( result['TreeA'], result['TreeB'] )

SLT.subset_a( SLT.TreeA.root )
print 'subset size, a :', SLT.subset_a_size
print 'subset size, b :', SLT.subset_b_size
print 'subset links   :', SLT.subset_n_links
print 'link pairs     :', ( SLT.subset_n_links * ( SLT.subset_n_links -1 ) ) / 2

SLT.get_column_leafs( 0, as_row_ids=True )

SLT.get_column_links( 0 )

result_sampled = SLT.sample_linked_distances(sigma=0.05, n=10000, buckets=10)
result_sampled

seaborn.jointplot( result_sampled['TreeA'], result_sampled['TreeB'] )

seaborn.kdeplot(result['TreeB'])

seaborn.kdeplot(result_sampled['TreeB'])

sd = sorted(list(set(result_sampled['TreeB'])))
ad = sorted(list(set(result['TreeB'])))

plot( linspace(0,1,len(sd)), sd )
plot( linspace(0,1,len(ad)), ad )

seaborn.kdeplot(array(sd))

seaborn.kdeplot(array(ad))

import pyprind

p = pyprind.ProgBar( len( list( SLT.TreeB.get_internal_nodes() ) ), monitor=True, title='sampling trees...' )

big_nodes = []
table = {}
for n,node in enumerate( SLT.TreeB.get_internal_nodes() ) :
    p.update()
    SLT.subset_b( node )
    if SLT.subset_n_links > 4000 :
        big_nodes.append( node )
        result = SLT.sample_linked_distances( sigma=0.05, n=1000, buckets=100)
    else :
        result = SLT.linked_distances()
    table[node] = { 'n_leafs'    : SLT.subset_b_size, 
                    'n_links'    : SLT.subset_n_links,
                    'n_pairs'    : result['n_pairs'],
                    'n_samples'  : result['n_samples'],
                    'deviatnon_a': result['deviation_a'],
                    'deviation_b': result['deviation_b'],
                    'r'          : pearson( result['TreeA'], result['TreeB'] ) }

#C.to_csv( 'docs/fishpoo.csv' )
C = pd.DataFrame.from_csv( 'docs/fishpoo.csv' )

#C = pd.DataFrame( table ).T
seaborn.jointplot( 'n_links', 'r', data=C )

seaborn.jointplot( 'n_links', 'r', data=C.query('n_leafs > 5 and r > 0.05')  )

CC = C.query('n_leafs > 5 and r > 0.2').sort_values('r', ascending=False)
print CC.shape
CC.head()

from scipy.stats import kendalltau, pearsonr

pearson_p   = {}
kendall_tau = {}
kendall_p   = {}

for n,node in enumerate( CC.index ) :
    SLT.subset_b(node)
    result = SLT.linked_distances()
    p_r,p_p = pearsonr(   result['TreeA'], result['TreeB'] )
    k_t,k_p = kendalltau( result['TreeA'], result['TreeB'] )
    pearson_p[node]  = p_p
    kendall_tau[node] = k_t
    kendall_p[node]   = k_p

CC['pearson_p'] = pd.Series(pearson_p)
CC['kendall_tau'] = pd.Series(kendall_tau)
CC['kendall_p'] = pd.Series(kendall_p)
#CC.head()
#CC

seaborn.jointplot( 'r', 'kendall_tau', data=CC ) 

seaborn.jointplot( 'n_links', 'kendall_tau', data=CC )

figure(figsize=(10,8))

for n,(node_id,row) in enumerate( CC.iterrows() ) :
    data = dict(row)
    subplot(4,4,n+1)
    SLT.subset_b( node_id )
    result = SLT.linked_distances()
    scatter( result['TreeA'], result['TreeB'], marker='o', s=3, c='black', alpha=0.4 )
    xticks([])
    yticks([])
    xlim((-0.1,max( result['TreeA'] )+0.1))
    ylim((-0.1,max( result['TreeB'] )+0.1))
    title( str(node_id) + ' : ' + str(data['kendall_tau']) )
    if n == 15 : break

tight_layout()

from skbio import TreeNode

skt = TreeNode.read('http://edhar.genomecenter.ucdavis.edu/~russell/fishpoo/fishpoo2_p200_c2_unique_2_clustalo_fasttree.tree', convert_underscores=False)

cladeid = 42333
SLT.subset_b(cladeid)

sfeal = dict( zip(SLT.TreeB.leafs.values(), SLT.TreeB.leafs.keys() ) )
clade_leafs = map( lambda x : sfeal[x], SLT.subset_b_leafs )

clade = skt.shear( clade_leafs )
clade.children[0].length = 0
clade.write( str(cladeid) + '.tree' )

with open( str(cladeid) + '.tree' ) as f1 :
    with open( str(cladeid) + '_noquote.tree', 'w' ) as f2 :
        f2.write( f1.read().replace('\'','') )

from ete2 import Tree, TreeStyle, NodeStyle, TextFace
from numpy import linspace

ts = TreeStyle()
ts.mode = 'r'
ts.show_leaf_name = True
ts.branch_vertical_margin = 2
ts.scale = 30000
ts.show_leaf_name = False
ts.show_scale = False

nstyle = NodeStyle()
nstyle['size'] = 0

ete_tree = Tree( str(cladeid) + '.tree' )

for node in ete_tree.traverse() :
    node.set_style(nstyle)
    if node.is_leaf :
        tf = TextFace( node.name.replace('_',' ').replace('\'','') )
        tf.fsize = 100
        tf.hz_align = 100
        node.add_face( tf, 0 )

ete_tree.render("%%inline", w=120, units="mm", tree_style=ts)

clinks = links[ clade_leafs ]
uclinks = clinks.applymap( bool ).unstack()
uclinks = uclinks[ uclinks ]
with open( str(cladeid) + '.links', 'w' ) as f :
    for pair in list(uclinks.index) :
        f.write( '\t'.join(pair) + '\n' )

with open( str(cladeid) + '.rev_links', 'w' ) as f :
    for pair in list(uclinks.index) :
        f.write( '\t'.join(pair[::-1]) + '\n' )

from screed import ScreedDB, read_fasta_sequences

read_fasta_sequences( 'fishpoo2_p200_c2_unique_2_clustalo.fasta' )

db = ScreedDB( 'fishpoo2_p200_c2_unique_2_clustalo.fasta' )

with open( str(cladeid) + '.aln.fasta', 'w' ) as f :
    for leaf in clade_leafs :
        a = db[leaf]
        f.write( '>' + a.name + '\n' + str(a.sequence) + '\n' )

get_ipython().magic('load_ext rpy2.ipython')

get_ipython().run_cell_magic('R', '-w 30 -h 20 -u cm', "\nlibrary('phytools')\n\ntr1 <- read.tree( 'mcgee.tree' )\ntr2 <- read.newick( '42333_noquote.tree' )\ntr2 <- collapse.singles(tr2)\nassoc = as.matrix(read.csv( '42333.rev_links', sep='\\t', header=FALSE ))\ncolnames(assoc)<-c('tips.tr1','tips.tr2')\n\nobj <- cophylo( tr1, tr2, assoc=assoc )\nplot(obj)")

