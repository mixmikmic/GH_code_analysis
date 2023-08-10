get_ipython().magic('pylab inline')
import seaborn

from SuchTree import SuchTree

from ete2 import Tree, TreeStyle, NodeStyle, TextFace
from numpy import linspace

ts = TreeStyle()
ts.mode = 'r'
ts.show_leaf_name = True
ts.branch_vertical_margin = 2
ts.scale = 10
ts.show_leaf_name = False
ts.show_scale = False

nstyle = NodeStyle()
nstyle['size'] = 0

ete_tree = Tree( 'SuchTree/tests/test.tree' )

for node in ete_tree.traverse() :
    node.set_style(nstyle)
    if node.is_leaf :
        tf = TextFace( node.name.replace('_',' ').replace('\'','') )
        tf.fsize = 10
        tf.hz_align = 100
        node.add_face( tf, 0 )

ete_tree.render("%%inline", w=120, units="mm", tree_style=ts)

T = SuchTree( '../SuchTree/tests/test.tree')

D1 = zeros( ( len(T.leafs),len(T.leafs) ) )
for i,a in enumerate(T.leafs.values()) :
    for j,b in enumerate( T.leafs.values() ) :
        D1[i,j] = T.distance( a, b )

seaborn.clustermap(D1)

D2_list = []
for i,a in enumerate(T.leafs.values()) :
    for j,b in enumerate( T.leafs.values() ) :
        D2_list.append( ( a, b ) )
D2_array = array( D2_list )
D2 = T.distances( D2_array )
D2 = D2.reshape( ( len(T.leafs), len(T.leafs) ) )

seaborn.clustermap(D2)

T3 = SuchTree( 'http://litoria.eeb.yale.edu/bird-tree/archives/PatchClade/Stage2/set1/Spheniscidae.tre' )

D3_list = []
for i,a in enumerate(T3.leafs.values()) :
    for j,b in enumerate( T3.leafs.values() ) :
        D3_list.append( ( a, b ) )
D3_array = array( D3_list )
D3 = T3.distances( D3_array )
D3 = D3.reshape( ( len(T3.leafs), len(T3.leafs) ) )

seaborn.clustermap(D3)

T1 = SuchTree( 'http://edhar.genomecenter.ucdavis.edu/~russell/fishpoo/fishpoo2_p200_c2_unique_2_clustalo_fasttree.tree' )
T2 = SuchTree( 'http://edhar.genomecenter.ucdavis.edu/~russell/fishpoo/fishpoo2_p200_c2_unique_2_clustalo_fasttree_nj.tree' )

print 'nodes : %d, leafs : %d' % ( T1.length, len(T1.leafs) )
print 'nodes : %d, leafs : %d' % ( T2.length, len(T2.leafs) )

import random

N = 1000000

v = T1.leafs.keys()

pairs = []
for i in range(N) :
    pairs.append( ( random.choice( v ), random.choice( v ) ) )

get_ipython().magic('time D1, D2 = T1.distances_by_name( pairs ), T2.distances_by_name( pairs)')

import pandas as pd

df = pd.DataFrame( { 'ML' : D1, 'neighbor joining' : D2 } )

with seaborn.axes_style("white"):
    seaborn.jointplot( 'ML', 'neighbor joining', data=df, color=seaborn.xkcd_rgb['dark teal'], alpha=0.3,
                       xlim=(0,3.5), ylim=(0,1.2), size=8 )

from scipy.stats import spearmanr, kendalltau, pearsonr

print 'Spearman\'s rs : %0.3f' % spearmanr( D1, D2 )[0]
print 'Kendall\'s tau : %0.3f' % kendalltau( D1, D2 )[0]
print 'Pearson\'s r   : %0.3f' % pearsonr( D1, D2 )[0]

from threading import Thread
from Queue import Queue

n = 2
m = 12

work_q = Queue()
done_q = Queue()

for i in xrange( m ) :
    work_q.put( pairs )

for i in xrange( n ) :
    work_q.put( 'STOP' )
    
def worker( work_q, done_q ) :
    for task in iter( work_q.get, 'STOP' ) :
        D1 = T1.distances_by_name( task )
        D2 = T2.distances_by_name( task )
        r = correlation( D1, D2 )
        done_q.put( r )
    return True

threads = []

for i in xrange( n ) :
    thread = Thread( target=worker, args=( work_q, done_q ) )
    thread.start()
    threads.append( thread )
    
for thread in threads :
    thread.join()

done_q.put( 'STOP' )

for r in iter( done_q.get, 'STOP' ) :
    print r

