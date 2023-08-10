from igraph import Graph, summary
from csv import DictReader

# 2008 will be the sample year for this introductory analysis, but let's define it as a variable
# for future development of functions that will work for all years
year = '2008'  

# import edge list and vertex list as lists of dictionaries:
e = []
with open('data/input/author_threadID_edgelist_' + year + '.csv') as csvfile:
    reader = DictReader(csvfile, dialect='excel')
    reader.fieldnames = [name.lower() for name in reader.fieldnames]  # igraph expects lowercase attribute names
    for row in reader:
        row['weight'] = int(row['weight'])  # convert weight from string to int
        e.append(row)
v = []
with open('data/input/nodelist_aut_doc_' + year +'.csv') as csvfile:
    reader = DictReader(csvfile, dialect='excel')
    reader.fieldnames = [name.lower() for name in reader.fieldnames]  # igraph expects lowercase attribute names
    for row in reader:
        v.append(row)

# build a graph from the lists; see http://igraph.org/python/doc/igraph.Graph-class.html#DictList
ml = Graph.DictList(vertices=v, edges=e, directed=True, vertex_name_attr='id')  # specify the 'id' attribute here
                                                                                # because igraph anticipates a 'name' instead

ml['name'] = year + ' Full Disclosure network'  # provide a name for this graph

summary(ml)  # list properties and attributes

print('List of authors and their connected (document) vertices, among first 45 in total list,\nlimited to those with neighbors with max in-degree < 2')
for node in ml.vs(nodetype_eq='author')[0:45]:
    indegrees = [ml.degree(_, mode='in') for _ in ml.neighbors(node)]
    if indegrees:  # a few authors have 0 neighbors... 
        if max(indegrees) < 2:
            print()
            print('-' * 55)
            print(node['label'], '\n')
            print('  In-degree\t Label')
            print('  ---------\t -----')
            for _ in ml.neighbors(node):
                print(' ', ml.degree(_, mode='in'), '\t\t', ml.vs[_]['label'][0:60] )
            print('  MAX:', max(indegrees))
            print('  AVERAGE:', sum(indegrees) / max(len(indegrees), 1))

pr = ml.pagerank(directed=True, weights='weight')
for idx, _ in enumerate(ml.vs):
    _['original_pagerank'] = pr[idx]
    _['original_enumeration'] = idx - 1  # this can link us back to the node list if needed

def nodeCount(g):
    print('Total vertices for', g['name'], ':\n    ', len(g.vs), 'including', len(g.vs(nodetype_eq='author')), 'authors')

nodeCount(ml)

# now let's start filtering...
# first remove authors with neighbors mostly of in-degree 1
print('Filtering out authors with neighbors mostly of in-degree 1...')

for node in ml.vs:
    if node['nodetype'] == 'author':
        indegrees = [ml.degree(_, mode='in') for _ in ml.neighbors(node)]
        if indegrees:  # a few authors have 0 neighbors... this is something to investigate later
            if sum(indegrees) / max(len(indegrees), 1) < 1.2: #  there are examples where most of the indegrees are 1 but
                                                     #  we have a random response, so let's use the mean
                ml.delete_vertices(node) #  deleting vertices also deletes all of its edges
ml['name'] = ml['name'] + ' (fireworks removed)'

nodeCount(ml)

print('Filtering out documents or authors with degree 0...')
for node in ml.vs:
    if ml.degree(node) == 0: 
        ml.delete_vertices(node)
ml['name'] = ml['name'] + ' (isolated nodes removed)'
nodeCount(ml)

com = ml.community_leading_eigenvector(weights='weight', clusters=3)

ml.to_undirected(combine_edges='max') # eliminate the directionality

# Attempt to identify communities

com = ml.community_leading_eigenvector(weights='weight', clusters=3)
print('clustering attempt, leading eigenvector:')
summary(com)

def saveClusterInfo(g, com, attr):
    '''add to graph 'g' a cluster number from community 'com' as 'attr' attribute'''
    for idx, c in enumerate(com):
        for _ in c:
            g.vs[_][attr] = idx

# Apply above function to our overall graph
saveClusterInfo(ml, com, 'filtered_clustering')

# add betweenness centrality, pagerank, and clustering info to original graph
def saveCentrality(g, name):
    bc = g.betweenness()
    for idx, node in enumerate(g.vs):
        g.vs[idx][name + '_betweenness'] = bc[idx]

    pr = g.pagerank(directed=False)
    for idx, node in enumerate(g.vs):
        g.vs[idx][name + '_pagerank'] = pr[idx]
    return bc, pr

bc, pr = saveCentrality(ml, 'filtered')

# First, we will define a function to summarize the clustering attempt.
def summarizeClusters(com, n=5):
    print(len(com), 'clusters.')
    print('maximum size:', len(max(com, key=len)))
    print('minimum size:', len(min(com, key=len)))

    print('\nSummary of first', n, 'clusters:')
    for i in range(n):
        etc = ''
        if len(com[i]) > 5:
            etc += '(and ' + str(len(com[i]) - 5) + ' more)'
        print('[{}] has size of {}. Vertices: {} {}'.format(i, len(com[i]), com[i][0:5], etc))

summarizeClusters(com, n=10)

# sort the betweenness and then list nodes in order of centrality
bc.sort(reverse=True)
max_bc = max(bc)
bc_normalized = [x / max_bc for x in bc]

for idx,val in enumerate(bc_normalized[0:15]):
    print('Node', idx, ':', '{:.0%}'.format(bc_normalized[idx]))

# export the ml graph
filename = 'data/output/out_' + year + '_filtered.graphml'
ml.save(filename, format='graphml') # export graph

# The blob has been manually identified as cluster 0

blob = ml.induced_subgraph(com[0])
blob['name'] = year + ' subgraph for cluster 0 (blob)'
summary(blob)

# Now working with the blob, we can re-attempt clustering and bc!

com = blob.community_leading_eigenvector(weights='weight', clusters=8)
print('clustering attempt, leading eigenvector:')
summary(com)
saveClusterInfo(blob, com, 'blob_clustering')

bc, pr = saveCentrality(blob, 'blob')

# now let's try each cluster separately

subs = {}  # build all the clusters into a dictionary
for idx, c in enumerate(com):
    subs[idx] = blob.induced_subgraph(com[idx])
    subs[idx]['name'] = year + ' blob subgraph ' + str(idx) + ' of ' + str(len(com))
    # rerun the centrality, clustering, pagerank analyses
    subcom = subs[idx].community_leading_eigenvector(weights='weight', clusters=5)
    saveClusterInfo(subs[idx], subcom, 'local_clustering')
    bc, pr = saveCentrality(subs[idx], 'local')    
    filename = 'output/' + subs[idx]['name'].replace(' ', '_') + '_out.graphml'
    # subs[idx].save(filename, format='graphml')



