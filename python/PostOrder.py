import dendropy
import pandas as pd

data = pd.read_csv('../Data/PyronParityData.csv', index_col=0, header=False)

taxa = dendropy.TaxonSet()
mle = dendropy.Tree.get_from_path('../2598364_0', 'newick', taxon_set=taxa, preserve_underscores=True,extract_comment_metadata=True)

for idx, nd in enumerate(mle.leaf_iter()):
    if nd.label is None:
        lookup = '{}'.format(nd.taxon)
        nd.label = int(data.ix[lookup])
    else: 
        pass

origins = [] #changes to viviparity
reversions = [] #reversions to oviparity
total = [] #should equal 3951
childs = []

for index, node in enumerate(mle.postorder_node_iter()):
    if node.parent_node is None:
        pass
    if float(node.label) == 0 or 1 > float(node.label) > .5:
        total.append(node)
        if node.parent_node is None:
            pass
        
        elif float(node.parent_node.label) < 0.05:
            reversions.append(node)
            foci = node.parent_node
            if foci.parent_node is None:
                print 'root'
        elif float(node.parent_node.label) > .5:
             new_foci = node.parent_node 
             if new_foci.parent_node is None:
                 pass
             elif float(new_foci.parent_node.label) < 0.05:
                 reversions.append(new_foci)
        elif float(node.parent_node.label) > .05:
             new_foci = node.parent_node 
             if new_foci.parent_node is None:
                 pass
             elif float(new_foci.parent_node.label) < 0.05:
                 reversions.append(new_foci)                    

print len(set(reversions)), 'reversions'
print set(reversions)

origins = [] #changes to viviparity

for index, node in enumerate(mle.postorder_node_iter()):
    if node.parent_node is None:
        pass
    if float(node.label) == 1 or 0 < float(node.label) < .05:
        total.append(node)
        if node.parent_node is None:
            pass
        
        elif float(node.parent_node.label) > 0.95:
            origins.append(node)
            foci = node.parent_node
            if foci.parent_node is None:
                print 'root'
        elif float(node.parent_node.label) > .5:
             new_foci = node.parent_node 
             if new_foci.parent_node is None:
                 pass
             elif float(new_foci.parent_node.label) > 0.95:
                 origins.append(new_foci)
        elif float(node.parent_node.label) > .95:
             new_foci = node.parent_node 
             if new_foci.parent_node is None:
                 pass
             elif float(new_foci.parent_node.label) > 0.95:
                 origins.append(new_foci)                    

print len(set(origins)), 'reversions'
print set(origins)

print len(childs)





