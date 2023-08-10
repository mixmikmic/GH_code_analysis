import dendropy
import pandas as pd

data = pd.read_csv('../Data/PyronParityData.csv', index_col=0, header=False)

taxa = dendropy.TaxonSet()
mle = dendropy.Tree.get_from_path('../TotalOpt/annotatedTO_0param_2598364.dated', 'newick', taxon_set=taxa, preserve_underscores=True) 

for idx, nd in enumerate(mle.postorder_node_iter()):
    if nd.label is None:
        lookup = '{}'.format(nd.taxon)
        nd.label = int(data.ix[lookup])
    else: 
        pass

putative_c = []
putative_co = []
total = []
childs = []
for index, node in enumerate(mle.postorder_node_iter()):
    total.append(index)
    if node.parent_node is None:
        pass
    elif .5 < float(node.label) < 1 or float(node.label) == 0:    #Is likely oviparous 
        if float(node.parent_node.label) < .05 : #List of nodes that demonstrate change away from oviparity. 
            if node.taxon is not None :
                putative_co.append([node.parent_node.label, node.taxon])
            else:
                putative_co.append(node.parent_node.label)
                for nd in node.child_nodes():
#                    print nd.taxon
                     pass
    elif 0 < float(node.label) < .95 or float(node.label) == 1: 
            if float(node.parent_node.label) > .05: 
                putative_c.append([node.parent_node.label,node.taxon])      
print len(putative_c), 'changes to viviparity' 
print len(putative_co), 'reversions to oviparity'  



