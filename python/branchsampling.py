import simcat   
import toytree    ## requires github Master branch currently
import random
import numpy as np
from copy import deepcopy
import toyplot

def sample_branches(toytreeobj, betaval = 1,returndict = None):
    """
    Takes an input topology and samples branch lengths
    
    Parameters:
    -----------
    toytreeobj: toytree
        The topology for which we want to generate branch lengths
    betaval: int/float (default=1)
        The value of beta for the exponential distribution used to generate
        branch lengths. This is inverse of rate parameter.
    returndict: str (default=None)
        If "only", returns only a dictionary matching nodes to heights.
        If "both", returns toytree and dictionary as a tuple.
    """
    tree=deepcopy(toytreeobj)
    def set_node_height(node,height):
        childn=node.get_children()
        for child in childn:
            child.dist=height - child.height
    testobj = []
    
    # we'll just append each step on each chain to these lists to keep track of where we've already been
    allnodes = []
    allnodeheights = []
    
    # d is deprecated (it was being returned) and stores the values for each chain
    #d = {}
    
    ## testobj is our traversed full tree
    for i in tree.tree.traverse():
        testobj.append(i)
    
    # start by getting branch lengths for the longest branch
    longbranch=sum([testobj[-1] in i for i in testobj]) # counts the number of subtrees containing last leaf, giving length of longest branch
    longbranchnodes = np.random.exponential(betaval,longbranch) # segment lengths of longest branch
    longbranchnodes[0] = np.random.uniform(low=0.0,high=longbranchnodes[0]) # cut the edge to the leaf with a uniform draw
    
    # get the heights of nodes along this longest (most nodes) branch
    nodeheights = []
    currheight = 0
    for i in longbranchnodes:
        currheight += i
        nodeheights.append(currheight)
    nodeheights = nodeheights[::-1]

    # get indices to accompany long chain
    lcidx = []
    for i in testobj:
        if testobj[-1] in i.get_leaves():
            lcidx.append(i.idx)
    #d['0heights'] = np.array(nodeheights)
    #d['0nodes'] = np.array(lcidx[:-1])
    
    allnodes = allnodes + lcidx[:-1]
    allnodeheights = allnodeheights + nodeheights
    
    # get other necessary chains to parse
    other_chains = []
    for i in range(len(testobj)-1)[::2]:
        if len(testobj[i+1].get_leaves()) > 1:
            other_chains.append(testobj[i+1])

    # now solve
    for chainnum in range(len(other_chains)): # parse the remaining chains one at a time
        otr=other_chains[chainnum]
        # find where this chain connects to the a chain we've already solved
        firstancestor = otr.get_ancestors()[0].idx
        # which nodeheight does this branch from
        paridx=np.argmax(np.array(allnodes) == firstancestor) 
        
        # traverse the new 
        testobj1 = []
        nodes = []
        
        # save a list of nodes
        for i in otr.traverse():
            testobj1.append(i)
        
        # save the nodes that include the end of the chain (because branches out to other chains might not)
        for i in testobj1:
            if testobj1[-1] in i.get_leaves():
                nodes.append(i.idx)
        
        # make node index list to accompany lengths
        lennodes= nodes[:-1] # don't save ending leaf index
        lennodes.insert(0,firstancestor) # make chain list start with ancestor
        
        # figure out how many exponential draws to make for this chain (i.e. # new nodes + 1)
        num_new_branches = sum([testobj1[-1] in i for i in testobj1])+1
        
        # initialize array to hold the draws
        mir_lens = np.zeros((sum([testobj1[-1] in i for i in testobj1])+1))
        # draw until we have a new set of exponential branch lengths that fit the constraints of our tree height
        while not (sum(mir_lens[:(len(mir_lens)-1)]) < allnodeheights[paridx] and (sum(mir_lens) > allnodeheights[paridx])):
            mir_lens = np.random.exponential(betaval,num_new_branches) ## length of longest branch
        
        # now let's save each node value as a height
        mir_lens_heights = np.zeros((len(mir_lens)))
        subsum = 0
        for i in range(len(mir_lens)):
            mir_lens_heights[i] = allnodeheights[paridx] - subsum
            subsum = subsum + mir_lens[i]
            
        # add our new node indices with their heights to the full list
        allnodes = list(allnodes) + list(lennodes)
        allnodeheights = list(allnodeheights) + list(mir_lens_heights)
        
        #d[(str(chainnum+1)+"heights")] = mir_lens_heights
        #d[(str(chainnum+1)+"nodes")] = np.array(lennodes)
    
    # make a final dictionary of node heights, eliminating redundancy
    n = dict(set(zip(*[allnodes,allnodeheights])))
    
    if returndict == "only":
        return n #d
    elif returndict == "both":
        # create the tree object
        for node in n.keys():
            set_node_height(tree.tree.search_nodes(idx = node)[0],n[node])
        return (tree,n)
    else:
        # create the tree object
        for node in n.keys():
            set_node_height(tree.tree.search_nodes(idx = node)[0],n[node])
        return tree

## generate a random tree
random.seed(123)
tree = toytree.rtree(10)
tree.tree.convert_to_ultrametric()
tree.draw(tree_style='c', node_labels='idx', tip_labels=False, padding=50);

sample_branches(tree)

sample_branches(tree,returndict="only")

ntrees = 4
canvas = toyplot.Canvas(width=800, height=200)
axes = [canvas.cartesian(grid=(1, ntrees, idx)) for idx in range(ntrees)]
for i in range(ntrees):
    ax = axes[i]
    ax.show = False
    sample_branches(tree).draw(
        axes=ax, tree_style='c', node_labels='name', tip_labels=False, node_size=16)

random.seed(123)
tree = toytree.rtree(16)
tree.tree.convert_to_ultrametric()
sample_branches(tree).draw(tree_style='c', node_labels='idx', tip_labels=False, padding=50);

random.seed(123)
tree = toytree.rtree(17)
tree.tree.convert_to_ultrametric()
sample_branches(tree).draw(tree_style='c', node_labels='idx', tip_labels=False, padding=50);

random.seed(123)
tree = toytree.rtree(17)
tree.tree.convert_to_ultrametric()
samptree, sampdict = sample_branches(tree,returndict="both")
print("Top node height should be: "+str(sampdict[32]))
print("\n" + "Top node height is instead: " + str(samptree.tree.search_nodes(idx = 32)[0].height))

get_ipython().run_cell_magic('timeit', '', 'random.seed(123)\ntree = toytree.rtree(70)\ntree.tree.convert_to_ultrametric()\nprint(len(sample_branches(tree,returndict="only"))) ## 68 internal nodes!')

