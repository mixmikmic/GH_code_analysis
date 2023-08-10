get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
from bp import parse_newick, to_skbio_treenode, to_skbio_treearray
from skbio import TreeNode
import numpy as np
import glob
from random import shuffle

def _correct_gg_reroot_length_issue(t):
    # the greengenes trees on reroot had a node with a length set to None
    # find and correct if it exists
    try:
        gg_reroot_none_node = t.find('k__Bacteria')
        gg_reroot_none_node.length = 0.0
    except:
        pass

    return t

def preorder_names(fp):
    """Find any preorder node name inconsistencies as a proxy for topology testing"""
    skt = _correct_gg_reroot_length_issue(TreeNode.read(fp))
    bpt = parse_newick(open(fp).read())
    
    for sk_node, k in zip(skt.preorder(include_self=True), range(bpt.B.sum())):
        bp_idx = bpt.preorderselect(k)
        if (sk_node.name != bpt.name(bp_idx)) or (sk_node.length != bpt.length(bp_idx)):
            # bpt right now uses 0.0 for root length
            if sk_node.is_root() and bp_idx == bpt.root():
                continue
            else:
                return sk_node, bp_idx
    
    return None, None

def newick_comparison(fp):
    """Verify newick output is consistent
    
    Note: the BP tree is converted to TreeNode
    """
    tn = str(_correct_gg_reroot_length_issue(TreeNode.read(fp)))
    bp = str(to_skbio_treenode(parse_newick(open(fp).read())))
    
    for i in range(len(tn)):
        if tn[i] != bp[i]:
            return (tn[i-25:i+25], bp[i-25:i+25])
    return None, None
            
def check_to_array(fp):
    skt = _correct_gg_reroot_length_issue(TreeNode.read(fp)).to_array(nan_length_value=0.0)
    bpt = to_skbio_treearray(parse_newick(open(fp).read()))
    
    if list(skt['id_index'].keys()) != list(bpt['id_index'].keys()):
        return 'id_index keys are not equal'
    
    for k in skt['id_index']:
        if skt['id_index'][k].is_tip() != bpt['id_index'][k].is_tip():
            return "id index tip identification is not equal"
    
    if not np.allclose(skt['child_index'], bpt['child_index']):
        return 'child_index is not equal'
    
    if not np.allclose(skt['length'], bpt['length']):
        return 'length is not equal'
    
    return None
    
def check_shear(fp):
    """Verify a random shear/collapse is comparable
    
    Note: skbio.TreeNode can alter the order of children in the tree. This does not
    represent a change in topology. Because of this, we are testing node subsets
    which are invariant to child order. 
    """
    skt = _correct_gg_reroot_length_issue(TreeNode.read(fp))
    bpt = parse_newick(open(fp).read())
    
    # determine which tips to keep
    names = [n.name for n in skt.tips()]
    shuffle(names)
    to_keep = int(np.ceil(len(names) * 0.1))
    names_to_keep = set(names[:to_keep])
    
    # shear the treenode
    skt_shear = skt.shear(names_to_keep) 
    bpt_shear = bpt.shear(names_to_keep).collapse()
    
    res = skt_shear.subsets() == to_skbio_treenode(bpt_shear).subsets()
    if res:
        return None
    else:
        return "shear/collapse is not equivalent"

problems = {}
for f in glob.glob('../../../greengenes_release/gg_13_8_otus/trees/*_otus.tree'):
    obs = {}
    key = f.rsplit('/')[-1]
    print(key)
    
    obs['preorder_names'] = preorder_names(f)
    obs['newick_comparison'] = newick_comparison(f)
    obs['check_to_array'] = check_to_array(f)
    obs['shear/collapse'] = check_shear(f)
    problems[key] = obs
problems



