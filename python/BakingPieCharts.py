get_ipython().magic('matplotlib inline')
from ete3 import Tree, TreeStyle, TextFace,NodeStyle,faces, COLOR_SCHEMES
import random

t = Tree( "((a,b),c);" )

# Basic tree style
ts = TreeStyle()
ts.show_leaf_name = True

# Creates an independent node style for each node, which is
# initialized with a red foreground color.
for n in t.traverse():
   nstyle = NodeStyle()
   nstyle["fgcolor"] = "red"
   nstyle["size"] = 15
   n.set_style(nstyle)

# Let's now modify the aspect of the root node
#t.img_style["size"] = 30
#t.img_style["fgcolor"] = "blue"

t.render("%%inline",tree_style=ts)

#Modified from: https://github.com/etetoolkit/ete/blob/master/ete3/test/test_treeview/barchart_and_piechart_faces.py

def get_example_tree():
    t = Tree()
    ts = TreeStyle()
    ts.layout_fn = layout
    ts.mode="r"
    ts.show_leaf_name = True
    t.populate(10)
    return t, ts

schema_names = COLOR_SCHEMES.keys()

def layout(node):
    if node.is_leaf():
        pass
    else:
        F= faces.PieChartFace([10,20,5,5,60],
                              colors=COLOR_SCHEMES["set1"],
                              width=50, height=50)
        F.border.width = None
        F.opacity = 0.5
        faces.add_face_to_node(F,node, 0, position="branch-right")

t,ts = get_example_tree()

for n in t.traverse():
    nstyle=NodeStyle()
    nstyle["size"] = 0
    n.set_style(nstyle)

t.render("%%inline",tree_style=ts)

t = Tree( "((H,I), A, (B,(C,(J, (F, D)))));" )

#Root the tree 
t.set_outgroup("A")

#Name the nodes
edge_num = 0
for node in t.traverse():
    if not node.is_leaf():
        node.name = "Node-{}".format(edge_num)
    edge_num += 1

#Define a "Face Naming" function    
def node_name(node):
    if not node.is_leaf():
        F = TextFace(node.name)
        faces.add_face_to_node(F,node,0,"branch-top")
    
#Make the tips sorta line up...
t.convert_to_ultrametric()

#Use TreeStyle to associate the TextFace with our tree
ts = TreeStyle()
ts.layout_fn = node_name
ts.mode="r"
ts.show_leaf_name = True        


t.render("%%inline",tree_style=ts)



#Some pie chart data (must add up to 100)
node_pies = {"Node-0":[10,20,70],
             "Node-2":[70,20,10],
             "Node-3":[4,16,80],
             "Node-4":[22,25,53],
             "Node-8":[90,5,5],
             "Node-10":[10,70,20],
             "Node-12":[10,20,70],
             
            }

#Associate the PieChartFace only with internal nodes
def pie_nodes(node):
    if node.is_leaf():
        pass
    else: 
        F= faces.PieChartFace(node_pies[node.name],
                              colors=COLOR_SCHEMES["set1"],
                              width=50, height=50)
        F.border.width = None
        F.opacity = 0.5
        faces.add_face_to_node(F,node, 0, position="branch-right")

ts.layout_fn = pie_nodes
t.render("%%inline",tree_style=ts)

subtree = Tree("(J, (F, D));")
subtree.render("%%inline")

for node in t.traverse():
    if node.get_topology_id() == subtree.get_topology_id():
        print node

sptree = Tree("((((A,B),C),D),E);")
phyparts_node_key = ["Node0 (((A,B),C),D)","Node1 ((A,B),C)","Node2 (A,B)"]

#Node,concord,conflict1,conflict2,totConcord&Conflict

phyparts_hist = ["Node0,2.0,2.0,1.0,5.0", "Node1,6.0,1.0,1.0,8.0", "Node2,4.0,2.0,1.0,7.0"]

phyparts_pies = {}

for n in phyparts_hist:
    n = n.split(",")
    tot_genes = float(n.pop(-1))
    node_name = n.pop(0)
    phyparts_pies[node_name] = [float(x)/tot_genes*100 for x in n]

    
print phyparts_pies    
subtrees_dict = {n.split()[0]:Tree(n.split()[1]+";") for n in phyparts_node_key}

for node in sptree.traverse():
    for subtree in subtrees_dict:
        if node.get_topology_id() == subtrees_dict[subtree].get_topology_id():
            node.name = subtree
            
def phyparts_pie_layout(mynode):
    if mynode.name in phyparts_pies:
        F= faces.PieChartFace(phyparts_pies[mynode.name],
                              colors=COLOR_SCHEMES["set1"],
                              width=50, height=50)
        F.border.width = None
        F.opacity = 0.5
        faces.add_face_to_node(F,mynode, 0, position="branch-right")


ts = TreeStyle()
        
ts.layout_fn = phyparts_pie_layout
sptree.convert_to_ultrametric()
sptree.render("%%inline",tree_style=ts) 

    



get_ipython().magic('matplotlib inline')
from ete3 import Tree, TreeStyle, NodeStyle, faces, COLOR_SCHEMES

sptree = Tree("((((A,B),C),D),E);")
phyparts_node_key = ["Node0 (((A,B),C),D)","Node1 ((A,B),C)","Node2 (A,B)"]
phyparts_pies = {'Node1': [75.0, 12.5, 12.5], 'Node0': [40.0, 40.0, 20.0], 'Node2': [57.142, 28.57, 14.285]}
subtrees_dict = {n.split()[0]:Tree(n.split()[1]+";") for n in phyparts_node_key}

nstyle = NodeStyle()
nstyle["size"] = 0

for node in sptree.traverse():
    node.set_style(nstyle)
    for subtree in subtrees_dict:
        if node.get_topology_id() == subtrees_dict[subtree].get_topology_id():
            node.name = subtree
            
def phyparts_pie_layout(mynode):
    if mynode.name in phyparts_pies:
        F= faces.PieChartFace(phyparts_pies[mynode.name],
                              colors=COLOR_SCHEMES["set1"],
                              width=50, height=50)
        F.border.width = None
        F.opacity = 0.5
        faces.add_face_to_node(F,mynode, 0, position="branch-right")
    else:
        T = faces.TextFace(mynode.name)
        faces.add_face_to_node(T,mynode,0,position="aligned")


ts = TreeStyle()
ts.show_leaf_name = False        
ts.layout_fn = phyparts_pie_layout
ts.draw_guiding_lines = True
ts.guiding_lines_type = 0
ts.guiding_lines_color = "black"
ts.show_scale = False
ts.scale = 100
sptree.render("%%inline",tree_style=ts) 



