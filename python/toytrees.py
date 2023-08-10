import toytree
import toyplot
import numpy as np

#### Generate a random toytree
tre = toytree.rtree(ntips=10)

## draw the tree
tre.draw(
    node_color=tre.colors[3],
    node_labels=tre.get_node_values("support"),
    tip_labels=tre.get_tip_labels(),
    );

## the .tree attribute is the tree structure in memory
## which accesses the root node initially
tre.tree

## traverse the tree and access node attributes
for node in tre.tree.traverse(strategy="levelorder"):
    print "{:<5} {:<5} {:<5} {:<5}"          .format(node.idx, node.name, node.is_leaf(), node.is_root())

## traverse the tree and modify nodes (add new 'color' feature)
for node in tre.tree.traverse():
    if node.is_leaf():
        node.color=tre.colors[1]
    else:
        node.color=tre.colors[2]
        
## draw tree using new 'color' attribute
colors = tre.get_node_values('color', show_root=1, show_tips=1)
tre.draw(node_labels=True, node_color=colors);

tre.edges

## show the verts array
tre.verts

## draw a tree
canvas, axes = tre.draw();

## add scatterplot points to the axes (standard toyplot plotting)
axes.scatterplot(
    a=tre.verts[:, 0],
    b=tre.verts[:, 1],
    size=10,
    marker='o'
    );

## root on a list of names
tre.root(["t-1", "t-2"])

## or root on a wildcard selector (here all matches in 7-9)
tre.root(wildcard="t-[7-9]")

tre.draw(width=200);

tre.get_node_values(feature='dist', show_root=True, show_tips=True)

tre.get_edge_lengths()

tre.get_node_labels()

## an easily accessible list of 
tre.colors

