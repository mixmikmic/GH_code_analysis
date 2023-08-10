from collections import defaultdict
import json

# Edge is [child, parent] list
edges = [[0, 6], [17, 5], [2, 7], [4, 14], [12, 9], [15, 5], 
         [11, 1], [14, 8], [16, 6], [5, 1], [10, 7], [6, 10], [8, 2], [13, 1], 
         [1, 12], [7, 1], [3, 2], [19, 12], [18, 19]]

def construct_tree(ls_of_edges):
    """
    Construct a tree given a list of edges
    """
    # Represent the tree as a dictionary.
    # Each node is represented as a dictionary that holds references to its children.
    # Use a defaultdict, so that upon the first access of each element
    #  that element refers to an empty dictonary
    # https://docs.python.org/2/library/collections.html#collections.defaultdict
    tree = defaultdict(dict)
    # To find the root hold a set of parents and a set of children
    child_set = set()
    parent_set = set()
    # fill the dictionary
    for child, parent in ls_of_edges:
        # The parent holds the child nodes.
        tree[parent][child] = tree[child]
        # Get all the children and parents from the list of edges
        child_set.add(child)
        parent_set.add(parent)
    # Get and return the root
    root = parent_set.difference(child_set).pop()
    # Get the tree under the root and append the root as the root node
    return {9: tree[root]}

print(json.dumps(construct_tree(edges), indent=1))

