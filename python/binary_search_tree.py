get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Import tree implementations from this folder
from binary_search_tree import BST
from red_black_tree import RedBlackTree
import random

tree = BST()
print(tree)

sample = random.sample(range(1, 20), 8)
for x in sample:
    tree.put(x)
    print(tree)
    
print('min: ', tree.min())
print('max: ', tree.max())

# while not tree.root is None:
#     tree.delete_min()
#     print tree

for i in sample:
    tree.delete(i)
    print('del: ', i , ' result: ', tree)

tree = RedBlackTree()
print(tree)

sample = random.sample(range(1, 20), 8)
for x in sample:
    tree.put(x)
    print('x: ', x, 'tree: ', tree)
    
print('min: ', tree.min())
print('max: ', tree.max())

