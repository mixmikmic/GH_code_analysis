get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from __future__ import print_function
from collections import deque

tree = ('F',
        ('B',
         ('A', None, None),
         ('D',
          ('C', None, None),
          ('E', None, None))),
        ('G', 
         None, 
         ('I', 
          ('H', None, None),
          None)))

# Print Pre-order: F, B, A, D, C, E, G, I, H

def print_pre_order(tree):
    if tree: # if not None
        print(tree[0], end=", ")  # Print current node
        print_pre_order(tree[1])  # Print left
        print_pre_order(tree[2])  # Print right
        
print_pre_order(tree)

# Print In-order: A, B, C, D, E, F, G, H, I

def print_in_order(tree):
    if tree: # if not None
        print_in_order(tree[1])  # Print left
        print(tree[0], end=", ")  # Print current node
        print_in_order(tree[2])  # Print right
        
print_in_order(tree)

# Print Post-order: A, C, E, D, B, H, I, G, F

def print_post_order(tree):
    if tree: # if not None
        print_post_order(tree[1])  # Print left
        print_post_order(tree[2])  # Print right
        print(tree[0], end=", ")  # Print current node
        
print_post_order(tree)

# Print Level-order: F, B, G, A, D, I, C, E, H

def print_queue(queue):
    node = queue.popleft()
    if node: # if not None
        print(node[0], end=", ") # print first node in queue
        # add rest to queue
        queue.append(node[1])
        queue.append(node[2])

def print_level_order(tree):
    queue = deque() # Queue to hold the bread first search
    queue.append(tree)
    while queue:
        print_queue(queue)
        
print_level_order(tree)

