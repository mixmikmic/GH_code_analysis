"""
a ---> d ---> e <--- c
       ^
       |
       b
"""

from collections import defaultdict 

adjacency_list = defaultdict()
adjacency_list['a'] = ['d']
adjacency_list['b'] = ['d']
adjacency_list['c'] = ['e']
adjacency_list['d'] = ['e']
adjacency_list['e'] = []

visited_list = defaultdict()
visited_list['a'] = False
visited_list['b'] = False
visited_list['c'] = False
visited_list['d'] = False
visited_list['e'] = False

output_stack = []

def topology_sort(vertex):
    if not visited_list[vertex]:
        visited_list[vertex] = True
        for neighbor in adjacency_list[vertex]:
            topology_sort(neighbor)
        output_stack.insert(0, vertex)

for vertex in visited_list:
    topology_sort(vertex)

output_stack

