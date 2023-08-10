g = {
    0: [1, 2, 3],
    1: [0, 4],
    2: [0],
    3: [0, 5],
    4: [1, 5],
    5: [3, 4, 6, 7],
    6: [5],
    7: [5],
}

# print whole graph
print(g)

# print adjacency list of node 0
print(g[0])

# print adjacency list of node 5
print(g[5])

g = {
    'a': ['b', 'c', 'd'],
    'b': ['a', 'e'],
    'c': ['a'],
    'd': ['a', 'f'],
    'e': ['b', 'f'],
    'f': ['d', 'e', 'g', 'h'],
    'g': ['f'],
    'h': ['f'],
}

# print whole graph
print(g)

# print adjacency list of node 'a'
print(g['a'])

# print adjacency list of node 'e'
print(g['e'])

g = {
    0: [1, 2, 3],
    1: [0, 4],
    2: [0, 4],
    3: [0, 5],
    4: [5],
    5: [4, 6, 7],
    6: [],
    7: []
}

visited = [ False for k in g.keys() ]

def dfs(g, node):
    print("Visiting", node)
    visited[node] = True
    for v in g[node]:
        if not visited[v]:
            dfs(g, v)
            
dfs(g, 0)

def dfs_stack(g, node):
    s = []
    visited = [False for k in g.keys()]

    s.append(node)
    while len(s) != 0:
        print("Stack", s)
        c = s.pop()
        print("Visiting", c)
        visited[c] = True
        for v in g[c]:
            if not visited[v]:
                s.append(v)
    return visited

dfs_stack(g, 0)

g2 = {
  0: [1, 2, 3],
  1: [0, 4],
  2: [0],
  3: [0, 5],
  4: [1, 5],
  5: [3, 4, 6, 7],
  6: [5],
  7: [5]
}

dfs_stack(g2, 0)

def dfs_nd_stack(g, node):
    s = []
    visited = [ False for k in g.keys() ]
    instack = [ False for k in g.keys() ]

    s.append(node)
    instack[node] = True
    while len(s) != 0:
        print("Stack", s)
        c = s.pop()
        instack[c] = False
        print("Visiting", c)
        visited[c] = True
        for v in g[c]:
            if not visited[v] and not instack[v]:
                s.append(v)
                instack[v] = True
    return visited
  
dfs_nd_stack(g2, 0)

from collections import deque 

g = {
    0: [1, 2, 3],
    1: [0, 4],
    2: [0, 4],
    3: [0, 5],
    4: [5],
    5: [4, 6, 7],
    6: [],
    7: []
}

def bfs(g, node):
    
    q = deque()
    
    visited = [ False for k in g.keys() ]
    inqueue = [ False for k in g.keys() ]
    
    q.appendleft(node)
    inqueue[node] = True
    
    while not (len(q) == 0):
        print("Queue", q)
        c = q.pop()
        print("Visiting", c)
        inqueue[c] = False
        visited[c] = True
        for v in g[c]:
            if not visited[v] and not inqueue[v]:
                q.appendleft(v)
                inqueue[v] = True

    
bfs(g, 0)

input_filename = "example_graph_1.txt"

g = {}

with open(input_filename) as graph_input:
    for line in graph_input:
        # Split line and convert line parts to integers.
        nodes = [int(x) for x in line.split()]
        if len(nodes) != 2:
            continue
        # If a node is not already in the graph
        # we must create a new empty list.
        if nodes[0] not in g:
            g[nodes[0]] = []
        if nodes[1] not in g:
            g[nodes[1]] = []
        # We need to append the "to" node
        # to the existing list for the "from" node.
        g[nodes[0]].append(nodes[1])

print(g)

import pprint

pprint.pprint(g)

input_filename = "example_graph_2.txt"

g = {}

with open(input_filename) as graph_input:
    for line in graph_input:
        # Split line and convert line parts to integers.
        nodes = [int(x) for x in line.split()]
        if len(nodes) != 2:
            continue
        # If a node is not already in the graph
        # we must create a new empty list.
        if nodes[0] not in g:
            g[nodes[0]] = []
        if nodes[1] not in g:
            g[nodes[1]] = []
        # We need to append the "to" node
        # to the existing list for the "from" node.
        g[nodes[0]].append(nodes[1])
        # And also the other way round.
        g[nodes[1]].append(nodes[0])

pprint.pprint(g)

