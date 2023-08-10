import json, networkx as nx
import pickle

with open('wp_2017-11-05.json') as f: tree = json.load(f)

def getLinks(dictCat):
    return dictCat.get('topics',{})

def normalizeName(name): #normalize names to match with SQL output, see example here: https://quarry.wmflabs.org/query/23214
    if name.startswith("Wikipedia:"):
        name =name[10:]
    if not name.startswith("WikiProject"):
        name = "WikiProject_"+name
    name = name+"_articles"
    name = name.replace(' ','_')
    return name
    

root= {'topics':tree,'name':'root'}

visited = []
G = nx.Graph()
stack = [root]
while stack:
    #print(stack)
    current = stack.pop()
    childs = getLinks(current)
    #print(childs)
    visited.append(current)
    #print(current)
    for topic,topicDict in childs.items():
            if topicDict['name'] != current['name']: #avoid selfloops
                G.add_edge(topicDict['name'],current['name'])
            if c not in visited+stack:
                stack.append(topicDict)
            

nx.algorithms.is_tree(G)

len(nx.cycle_basis(G))

nx.cycle_basis(G)

nx.shortest_path_length(G,'Wikipedia:WikiProject India','Wikipedia:WikiProject Bangladesh')

nx.shortest_path_length(G,'Wikipedia:WikiProject India','Wikipedia:WikiProject Brazil')



