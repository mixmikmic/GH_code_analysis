# Tools for parsing xml
import xml.etree.ElementTree as ET

# Load the xml file
# If there are error during the loading, make sure the file is encoded in UTF8
# You may have to open it with a text editor and save it with encoding UTF8
xml_file_to_load = 'tolskeletaldumpUTF8.xml'
tree = ET.parse(xml_file_to_load)

# The data will be loaded in a networkx graph
# The networkx module can be installed using 'pip install networkx'
import networkx as nx

# Code for the tree construction
i=1
G = nx.DiGraph()
root = tree.getroot()
for livingElement in root.iter('NODE'):
    name = livingElement.find('NAME').text
    data_dic = livingElement.attrib
    node_id = data_dic['ID']
    if name == None:
        name = 'None'
    data_dic['name'] = name
    if not G.has_node(node_id):
        G.add_node(node_id,data_dic)
    if data_dic['CHILDCOUNT']!='0':
        for child in livingElement[1]:
            child_name = child.find('NAME').text
            child_data_dic = child.attrib
            child_id = child_data_dic['ID']
            if child_name == None:
                child_name = 'None'
            child_data_dic['name'] = child_name
            #print(child_name,child_data_dic)
            if not G.has_node(child_id):
                G.add_node(child_id,child_data_dic)
            if G.has_edge(node_id,child_id):
                print('found exisiting edge',name,child_name)
                print('data: ',data_dic,child_data_dic)
            G.add_edge(node_id,child_id,weight=1)
            i+=1
print('Number of nodes processed:',i)
print('Number of nodes in the graph:',G.number_of_nodes())
print('Number of edges in the graph:',G.number_of_edges())
print('The graph is a tree?',nx.is_tree(G))

# Find the root node, the only one that has in_degree 0
root_node_list = [n for n,d in G.in_degree().items() if d==0] 
root_node_id = root_node_list[0]
print('Root node id:',root_node_id)

# Details about the root node
print(G.node[root_node_id])
print('Degree:',G.degree(root_node_id))
print('Successors: ',[G.node[node]['name'] for node in G.successors(root_node_id)])

# Saving the graph in json format
from networkx.readwrite import json_graph
import json
with open('treeoflife.json', 'w') as outfile1:
    outfile1.write(json.dumps(json_graph.node_link_data(G)))

# Saving the graph in graphML format
nx.write_graphml(G, "treeoflife.graphml")



