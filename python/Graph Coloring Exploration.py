
# Install the Networkx library
get_ipython().system('pip install networkx')

# %load ../../load_magic/storage1.py

current_directory = get_ipython().getoutput('echo %cd%')
folder_list = current_directory[0].split('\\')
get_ipython().run_line_magic('run', "../../load_magic/storage2.py {len(folder_list) - folder_list.index('ipynb')}")
get_ipython().run_line_magic('who', '')


import matplotlib.pyplot as plt
import networkx as nx
import re
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

pos_dict = load_object('pos_dict')
jung_digraph = load_object('jung_digraph')


def draw_influences_digraph(influences_digraph, birth_year_based_layout):
    
    # Create x axis labels
    xtick_list = [int(round(elem)) for elem in pd.cut(np.array([1750, 2000]), 4, retbins=True)[1]]
    xtick_list = [-500] + xtick_list
    nominal_list = [-500, 0, 500, 1000, 1500, 2000]
    xticklabel_list = []
    bc_regex = re.compile(r'-(\d+)')
    for xtick in xtick_list:
        if xtick == 0:
            xticklabel_list.append('')
        elif xtick < 0:
            xticklabel_list.append(bc_regex.sub(r'\1 BC', str(xtick)))
        elif xtick > 0:
            xticklabel_list.append(str(xtick) + ' AD')
    
    # Set up the figure
    jung_fig = plt.figure(figsize=(20, 18))
    ax = jung_fig.add_subplot(111)
    plt.xticks(nominal_list, xticklabel_list, rotation=-45)
    plt.yticks([], [])

    node_collection = nx.draw_networkx_nodes(G=influences_digraph, pos=birth_year_based_layout, alpha=0.5)
    edge_collection = nx.draw_networkx_edges(G=influences_digraph, pos=birth_year_based_layout, alpha=0.5)
    labels_collection = nx.draw_networkx_labels(G=influences_digraph, pos=birth_year_based_layout, font_size=14)


# Draw the influences digraph using the birth-year-based layout
draw_influences_digraph(jung_digraph, pos_dict)


if nx.is_strongly_connected(jung_digraph):
    print(nx.center(jung_digraph, e=None))


get_ipython().run_line_magic('pprint', '')

sorted(nx.strongly_connected_components(jung_digraph), key=len, reverse=True)


jung_subgraph = jung_digraph.subgraph(max(nx.strongly_connected_components(jung_digraph), key=len))
draw_influences_digraph(jung_subgraph, pos_dict)


get_ipython().system('"C:\\Program Files\\Notepad++\\notepad++.exe" "c:\\users\\dev\\anaconda3\\lib\\site-packages\\networkx\\algorithms\\coloring\\greedy_coloring.py"')


from networkx.algorithms.coloring import greedy_coloring

strategy_list = greedy_coloring.STRATEGIES.keys()
for strategy_name in strategy_list:
    print()
    print(strategy_name)
    try:
        coloring_dictionary = nx.coloring.greedy_color(jung_digraph, strategy=strategy_name)
        print(type(coloring_dictionary))
        print(coloring_dictionary)
    except Exception as e:
        print(e)


get_ipython().run_line_magic('run', '../../load_magic/charts.py')
get_ipython().run_line_magic('who', '')


def colored_draw_influences_digraph(influences_digraph, birth_year_based_layout=None, strategy_name=None):
    
    # Create x axis labels
    xtick_list = [int(round(elem)) for elem in pd.cut(np.array([1750, 2000]), 4, retbins=True)[1]]
    xtick_list = [-500] + xtick_list
    nominal_list = [-500, 0, 500, 1000, 1500, 2000]
    xticklabel_list = []
    bc_regex = re.compile(r'-(\d+)')
    for xtick in xtick_list:
        if xtick == 0:
            xticklabel_list.append('')
        elif xtick < 0:
            xticklabel_list.append(bc_regex.sub(r'\1 BC', str(xtick)))
        elif xtick > 0:
            xticklabel_list.append(str(xtick) + ' AD')
    
    # Create node color
    if strategy_name is None:
        node_color = 'r'
    else:
        coloring_dict = nx.coloring.greedy_color(influences_digraph, strategy=strategy_name)
        if type(coloring_dict) == dict:
            node_color = [coloring_dict[node] for node in influences_digraph.nodes()]
        else:
            node_color = 'r'
    
    # Set up the figure
    jung_fig = plt.figure(figsize=(20, 18))
    ax = jung_fig.add_subplot(111)
    plt.yticks([], [])
    
    # Set the Strategy name as a text annotation and title
    cmap = r()
    if strategy_name is None:
        ax.set_title('('+cmap+')', size=18)
    else:
        ax.text(0.5, 0.01, strategy_name, transform=ax.transAxes, size=18, ha='center')
        ax.set_title(strategy_name+' ('+cmap+')', size=18)
    
    if birth_year_based_layout is None:
        pos = nx.random_layout(influences_digraph)
    else:
        pos = birth_year_based_layout
        plt.xticks(nominal_list, xticklabel_list, rotation=-45)
    node_collection = nx.draw_networkx_nodes(G=influences_digraph, pos=pos, alpha=0.5,
                                             node_color=node_color, cmap=cmap)
    edge_collection = nx.draw_networkx_edges(G=influences_digraph, pos=pos, alpha=0.5)
    labels_collection = nx.draw_networkx_labels(G=influences_digraph, pos=pos, font_size=14)


strategy_list = greedy_coloring.STRATEGIES.keys()
for strategy_name in strategy_list:
    try:
        colored_draw_influences_digraph(jung_digraph, pos_dict, strategy_name)
    except Exception as e:
        print(e)


len(jung_digraph.nodes())


def centrality_draw_influences_digraph(influences_digraph, birth_year_based_layout=None, coloring_dict=None,
                                       cmap='tab10', plot_title=''):
    
    # Create x axis labels
    xtick_list = [int(round(elem)) for elem in pd.cut(np.array([1750, 2000]), 4, retbins=True)[1]]
    xtick_list = [-500] + xtick_list
    nominal_list = [-500, 0, 500, 1000, 1500, 2000]
    xticklabel_list = []
    bc_regex = re.compile(r'-(\d+)')
    for xtick in xtick_list:
        if xtick == 0:
            xticklabel_list.append('')
        elif xtick < 0:
            xticklabel_list.append(bc_regex.sub(r'\1 BC', str(xtick)))
        elif xtick > 0:
            xticklabel_list.append(str(xtick) + ' AD')
    
    # Create node color
    if coloring_dict is None:
        node_color = 'r'
    else:
        if type(coloring_dict) == dict:
            node_color = [coloring_dict[node] for node in influences_digraph.nodes()]
        else:
            node_color = 'r'
    
    # Set up the figure
    jung_fig = plt.figure(figsize=(10, 9))
    ax = jung_fig.add_subplot(111)
    plt.yticks([], [])
    
    # Set the Strategy name as a text annotation and title
    ax.set_title(plot_title, size=18)
    
    if birth_year_based_layout is None:
        pos = nx.circular_layout(influences_digraph)
    else:
        pos = birth_year_based_layout
        plt.xticks(nominal_list, xticklabel_list, rotation=-45)
    node_collection = nx.draw_networkx_nodes(G=influences_digraph, pos=pos, alpha=0.5,
                                             node_color=node_color, cmap=cmap)
    edge_collection = nx.draw_networkx_edges(G=influences_digraph, pos=pos, alpha=0.5)
    labels_collection = nx.draw_networkx_labels(G=influences_digraph, pos=pos, font_size=10)


jung_digraph_i = nx.convert_node_labels_to_integers(jung_digraph, first_label=1, ordering='default',
                                                    label_attribute='old_label')
nx.relabel_nodes(jung_digraph_i, nx.get_node_attributes(G=jung_digraph_i, name='old_label'), copy=False)
nodes_dict_list = [nx.degree_centrality(jung_digraph_i), nx.in_degree_centrality(jung_digraph_i),
                   nx.out_degree_centrality(jung_digraph_i),
                   nx.closeness_centrality(jung_digraph_i, reverse=False),
                   nx.closeness_centrality(jung_digraph_i, reverse=True),
                   nx.betweenness_centrality(jung_digraph_i, normalized=True, endpoints=False),
                   nx.betweenness_centrality(jung_digraph_i, normalized=True, endpoints=True)]
title_list = ['degree_centrality', 'in_degree_centrality', 'out_degree_centrality',
              'closeness_centrality (reverse=False)', 'closeness_centrality (reverse=True)',
              'betweenness_centrality (endpoints=False)',
              'betweenness_centrality (endpoints=True)']
for nodes_dict, plot_title in zip(nodes_dict_list, title_list):
    centrality_draw_influences_digraph(jung_digraph_i, birth_year_based_layout=pos_dict, coloring_dict=nodes_dict,
                                       plot_title=plot_title)

get_ipython().run_line_magic('pinfo', 'nx.betweenness_centrality')


get_ipython().run_line_magic('pprint', '')

dir(nx)


edge_list = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('E', 'G'), ('F', 'G')]
G = nx.from_edgelist(edge_list, create_using=nx.Graph())
node_dict = nx.betweenness_centrality(G, k=None, normalized=False, weight=None, endpoints=False, seed=None)
node_dict



