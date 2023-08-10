from rmgpy.rmg.model import CoreEdgeReactionModel
from rmgpy.chemkin import loadChemkinFile
import os

# chemkin model name
mech = 'dbds_new'

path = os.path.abspath('../')
mechPath = path + '/data/' + mech
chemkinPath= mechPath + '/chem.inp'
dictionaryPath = mechPath + '/species_dictionary.txt'

model = CoreEdgeReactionModel()
model.core.species, model.core.reactions = loadChemkinFile(chemkinPath,dictionaryPath)

# generate paris for reactions that don't have flux pairs
for rxn in model.core.reactions:
    if not rxn.pairs: rxn.generatePairs()

import networkx as nx
import matplotlib.pyplot as plt
from extractInfoFromckcsv import getConcentrationDictFromCKCSV, getROPFromCKCSV, getFluxGraphEdgesDict
from rmgpy.chemkin import getSpeciesIdentifier
from IPython.display import display
import numpy as np
get_ipython().magic('matplotlib inline')

ckcsvPath= mechPath + '/CKSoln.ckcsv'
firstColDict, spc_mf_dict = getConcentrationDictFromCKCSV(ckcsvPath)

first_col_dict, spc_total_rop_dict, spc_rop_dict = getROPFromCKCSV(ckcsvPath)

graph_edges_dict = getFluxGraphEdgesDict(spc_rop_dict, model.core.reactions)

graph_edges_dict_simple = {}
for pair in graph_edges_dict:
    node1 = getSpeciesIdentifier(pair[0])
    node2 = getSpeciesIdentifier(pair[1])
    graph_edges_dict_simple[(node1, node2)] = graph_edges_dict[pair]

time_investigated = 0.1 # hour
timepoint_index = (np.abs(firstColDict['Time_(sec)']-time_investigated*3600)).argmin()
print "time is {0} secs".format(firstColDict['Time_(sec)'][timepoint_index])

G = nx.DiGraph()
for pair in graph_edges_dict:
    node1 = getSpeciesIdentifier(pair[0])
    node2 = getSpeciesIdentifier(pair[1])
    e_rawdata = graph_edges_dict[pair]
    total_flux = 0
    for rxn in e_rawdata:
        total_flux += e_rawdata[rxn][timepoint_index]
    if total_flux >= 0:
        G.add_edge(node2, node1, {"total_flux":total_flux}) # in G, positive means production of node1
    else: 
        G.add_edge(node1, node2, {"total_flux":-total_flux}) # in G, negative means consumption of node1      

spc_mf_dict['DBDS(1)'][timepoint_index]

paths = list(nx.all_simple_paths(G, source="PDD(1)", target="RAD1(14)", cutoff=5))

path_fluxes = []
for i, path in enumerate(paths):
#     print i, path
    path_steps = len(path) - 1
    fluxes = [G[path[step]][path[step+1]]['total_flux'] for step in range(path_steps) ]
    path_fluxes.append(min(fluxes))
sorted_path_fluxes = sorted(path_fluxes)
print sorted_path_fluxes[-1], path_fluxes.index(sorted_path_fluxes[-1])

path = paths[0]
path_steps = len(path) - 1
for step in range(path_steps):
    step_pair = (path[step], path[step+1])
    h_abs_rxns = []
    disp_rxns = []
    
    print "\n"
    print "**********Step{0}: {1} --> {2}: {3}*************".    format(step, step_pair[0], step_pair[1], G[step_pair[0]][step_pair[1]]['total_flux'])
    if step_pair not in graph_edges_dict_simple:
        step_pair = (step_pair[1], step_pair[0])
                
    for rxn in graph_edges_dict_simple[step_pair]:
        if rxn.family == "H_Abstraction":
            h_abs_rxns.append(rxn)
        elif rxn.family == "Disproportionation":
            disp_rxns.append(rxn)
        else:
            display(rxn)
            print "rxn#{0}: ".format(rxn.index) + str(rxn)
    if len(h_abs_rxns) > 0: 
        display(h_abs_rxns[0])
        print "rxn#{0}(1/{1} H_Abs): ".format(h_abs_rxns[0].index, len(h_abs_rxns)) + str(h_abs_rxns[0])
    if len(disp_rxns) > 0: 
        display(disp_rxns[0])
        print "rxn#{0}(1/{1} Disp): ".format(disp_rxns[0].index, len(disp_rxns)) + str(disp_rxns[0])

newG = G.subgraph(paths[0])
nx.draw(newG, with_labels=True)

source = "DBDS(1)"
# print "total product flux for {0} is {1}.".format(source, spc_total_rop_dict[source][1][timepoint_index])
depth = 1
current_node = source
path_top_list = [0, 0, 0, 0, 0, 0, 0, 0]
for step in range(depth):
    print "\n"    
    nextNode_flux_list = [(next_node, G[current_node][next_node]['total_flux']) for next_node in G[current_node]]
    sorted_nextNode_flux_list = sorted(nextNode_flux_list, key=lambda tup: -tup[1])
    
    # choose the top one as next node
    tup = sorted_nextNode_flux_list[path_top_list[step]]
    next_node = tup[0]
    step_flux = tup[1]
    
    print "**********Step{0}: {1} --> {2}: {3} mol/cm3/s*************".    format(step, current_node, next_node, step_flux)
    
    step_pair = (current_node, next_node)
    if step_pair not in graph_edges_dict_simple:
        step_pair = (next_node, current_node)
    
    h_abs_rxns = []
    disp_rxns = []
    for rxn in graph_edges_dict_simple[step_pair]:
        if rxn.family == "H_Abstraction":
            h_abs_rxns.append(rxn)
        elif rxn.family == "Disproportionation":
            disp_rxns.append(rxn)
        else:
            display(rxn)
            print rxn.family
            print "rxn#{0}: {1}: {2} ".format(rxn.index, str(rxn), graph_edges_dict_simple[step_pair][rxn][timepoint_index])
    if len(h_abs_rxns) > 0: 
        display(h_abs_rxns[0])
        print "rxn#{0}(1/{1} H_Abs): ".format(h_abs_rxns[0].index, len(h_abs_rxns)) + str(h_abs_rxns[0])
    if len(disp_rxns) > 0: 
        display(disp_rxns[0])
        print "rxn#{0}(1/{1} Disp): ".format(disp_rxns[0].index, len(disp_rxns)) + str(disp_rxns[0])
    
    current_node = next_node

print step_pair

total_flux = 0
rxn_flux_tups = []
for rxn in h_abs_rxns + disp_rxns:
    flux = graph_edges_dict_simple[step_pair][rxn][timepoint_index]
    
    rxn_flux_tups.append((rxn, flux))

rxn_flux_tups = sorted(rxn_flux_tups, key=lambda tup: tup[1], reverse=False)
for tup in rxn_flux_tups:
    rxn = tup[0]
    flux = tup[1]
    if flux > 0.1e-9:
        total_flux += flux
        print "**********************************************************************************"
        display(rxn) 
        print "rxn#{0}: {1}: {2} ".format(rxn.index, str(rxn), flux) # positive flux means production of pair node1
print "***************************************"
print "TOTAL flux from h_abs and disp is {0}.".format(total_flux)
print len(h_abs_rxns + disp_rxns)

target = "A3yl(61)"
print "total product flux for {0} is {1}.".format(target, spc_total_rop_dict[target][1][timepoint_index])
depth = 2
current_node = target
path_top_list = [1, 0, 0, 0, 0]
for step in range(depth):
    print "\n"
    prev_nodes = []
    for node1 in G:
        if current_node in G[node1]:
            prev_nodes.append(node1)
    prevNode_flux_list = [(prev_node, G[prev_node][current_node]['total_flux']) for prev_node in prev_nodes]
    sorted_prevNode_flux_list = sorted(prevNode_flux_list, key=lambda tup: -tup[1])
    
    # choose the top one as next node
    tup = sorted_prevNode_flux_list[path_top_list[step]]
    prev_node = tup[0]
    step_flux = tup[1]
    
    print "**********Step{0}: {1} <-- {2}: {3}*************".    format(step, current_node, prev_node, step_flux)
    
    step_pair = (prev_node, current_node)
    if step_pair not in graph_edges_dict_simple:
        step_pair = (current_node, prev_node)
    
    h_abs_rxns = []
    disp_rxns = []
    for rxn in graph_edges_dict_simple[step_pair]:
        if rxn.family == "H_Abstraction":
            h_abs_rxns.append(rxn)
        elif rxn.family == "Disproportionation":
            disp_rxns.append(rxn)
        else:
            display(rxn)
            print "rxn#{0}: ".format(rxn.index) + str(rxn)
    if len(h_abs_rxns) > 0: 
        display(h_abs_rxns[0])
        print "rxn#{0}(1/{1} H_Abs): ".format(h_abs_rxns[0].index, len(h_abs_rxns)) + str(h_abs_rxns[0])
    if len(disp_rxns) > 0: 
        display(disp_rxns[0])
        print "rxn#{0}(1/{1} Disp): ".format(disp_rxns[0].index, len(disp_rxns)) + str(disp_rxns[0])
    
    current_node = prev_node

print step_pair
total_flux = 0
rxn_flux_tups = []
for rxn in h_abs_rxns + disp_rxns:
    flux = graph_edges_dict_simple[step_pair][rxn][timepoint_index]
    
    rxn_flux_tups.append((rxn, flux))

rxn_flux_tups = sorted(rxn_flux_tups, key=lambda tup: tup[1], reverse=True)
for tup in rxn_flux_tups:
    rxn = tup[0]
    flux = tup[1]
    if abs(flux) > 0e-9:
        total_flux += flux
        print "**********************************************************************************"
        display(rxn) 
        print "rxn#{0}: {1}: {2} ".format(rxn.index, str(rxn), flux) # positive flux means production of pair node1
print "***************************************"
print "TOTAL flux from h_abs and disp is {0}.".format(total_flux)



