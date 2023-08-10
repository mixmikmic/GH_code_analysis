from DSGRN import *

network = Network();
network.assign("X : X + Y \n" + "Y : ~X \n")
DrawGraph(network)

parameter_graph = ParameterGraph(network)
print(parameter_graph.size())

parameter = parameter_graph.parameter(31)

domain_graph = DomainGraph(parameter)
DrawGraph(domain_graph)

search_graph = SearchGraph(domain_graph)
DrawGraph(search_graph)

events = [("X", "min"), ("Y", "min"), ("X", "max"), ("Y", "max")]
event_ordering = [(0,2),(1,3)]
poe = PosetOfExtrema(network, events, event_ordering )
DrawGraph(poe)

pattern_graph = PatternGraph(poe);
DrawGraph(pattern_graph)

matching_graph = MatchingGraph(search_graph, pattern_graph);
DrawGraph(matching_graph)

path_match = PathMatch(matching_graph)

print(path_match)

cycle_match = CycleMatch(matching_graph)

print(cycle_match)

DrawGraphWithHighlightedPath(matching_graph, path_match)

search_graph_path_match = [ pair.first for pair in path_match ]
DrawGraphWithHighlightedPath(search_graph, search_graph_path_match)

pattern_graph_path_match = [ pair.second for pair in path_match ]
DrawGraphWithHighlightedPath(pattern_graph, pattern_graph_path_match)

