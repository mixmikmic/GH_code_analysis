import numpy
import toyplot.layout

toyplot.layout._floyd_warshall_shortest_path(2, numpy.array([[0, 1]]))

toyplot.layout._floyd_warshall_shortest_path(3, numpy.array([[0, 1], [1, 2]]))

toyplot.layout._floyd_warshall_shortest_path(3, numpy.array([[0, 1], [1, 2], [2, 0]]))

edges = numpy.array([[0, 1]])
toyplot.graph(edges, width=200);
print toyplot.layout._adjacency_list(2, edges)

edges = numpy.array([[0, 1], [1, 2]])
toyplot.graph(edges, width=200);
print toyplot.layout._adjacency_list(3, edges)

edges = numpy.array([[0, 1], [1, 2], [0, 2]])
toyplot.graph(edges, width=200);
print toyplot.layout._adjacency_list(3, edges)

edges = numpy.array([[0, 1]])
toyplot.graph(edges, width=200);
print toyplot.layout._require_tree(toyplot.layout._adjacency_list(2, edges))

edges = numpy.array([[0, 1], [0, 2]])
toyplot.graph(edges, width=200);
print toyplot.layout._require_tree(toyplot.layout._adjacency_list(3, edges))

edges = numpy.array([[0, 1], [1, 2]])
toyplot.graph(edges, width=200);
print toyplot.layout._require_tree(toyplot.layout._adjacency_list(3, edges))

edges = numpy.array([[0, 1], [1, 2], [2, 1]])
toyplot.graph(edges, width=200, layout=toyplot.layout.FruchtermanReingold(toyplot.layout.CurvedEdges()));
print toyplot.layout._require_tree(toyplot.layout._adjacency_list(3, edges))



