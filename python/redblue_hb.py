'''initial code setup'''
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
get_ipython().run_line_magic('matplotlib', 'inline')
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout, to_agraph
import matplotlib.pyplot as plt
from copy import deepcopy
class Graph:
    def __init__(self):
        self.nodecolors = {}
        self.G = nx.DiGraph()
    
    def drawGraph(self):
        plt.gca().invert_yaxis()
        pos =graphviz_layout(self.G, prog='dot')
        edges = self.G.edges()
        colors = [self.nodecolors[u][v] for u,v in edges]
        nx.draw(self.G, pos, with_labels=False, arrows=False, edge_color=colors, node_color='black', width=5)
        plt.show()

    def add_edge(self, From, To, color='black'):
        self.G.add_edge(From,To)
        if From in self.nodecolors:
            self.nodecolors[From][To] = color
        else:
            self.nodecolors[From] = {To:color}

    def createGraph(self):
        self.G.add_node("ROOT")
        for i in range(5):
            self.G.add_node("Child_%i" % i)
            self.G.add_node("Grandchild_%i" % i)
            self.G.add_node("Greatgrandchild_%i" % i)
            self.add_edge("ROOT", "Child_%i" % i,'b')
            self.add_edge("Child_%i" % i, "Grandchild_%i" % i,'r')
            self.add_edge("Grandchild_%i" % i, "Greatgrandchild_%i" % i,'b')
    
    def remove_node(self,node):
        if node in self.G.nodes():
            nodes = deepcopy(self.G[node])
            for v in nodes:
                self.remove_edge(node,v)
            self.G.remove_node(node)
    
    def remove_edge(self,From,To):
        if From in self.G and To in self.G[From]:
            self.G.remove_edge(From,To)
            self.nodecolors[From].pop(To,None)
            self.remove_node(To)

G = Graph()            
G.createGraph()
G.drawGraph()
#left player moves
#disconnects one of the components from the root, all the components that got disconnected from root also got removed
G.remove_edge('ROOT','Child_1')
G.drawGraph()
#right player moves 
#removes an intermediate edge
G.remove_edge('Child_3','Grandchild_3')
G.drawGraph()
#left player moves
#one of the top edges get deleted, so only one node is removed.
G.remove_edge('Grandchild_2','Greatgrandchild_2')
G.drawGraph()

G = Graph()
G.add_edge(0,1,'r')
G.add_edge(1,2,'r')
G.add_edge(2,3,'r')
G.add_edge(3,4,'b')
G.add_edge(4,5,'b')
G.add_edge(5,6,'r')
G.add_edge(7,8,'b')
G.add_edge(8,9,'r')
G.add_edge(10,11,'r')
G.add_edge(11,12,'r')
G.add_edge(12,13,'r')
for i in range(14,20):
    G.add_edge(i,i+1,'b')
G.drawGraph()

