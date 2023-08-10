'''Code Setup'''
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
        self.counter=0
        self.roots=[]
    
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
        

''' Core Reduction Code'''
def change_color(G,node,col):
    for child in G.G[node]:
        G.nodecolors[node][child]=col
        change_color(G,child,col)   
        
def add_chain(G,node,length,color='grey'):
    if(length==0):
        return
    G.add_edge(node,G.counter,color)
    G.counter+=1
    add_chain(G,G.counter-1,length-1,color)
    
def dfs(G,node):
    if len(G.G[node])==0:
        return 1
    sz=0
    xor_value=0
    for child in G.G[node]:
        chain_len=dfs(G,child)
        sz+=chain_len
        xor_value^=chain_len
    if len(G.G[node])==1:
        return sz+1
    G.drawGraph()
    change_color(G,node,'purple')
    G.drawGraph()
    adj_list=deepcopy(G.G[node])
    for child in adj_list:
        G.remove_edge(node,child)
    if xor_value==0:
        return 1
    add_chain(G,node,xor_value)
    G.drawGraph()
    change_color(G,node,'g')
    return xor_value+1

def reduce_stalks(G):
    xor_value=0
    for root in G.roots:
        val=len(list(nx.dfs_edges(G.G,root)))
        xor_value^=val
        change_color(G,root,'purple')
    G.drawGraph() 
    temproot=deepcopy(G.roots)
    for root in temproot:
        G.remove_node(root)
        G.roots.remove(root)
    G.counter+=1
    G.roots.append(G.counter-1)
    if(xor_value==0):
        print()
        G.add_edge(0,0)
        G.drawGraph()
        return
    add_chain(G,G.counter-1,xor_value)
    G.drawGraph()    
    change_color(G,G.roots[0],'g')
    G.drawGraph()   

def reduce_board(G):
    for root in G.roots:
        dfs(G,root)
    G.drawGraph()
    if len(G.roots)<=1:
        return
    reduce_stalks(G)
    

'''Board Generator Code'''

def dfs_add(G,node,ans,parent):
    for child in G[node]:
        if child==parent:
            continue
        ans.add_edge(node+ans.counter,child+ans.counter,'g')
        dfs_add(G,child,ans,node)
        
def convert_graph(G,ans):
    ans.roots.append(ans.counter)
    dfs_add(G,0,ans,-1)
    ans.counter+=len(G.node())
    
def get_forest(nlist):
    ans=Graph()
    for i in nlist:
        convert_graph(nx.random_tree(i),ans)
    return ans

def get_stalks(nlist):
    ans=Graph()
    for i in nlist:
        root=ans.counter
        ans.roots.append(root)
        ans.counter+=1
        add_chain(ans,root,i,color='green')
    return ans

G = get_forest([8])            
reduce_board(G)

example_stalks=get_stalks([1,2,3,2,4])
example_stalks.drawGraph()

multi_stalk_example=get_stalks([1,4,3])
reduce_board(multi_stalk_example)

second_player_win=get_stalks([6,5,3])
reduce_board(second_player_win)

example_tree=get_forest([10,9,8])
example_tree.drawGraph()

test_colon=get_stalks([5,6,4])
new_root=test_colon.counter
for root in test_colon.roots:
    test_colon.add_edge(new_root,root,'g')
test_colon.roots=[new_root]
test_colon.counter+=1
reduce_board(test_colon)

colon_test_2=get_stalks([6,7,5])
reduce_board(colon_test_2)

test_tree=get_forest([10])
reduce_board(test_tree)

rt1=get_forest([5,6,7,8])
reduce_board(rt1)

