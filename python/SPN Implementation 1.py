import numpy as np
import tensorflow as tf
class ProdLayer:
    def __init__(self, id, connections):
        self.t = 'P'
        self.id = id
        self.connections = connections
        self.results = [0]*len(connections)
        
class SumLayer:
    def __init__(self, id, connections, weights):
        self.t = 'S'
        self.id = id
        self.connections = connections
        self.results = [0]*len(connections)
        self.weights = weights
        
class InputLayer:
    def __init__(self, id, size):
        self.id = id
        self.t = 'I'
        self.results = [0]*size
        
class SPN:
    def __init__(self):
        self.input_pls = [];
        self.input_nums = [];
        self.tensors = [];
        self.layers = [];
        self.weights = [];
        self.labels = [];
        self.tf_indices = []
        self.tf_weights = [];
        self.loss = None;
        self.optimizer = None;
        self.session = None;
        self.out = None;
        self.inp = None;
        self.writer = None;
        self.tf_shapes = []
    def add_layer(self, name, connections=[], weights=[], size=0):
        if name == 'I':
            assert len(self.layers) == 0
            assert size > 0
            self.layers.append(InputLayer(0, size))
        elif name == 'S':
            assert len(weights) > 0
            assert len(connections) > 0
            assert len(weights) == len(connections)
            self.layers.append(SumLayer(len(self.layers)-1, 
                                        connections, 
                                        weights))
        elif name == 'P':
            assert len(connections) > 0
            self.layers.append(ProdLayer(len(self.layers)-1, 
                                        connections))
            
    def compute_sum(self, layer):
        for i in range(len(layer.weights)):
            s = 0
            for j in range(len(layer.weights[i])):
                a = layer.connections[i][j][0]
                b = layer.connections[i][j][1]
                w = layer.weights[i][j]
                s += w*(self.layers[a].results[b])
            layer.results[i] = s
    
    def compute_product(self, layer):
        for i in range(len(layer.connections)):
            p = 1
            for j in range(len(layer.connections[i])):
                a = layer.connections[i][j][0]
                b = layer.connections[i][j][1]
                p *= (self.layers[a].results[b])
            layer.results[i] = p    
            
    def compute_slow(self, inp):
        self.layers[0].results = inp
        for L in self.layers[1:]:
            if L.t == 'P':
                self.compute_product(L)
            else:
                self.compute_sum(L)
    
    def get_val(self):
        return self.layers[-1].results   
    
    def build_product_matrix(self, layer, mat):
        connections =  layer.connections
        print mat
        for i in range(len(connections)):
            for r, n in connections[i]:
                mat[i, n] = 1
        return mat

    def build_sum_matrix(self, layer, mat):
        connections =  layer.connections
        weights = layer.weights
        for i in range(len(connections)):
            j = 0
            for r, n in connections[i]:
                mat[i, n] = weights[i][j]
                j += 1
        return mat
    
    def build_sparse_matrix_sum(self, layer, shape):
        connections =  layer.connections
        weights = layer.weights
        indz = []
        weightz = []
        for i in range(len(connections)):
            j = 0
            for r, n in connections[i]:
                weightz.append(0.5)
                indz.append([i, n])
                j += 1
        return weightz, indz
    
    def build_sparse_matrix_prod(self, layer, shape):
        connections =  layer.connections
        indz = []
        weightz = []
        for i in range(len(connections)):
            j = 0
            for r, n in connections[i]:
                weightz.append(1.0)
                indz.append([i, n])
                j += 1
        return weightz, indz
    
    def initialize_np(self):
        self.weights = []
        self.sizes = []
        for layer in self.layers:
            self.sizes.append(len(layer.results))
            self.labels.append(layer.t)
        weights = []
        for i in range(len(self.sizes) - 1):
            mat = np.matrix(np.zeros((self.sizes[i+1], self.sizes[i])))
            if self.layers[i+1].t == 'S':
                weights.append(self.build_sum_matrix(self.layers[i+1], mat))
            else:
                weights.append(self.build_product_matrix(self.layers[i+1], mat))
        self.weights = weights
        
    def initialize_tf(self):
        self.tf_indices = []
        self.tf_weights = []
        self.sizes = []
        self.labels =  []
        self.tf_shapes = []
        for layer in self.layers:
            self.sizes.append(len(layer.results))
            self.labels.append(layer.t)
        weights = []
        inds = []
        shapes = []
        for i in range(len(self.sizes) - 1):
            shape = [self.sizes[i+1], self.sizes[i]]
            if self.layers[i+1].t == 'S':
                w, ix = self.build_sparse_matrix_sum(self.layers[i+1], shape)
                weights.append(tf.Variable(w, dtype=tf.float64))
            else:
                w, ix = self.build_sparse_matrix_prod(self.layers[i+1], shape)
                weights.append(tf.Variable(w, dtype=tf.float64, trainable=False))
            inds.append(tf.constant(ix, dtype=tf.int64))
            shapes.append(tf.constant(shape, dtype=tf.int64))
        self.tf_weights = weights
        self.tf_shapes = shapes
        self.tf_indices = inds
        self.build_graph()
  
    def compute_np(self, inp):
        inp = np.matrix(inp).T
        curr = inp
        for i in range(1, len(self.labels)):
            if self.labels[i] == 'S':
                curr = self.weights[i-1]*curr
            else:
                curr = np.exp(self.weights[i-1]*np.log(curr))
#         print curr

    def build_graph(self):
        self.inp = tf.placeholder(tf.float64, shape=(4, None))
        curr = self.inp
        self.tensors = []
        for i in range(1, len(self.labels)):
            print curr.get_shape()
            mat = tf.SparseTensor(self.tf_indices[i-1], tf.identity(tf.nn.relu(self.tf_weights[i-1])), self.tf_shapes[i-1])
            self.tensors.append(mat)
            if self.labels[i] == 'S':
                curr = tf.sparse_tensor_dense_matmul(mat, curr)
            else:
                curr = tf.exp(tf.sparse_tensor_dense_matmul(mat, tf.log(curr)))
        self.out = curr
#         self.loss = -tf.log(self.out)
        lam = tf.constant(10000, dtype=tf.float64)
        one = tf.constant(1, dtype=tf.float64)
        value_loss = tf.reduce_sum(-tf.log(self.out))
        lambda_loss = 0#tf.reduce_sum([tf.reduce_sum(tf.sub(tf.sparse_reduce_sum(x, 0), one)) for x in self.tensors])
        self.loss = value_loss#tf.add(value_loss, tf.abs(tf.mul(lam, lambda_loss)))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
    
    def start_session(self):
        self.init = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(self.init)
    
    def compute_tf(self, inp):
        output = self.out.eval(session=self.session, feed_dict={self.inp: inp})
        if self.writer != None:
            self.writer.add_summary(summary)
        print output
    
    def close_session(self):
        self.session.close()
        self.session = None;
    
    def train(self, d, epochs):
        for e in range(epochs):
            _, l, predictions = self.session.run([self.out, self.loss, self.optimizer], feed_dict={self.inp: d})
            print l, _
                
    
    def show_graph(self):
        g = a.Graph()
        points = []
        for layer in self.layers:
            la = []
            for j in range(len(layer.results)):
                la.append(g.add_vertex())
            points.append(la)
        
        for j, layer in enumerate(self.layers[1:]):
            for i, con in enumerate(layer.connections):
                for c in con:
                    g.add_edge(points[j+1][i], points[c[0]][c[1]])
        draw.graph_save(g, vertex_text=g.vertex_index, vertex_font_size=18, output_size=(200, 200), output="two-nodes.png")

data = [[[1], [0], [1], [0]], [[1], [0], [1], [0]], [[0], [1], [0], [1]]]
data = np.array([[1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]])
SPN1 = SPN()
SPN1.add_layer('I', size=4)
c1 = [[(0, 0), (0, 1)],
      [(0, 0), (0, 1)],
      [(0, 2), (0, 3)],
      [(0, 2), (0, 3)]]
w1 = [
    [0.6, 0.4],
    [0.9, 0.1],
    [0.3, 0.7],
    [0.2, 0.8],
]
c2 = [
    [(1, 0), (1, 2)],
    [(1, 0), (1, 3)],
    [(1, 1), (1, 3)],
     ]
c3 = [
    [(2, 0), (2, 1), (2, 2)]
]
w3 = [
    [0.5, 0.2, 0.3]
]
SPN1.add_layer('S', c1, w1)
SPN1.add_layer('P', c2)
SPN1.add_layer('S', c3, w3)
SPN1.initialize_tf()
SPN1.start_session()
SPN1.train(data.T, 10)
SPN1.compute_tf([[1], [0], [1], [0]])
SPN1.compute_tf([[1], [0], [0], [1]])
SPN1.train(data.T, 100)
print 'Values'
SPN1.compute_tf([[1], [0], [1], [0]])
SPN1.compute_tf([[1], [0], [0], [1]])
SPN1.compute_tf([[1], [1], [1], [1]])



for w in SPN1.tf_weights:
    print w.eval(session=SPN1.session)

SPN2 = SPN()
SPN2.add_layer('I', size=6)
c1 = [
    [(0, 0)],
    [(0, 1), (0, 2)],
    [(0, 1), (0, 2)],
    [(0, 3), (0, 4)],
    [(0, 3), (0, 4)],
    [(0, 5)]
]

w1 = [
    [1],
    [0.1, 0.9],
    [0.2, 0.8],
    [0.5, 0.5],
    [0.3, 0.7],
    [1]
]

c2 = [
    [(1, 0), (1, 1), (1, 3)],
    [(1, 2), (1, 4), (1, 5)]
]

c3 = [
    [(2, 0), (2, 1)]
]
w3 = [
    [0.95, 0.05]
]

SPN2.add_layer('S', c1, w1)
SPN2.add_layer('P', c2)
SPN2.add_layer('S', c3, w3)

SPN2.compute_slow([1]*6)
SPN2.get_val()

print SPN1.weights

get_ipython().magic('timeit SPN1.compute_slow([1]*4)')

get_ipython().magic('timeit SPN1.compute_np([1]*4)')

get_ipython().magic('timeit SPN1.compute_tf([[1]]*4)')

from graph_tool.all import graph_draw,Graph  

#create your graph object
g = Graph()

#add a vertex at least
g.add_vertex()

#draw you graph 
graph_draw(
    g,
    output_size=(200,200),
    output="test.png"
)   

SPN1.show_graph()

tmovie = SPN()

my_file = open('tmovie.spn.txt', 'r')

t = 1
lines = []
while t != '':
    t = my_file.readline()
    lines.append(t[:-1])



Leaves = []
prods = []
sums = []
nons = []
for l in lines:
    if 'PRD' in l:
        prods.append(l)
    elif 'SUM' in l:
        sums.append(l)
    elif 'LEAVE' in l:
        Leaves.append(l)
        if len(l.split(',')) != 5:
            print l
    else:
        nons.append(l)

class SumNode:
    def __init__(self, id):
        self.id = id;
        self.children = []
        self.parents = []
        self.weights = []
        self.rank = 0
        self.Trank = 0
        
class PrdNode:
    def __init__(self, id):
        self.id = id
        self.children = []
        self.parents = []
        self.rank = 0
        self.TRank = 0
        
class Leaf:
    def __init__(self, id, a, b, i):
        self.id = id;
        self.inp = i;
        self.children = []
        self.parents = [];
        self.weights = [a, b];
        self.rank = 1;
        self.TRank = 0;
        
big_dict = {}

n = 0
for i in range(len(lines)):
    if "EDGES" in lines[i]:
        n = i;
        break;

nodez = lines[0:n]
edgez = lines[n+1:]

big_dict = {}
Leaves = []
Prods = []
Sums = []
for l in nodez:
    if 'PRD' in l:
        arr = l.split(',')
        node = PrdNode(arr[0])
        big_dict[arr[0]] = node
        Prods.append(arr[0])
#         print 'hi'
    elif 'SUM' in l:
        arr = l.split(',')
        node = SumNode(arr[0])
        big_dict[arr[0]] = node
        Sums.append(arr[0])
    elif 'LEAVE' in l:
        arr = l.split(',')
        node = Leaf(arr[0], arr[3], arr[4], arr[2])
        big_dict[arr[0]] = node
        Leaves.append(arr[0])
#     else:
#         print n

for e in edgez :
    a = e.split(',')
    if a[0] == '' or a[1] == '':
        continue
    big_dict[a[0]].children.append(a[1])
    big_dict[a[1]].parents.append(a[0])
    if len(a) == 3:
        big_dict[a[0]].weights.append(a[2])

currs = set(Leaves)
rank = 1
while len(currs) > 0:
    prev_currs = currs
    new_currs = set()
    for s in list(currs):
        for p in big_dict[s].parents:
            new_currs.add(p)
        big_dict[s].rank = rank
    currs = new_currs
    rank += 1
orank = rank
print orank
rank -= 1
currs = prev_currs
while len(currs) > 0:
    new_currs = set()
    for s in list(currs):
        for p in big_dict[s].children:
            new_currs.add(p)
        big_dict[s].TRank = rank
    currs = new_currs
    rank -= 1

node_list = [[] for x in range(0, orank)]
new_dict = {}
for k in big_dict.keys():
    n = big_dict[k]
    print n.TRank
    node_list[n.TRank].append(n)
    new_dict[k] = (n.TRank-1, len(node_list[n.rank-1]) -1)

for i in range(len(node_list)):
    for j in range(len(node_list[i])):
        node_list[i][j].parents = map(lambda x: new_dict[x], node_list[i][j].parents)

connections = []
weights = []
for n in node_list[1:]:
    conns = []
    wz = []
    maxx = 0
    minn = 10000000
    for m in n:
        if isinstance(m, SumNode):
            wz.append(m.weights)
        maxx = max(maxx, len(m.children))
        minn = min(minn, len(m.children))
        conns.append(map(lambda x: new_dict[x], m.children))
    connections.append(conns)
    weights.append(wz)
    print minn, maxx, len(n), n[0]

Inputz = []
layer_inps = []
lay = 0
for n in node_list[1:]:
    lay += 1
    counter = 0
    for m in n:
        if isinstance(m, Leaf):
            Inputz.append(m.inp)
            counter += 1;
    layer_inps.append((lay, counter))

#tensorflow graph
inputs = tf.placeholder(dtype=tf.float64, )

isinstance(node_list[0][0], SumNode)

node_list[0][0]

#tests
a = tf.Variable([3, 2])

a[a < 3] = 5



