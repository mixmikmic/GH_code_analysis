# graph

from collections import namedtuple
from enum import Enum

import numpy as np
from nltk.util import skipgrams


"""
Node is a (NodeType, str) pair where str is the node's value

Connections is a dictionary from Node to set of Node
representing the connections between nodes in a graph

Tags is a dictionary from Node to numpy.ndarray representing weights
"""

Graph = namedtuple('Graph', ['connections', 'learned_tags', 'seed_tags'])
"""
:attr connections: graph connections
:type connections: Connections

:attr learned_tags: learned tags
:type learned_tags: Tags

:attr seed_tags: tags given to seed the algorithm
:type seed_tags: Tags
"""


class Tag(Enum):
    positive = 1
    negative = 2
    

class NodeType(Enum):
    message = 1
    feature = 2
    
    
def add_messages(graph, messages, tag=None):
    """adds message nodes to the graph
    
    :param graph: graph
    :type: Graph
    
    :param messages: messages to add
    :type messages: list of str
    
    :param tag: optional seed tag for the message
    :type tag: 
    
    :returns: updated graph
    :rtype: Graph
    """
    
    for message in messages:
        graph = add_message(graph, message, tag)

    return graph

    
def add_message(graph, message, tag=None):
    """adds a message node to the graph
    
    :param graph: graph
    :type: Graph
    
    :param message: message to add
    :type message: str
    
    :param tag: optional seed tag for the message
    :type tag: 
    
    :returns: updated graph
    :rtype: Graph
    """
    
    node = (NodeType.message, message)
    
    graph = connect_to_features(graph, node)
    graph = uniform_tag(graph, node) if tag is None else seed_tag(graph, node, tag)
    
    return graph


def connect_to_features(graph, node):
    """connects a node to features of that node
    
    :param graph: graph
    :type graph: Graph
    
    :param node: node to connect
    :type node: Node
    
    :returns: updated graph
    :rtype: Graph
    """
    
    features = compute_features(get_value(node))
    
    for feature in features:
        feature_node = (NodeType.feature, feature)
        
        graph = connect(graph, node, feature_node)
        graph = uniform_tag(graph, feature_node)
        
    return graph
    
    
def connect(graph, a, b):
    """adds a connection between a message and a feature
    
    :param graph: graph
    :type graph: Graph
    
    :param a: node
    :type a: Node
    
    :param b: node
    :type b: Node
    
    :returns: updated connections
    :rtype: Connections
    """
    
    graph.connections[a] = graph.connections.get(a, set()).union({b})
    graph.connections[b] = graph.connections.get(b, set()).union({a})
    
    return graph


def compute_features(text):
    """computes text features for text
    
    :param text: text to compute features
    :type text: str
    
    :returns: features
    :rtype: set of str
    """
    
    sep = '_'
    max_length = 3
    max_skip = 3
    
    words = text.split()
    grams = (
        feat 
        for n_gram in xrange(2, max_length+1)
        for feat in skipgrams(words, n_gram, max_skip)
    )
    
    return map(sep.join, grams) + words
        

def uniform_tag(graph, node):
    """sets the tag distribution for a node to be uniform
    
    :param graph: graph
    :type graph: Graph
    
    :param node: node
    :type node: Node
    
    :returns: updated graph
    :rtype: Graph
    """
    
    graph.learned_tags[node] = get_uniform_tags()
    return graph


def seed_tag(graph, node, seed_tag):
    """seeds the graph with the given tag
    
    :param graph: graph
    :type graph: Graph
    
    :param node: node
    :type node: Node
    
    :param seed_tag: seed tag
    :type seed_tag: Tag
    
    :returns: updated graph
    :rtype: Graph
    """
    
    onehot = np.array([1.0 if tag is seed_tag else 0.0 for tag in Tag])
    
    graph.learned_tags[node] = onehot
    graph.seed_tags[node] = onehot
    
    return graph


## helpers ##

def get_value(node): return node[1]
def get_type(node): return node[0]

def message_node(message): return (NodeType.message, message)
def feature_node(feature): return (NodeType.feature, feature)

def empty_graph(): return Graph(connections={}, learned_tags={}, seed_tags={})
def get_uniform_tags(): return np.ones(len(Tag), dtype='f') / len(Tag)

def print_message_tags(graph, messages):
    for msg in messages:
        tag_weights = graph.learned_tags[message_node(msg)]
        print(msg + "\n" + ("-" * len(msg)))
        for pair in zip(map(str, Tag), tag_weights):
            print pair
        print("")

# EXPANDER

def step(graph, mu_seed, mu_neighbor, mu_uniform):
    """runs one step of EXPANDER
    
    :param graph: graph
    :type graph: Graph
    
    :param mu_seed: penalty for deviating from seed tags
    :type mu_seed: float
    
    :param mu_neighbor: penalty for deviating from neighbor tags
    :type mu_neighbor: float
    
    :param mu_uniform: penalty for deviating from a uniform prior
    :type mu_uniform: float
    
    :returns: updated graph
    :rtype: Graph
    """

    updated_tags = {}
    nodes = graph.connections.keys()

    for node in nodes:
        updated_tags[node] = calculate_updated_tags(
            graph, node, mu_seed, mu_neighbor, mu_uniform)
        
    return Graph(
        connections=graph.connections,
        learned_tags=updated_tags,
        seed_tags=graph.seed_tags)


def n_step(n, graph, mu_seed, mu_neighbor, mu_uniform, early_stop=True):
    """runs n steps of EXPANDER
    
    :param n: number of times to step
    :type n: int
    
    :param graph: graph
    :type graph: Graph
    
    :param mu_seed: penalty for deviating from seed tags
    :type mu_seed: float
    
    :param mu_neighbor: penalty for deviating from neighbor tags
    :type mu_neighbor: float
    
    :param mu_uniform: penalty for deviating from a uniform prior
    :type mu_uniform: float
    
    :param early_stop: whether to stop once the model converges
    :type early_stop: boolean
    
    :returns: updated graph
    :rtype: Graph
    """
    
    for i in range(n):
        old_graph = graph
        graph = step(graph, mu_seed, mu_neighbor, mu_uniform)
        
        if early_stop and tags_equal(old_graph.learned_tags, graph.learned_tags):
            break
        
    return graph

def tags_equal(some_tags, other_tags):
    """whether two sets of tags are equal
    
    :param some_tags: some tags
    :type some_tags: Tags
    
    :param other_tags: other tags
    :type other_tags: Tags
    
    :return: whether the tags are equal
    :rtype: boolean
    """
    
    tol = 0.001
    return all(
        (np.isclose(some_tags[key], other_tags[key], atol=tol)).all()
        for key in some_tags.keys())
    

def calculate_updated_tags(graph, node, mu_seed, mu_neighbor, mu_uniform):
    """calculates the updated tag weights (Y_v) for the given node
    
    see figure 2 of http://arxiv.org/pdf/1512.01752v2.pdf
    
    :param graph: graph
    :type graph: Graph
    
    :param node: node to update
    :type node: Node
    
    :param mu_seed: penalty for deviating from seed tags
    :type mu_seed: float
    
    :param mu_neighbor: penalty for deviating from neighbor tags
    :type mu_neighbor: float
    
    :param mu_uniform: penalty for deviating from a uniform prior
    :type mu_uniform: float
    
    :returns: updated tag weights
    :rtype: numpy.ndarray
    """
    
    normalizer = calculate_normalizer(
        graph, node, mu_seed, mu_neighbor, mu_uniform)

    seed_term = calculate_seed_term(graph, node, mu_seed)
    neighbor_term = calculate_neighbor_term(graph, node, mu_neighbor)
    uniform_term = calculate_uniform_term(mu_uniform)
    
    return (seed_term + neighbor_term + uniform_term) / normalizer


def calculate_normalizer(graph, node, mu_seed, mu_neighbor, mu_uniform):
    """calculates the normalization constant M_v for updating tags
    
    see figure 2 of http://arxiv.org/pdf/1512.01752v2.pdf
    
    :param graph: graph
    :type graph: Graph
    
    :param node: node to update
    :type node: Node
    
    :param mu_seed: penalty for deviating from seed tags
    :type mu_seed: float
    
    :param mu_neighbor: penalty for deviating from neighbor tags
    :type mu_neighbor: float
    
    :param mu_uniform: penalty for deviating from a uniform prior
    :type mu_uniform: float
    
    :returns: normization constant
    :rtype: float
    """
    
    seed_term = mu_seed * (1.0 if node in graph.seed_tags else 0.0)
    neighbor_term = mu_neighbor * len(graph.connections[node])
    uniform_term = mu_uniform
    
    return (seed_term + neighbor_term + uniform_term)


def calculate_seed_term(graph, node, mu_seed):
    """calculates the seed update term
    
    see figure 2 of http://arxiv.org/pdf/1512.01752v2.pdf
    
    :param graph: graph
    :type graph: Graph
    
    :param node: node to update
    :type node: Node
    
    :param mu_seed: penalty for deviating from seed tags
    :type mu_seed: float
    
    :returns: seed term value
    :rtype: numpy.ndarray
    """
    
    return mu_seed * graph.seed_tags.get(node, np.zeros(len(Tag)))


def calculate_neighbor_term(graph, node, mu_neighbor):
    """calculates the neighbor update term
    
    see figure 2 of http://arxiv.org/pdf/1512.01752v2.pdf
    
    :param graph: graph
    :type graph: Graph
    
    :param node: node to update
    :type node: Node
    
    :param mu_neighbor: penalty for deviating from neighbor tags
    :type mu_neighbor: float
    
    :returns: neighbor term value
    :rtype: numpy.ndarray
    """
        
    neighbor_tags = map(lambda n: graph.learned_tags[n], graph.connections[node])
    return mu_neighbor * np.array(neighbor_tags).sum(axis=0)    


def calculate_uniform_term(mu_uniform):
    """calculates the uniform update term
    
    see figure 2 of http://arxiv.org/pdf/1512.01752v2.pdf
    
    :param mu_uniform: penalty for deviating from a uniform prior
    :type mu_uniform: float
    
    :returns: uniform term value
    :rtype: numpy.ndarray
    """
    
    return mu_uniform * get_uniform_tags()

# demo

# initialize graph
graph = empty_graph()

# specify messages
positive_messages = [
    'i am happy',
    'sun is shining',
    'flowers smell pretty',
]

negative_messages = [
    'i am not happy',
    'this is sad',
    'ground is wet',
]
    
untagged_messages = [
    "I am indifferent",
    "sun makes me happy",
    "rain is sad",
    "It will rain",
]

# add to graph
add_messages(graph, positive_messages, tag=Tag.positive)
add_messages(graph, negative_messages, tag=Tag.negative)
add_messages(graph, untagged_messages)

# step
n = 1000
mu_seed = 100
mu_neighbor = 50
mu_uniform = 1

graph = n_step(n, graph, mu_seed, mu_neighbor, mu_uniform)

# print learned tags
print_message_tags(graph, untagged_messages)



