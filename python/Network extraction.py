from itertools import combinations

names = ['Galaad', 'Lancelot', 'Perceval']
relations = combinations(names, 2)
for first, second in relations:
    print(first + ' + ' + second)

from lxml import etree

doc = etree.parse('qgraal_cm_2013-07-cour.xml')
doc.getroot()

import networkx

graph = networkx.Graph()
print(networkx.info(graph))

for sentence in doc.iter('s'):
    words = sentence.iter('w')
    name_words = [word for word in words if word.get('type') == 'NOMpro']
    names = [etree.tostring(word, method='text', encoding='unicode') for word in name_words]
    names = [name.strip() for name in names]
    for source, target in combinations(names, 2):
        graph.add_edge(source, target)

print(networkx.info(graph))

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 15, 15

networkx.draw_networkx(graph, edge_color='gray')

from collections import Counter
nodes_by_degree = Counter(graph.degree())
nodes_by_degree.most_common(10)

networkx.write_graphml(graph, 'Graal.graphml')

