from snpp.utils.data import load_csv_network
from snpp.utils.graph import get_triangles
from itertools import combinations
from collections import Counter
from tqdm import tqdm
import pandas as pd

g = load_csv_network('data/soc-sign-slashdot.txt').to_undirected()

g = g.to_undirected()

tris = get_triangles(g)


freqs = Counter()
for tri in tqdm(tris):
    neg_cnt = sum(1 for u, v in combinations(tri, 2) 
                    if g[u][v]['sign'] == -1)
    freqs[neg_cnt] += 1    
print(freqs)

vals = np.array(list(freqs.values()))
cols = list(map(lambda c: "{} triangle(s)".format(c),
            freqs.keys()))
df = pd.DataFrame(data=[vals / np.sum(vals) * 100],
                  columns=cols)
print(df)

freqs = Counter()
for u, v in tqdm(g.edges_iter()):
    if set(g.adj[u]).intersection(set(g.adj[v])):
        freqs['#edge-without-triangle'] += 1
    else:
        freqs['#edge-with-triangle'] += 1

print(freqs)
vals = np.array(list(freqs.values()))
print(vals / np.sum(vals))

