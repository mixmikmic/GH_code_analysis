import sys, collections, re

from laf.fabric import LafFabric
from etcbc.preprocess import prepare
fabric = LafFabric()

source='etcbc'
version='4b'

API=fabric.load(source+version, 'lexicon', 'workshop', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        otype
        lex g_word
        sp pdp nametype ls gloss language code
        chapter verse
    ''','mother'),
    "prepare": prepare,
    "primary": False,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))

T.text(book='Genesis', chapter=1, verse="1", fmt="ha", html=False, otype="word")

v = list(F.otype.s('verse'))[0]
L.d('word', v)

i = 0
for n in NN():
    print('{} {}'.format(n, F.otype.v(n)))
    i += 1
    if i >= 20: break

msg('Counting')

i = 0
for n in NN(): i += 1
print(i)

msg('Done. {} nodes'.format(i))

msg('Counting per object type')

counts = collections.Counter()
for n in NN(): counts[F.otype.v(n)] += 1

msg('Done. {} distinct object types'.format(len(counts)))

for tp in counts: print('{} has {} nodes'.format(tp, counts[tp]))

for (tp, n) in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
    print('{:<20}: {:>7} nodes'.format(tp, n))

# for convenience, we use swahili bible book names
my_node = T.node_of('Mwanzo', 1, 1, lang='sw')

print(my_node)

T.langs

my_words = L.d('word', my_node)
print(my_words)

print(T.words(my_words))

print(T.words(my_words, fmt='hc'))

T.formats()

for f in T.formats(): print('{}'.format(T.words(my_words, fmt=f)))

for (f, (desc, method)) in T.formats().items(): print('{}={} {}'.format(f, desc, T.words(my_words, fmt=f)))

msg('collecting relationships')
mothers = {}
for source in NN():
    targets = set(C.mother.v(source))
    if targets: mothers[source] = targets
msg('Done. {}'.format(len(mothers)))

msg('Creating a set of mother nodes')
mother_nodes = set()
for mset in mothers.values(): mother_nodes |= mset
msg('Done. {} mother nodes'.format(len(mother_nodes)))

mother_count = collections.Counter()
for m in mother_nodes:
    mother_count[F.otype.v(m)] += 1

mother_count

def get_mothers(nodeset):
    mother_nodes = set()
    for n in nodeset:
        mother_nodes |= mothers.get(n, set())
    return mother_nodes  

len(get_mothers(set(NN())))

len(get_mothers(mother_nodes))

def get_ancestors(nodeset):
    my_mothers = get_mothers(nodeset)
    my_ancestors = my_mothers | get_ancestors(my_mothers)
    return my_ancestors

def longest_chain(nodeset):
    mset = get_mothers(nodeset)
    return 0 if not mset else 1 + max({longest_chain({n}) for n in mset})

msg('Computing longest chain')
lc = longest_chain(set(NN()))
msg('Done: {}'.format(lc))

msg('Computing mother levels')
mother_level = {}
rest = set(NN())
level = 0
while rest:
    level += 1
    rest = get_mothers(rest)
    if rest: mother_level[level] = rest
msg('Done. {} levels'.format(max(mother_level.keys())))  

for n in mother_level[46]:
    words = L.d('word', n)
    fw = words[0]
    b = L.u('book', fw)
    v = L.u('verse', fw)
    passage = '{} {}:{}'.format(
        T.book_name(b, lang='en'),
        F.chapter.v(v),
        F.verse.v(v),
    )
    ot = F.otype.v(n)
    print('{} => {} node {} = {} ({})'.format(
        passage,
        ot, n, 
        T.words(words, fmt='ec'),
        T.words(words, fmt='ha'),
    ))
    

def gfile(m, ext): return 'graph_{}.{}'.format(m, ext)

for m in mother_level[46]:
    print('Writing graph {}'.format(m))
    fh = open(gfile(m, 'ncol'), 'w')
    visited = {m}
    to_add = {(m, n) for n in set(Ci.mother.v(m))}
    new_nodes = {e[1] for e in to_add if e[1] not in visited}
    while to_add:
        # print('{} edges to add'.format(len(to_add)))
        for e in to_add: fh.write('{}\t{}\n'.format(*e))
        new_nodes = {e[1] for e in to_add if e[1] not in visited}
        visited |= new_nodes
        to_add = set()
        for x in new_nodes: to_add |= {(x, n) for n in set(Ci.mother.v(x))}
    fh.close()

from IPython.display import HTML, display_pretty, display_html
import networkx as nx
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def show_graph(m):
    g = nx.read_weighted_edgelist(gfile(m, 'ncol'))
    plt.figure(figsize=(18,18))

    nx.draw_networkx(g)
    # save the plot as pdf
    plt.savefig(gfile(m, 'pdf'))

show_graph(516777)

show_graph(532101)

focus_node = T.node_of('Mwanzo', 1, 1, lang='sw')
focus_node

word_nodes = L.d('word', focus_node)

word_nodes

field_names = '''
    g_word
    sp
    pdp
    ls
    nametype
    language
    lex
'''.strip().split()
row_template = ('{}\t' * (len(field_names) - 1))+'{}\n'

# this cell is now obsolete, use the next cell
fh = outfile('first_words.tsv')
fh.write(row_template.format(*field_names))
for wn in word_nodes:
    fh.write(row_template.format(
        F.g_word.v(wn),
        F.sp.v(wn),
        F.pdp.v(wn),
        F.item['ls'].v(wn),
        F.nametype.v(wn),
        F.language.v(wn),
    ))
fh.close()

fh = outfile('first_words.tsv')
fh.write(row_template.format(*field_names))
for wn in word_nodes:
    fh.write(row_template.format(*[F.item[feat].v(wn) for feat in field_names]))
fh.close()

focus_node = T.node_of('Mwanzo', 1, 3, lang='sw')
clauses = L.d('clause', focus_node)
for c in clauses:
    print(T.words(L.d('word', c)).replace('\n', ' '))    
    cas = L.d('clause_atom', c)
    for ca in cas:
        print('{} {}'.format(
            T.words(L.d('word', ca)).replace('\n', ' '), 
            F.code.v(ca),
        ))

msg('Printing clause atom relationships')
cars = []
for c in F.otype.s('clause'):
    cars.append(','.join(F.code.v(ca) for ca in L.d('clause_atom', c)))
carstxt = '\n'.join(cars)
f = outfile('car.txt')
f.write('\n'.join(cars))
f.close()
msg('Done {} clauses.'.format(len(cars)))
if '#' in carstxt: msg('Some clause atoms have no code')
else: msg('All clause atoms have a code')

nlines = 20
print('Showing first {} lines'.format(nlines))
sys.stdout.write('\n'.join(cars[0:nlines]))



