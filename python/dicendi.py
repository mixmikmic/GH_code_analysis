import sys
import collections

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
from etcbc.lib import Transcription, monad_set
from etcbc.trees import Tree

fabric = LafFabric()
tr = Transcription()

API = fabric.load('etcbc4', '--', 'dicendi', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        oid otype monads
        sp vt lex
    ''','''
        mother
    '''),
    "prepare": prepare,
}, verbose='NORMAL')
exec(fabric.localnames.format(var='fabric'))

tree_types = ('clause', 'word')
(root_type, leaf_type, clause_type) = (tree_types[0], tree_types[-1], 'clause')

tree = Tree(API, otypes=tree_types, 
    clause_type=clause_type,
    ccr_feature=None,
    pt_feature=None,
    pos_feature='sp',
    mother_feature =None,
)
results = tree.relations()
parent = results['eparent']
children = results['echildren']
msg("Ready for processing")

count_clauses = collections.defaultdict(lambda: 0)
lemor = set()

for n in children:
    nwords = len(children[n])
    if nwords == 2:
        prep = children[n][0]
        word = children[n][1]
        if F.sp.v(prep) == 'prep' and F.sp.v(word) == 'verb':
            lex = F.lex.v(word)
            count_clauses[lex] += 1
            if lex == '>MR': lemor.add(word)

print("{} clauses consisting of just a single verb.\n".format(len(count_clauses)))
for item in sorted(count_clauses.items(), key=lambda x: (-x[1], x[0])):
    print("{}: {} x".format(item[0], item[1]))

lemor_daughters = {}
for l in sorted(lemor):
    daughters = Ci.mother.v(l)
    lemor_daughters[l] = daughters
msg("{} lemors have daughters".format(len(lemor_daughters)))



