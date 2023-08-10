import DSGRN

br=DSGRN.Network("br.txt")

print(br)

import graphviz

print(br.graphviz())

graph=graphviz.Source(br.graphviz())

graph

br_pg=DSGRN.ParameterGraph(br)

print(br_pg.size())

br_64 = br_pg.parameter(64)

print(br_64.inequalities())

br_dg_64=DSGRN.DomainGraph(br_64)

graphviz.Source(br_dg_64.graphviz())

br_morsedecomposition=DSGRN.MorseDecomposition(br_dg_64.digraph())

graphviz.Source(br_morsedecomposition.graphviz())

br_morsegraph=DSGRN.MorseGraph()

br_morsegraph.assign(br_dg_64,br_morsedecomposition)

print(br_morsegraph)

graphviz.Source(br_morsegraph.graphviz())

br2=DSGRN.Network("br2.txt")

print(br2)

graph = graphviz.Source(br2.graphviz())

graph

parametergraph = DSGRN.ParameterGraph(br2)

br2_141=parametergraph.parameter(141)

domaingraph_br2_141 = DSGRN.DomainGraph(br2_141)

graphviz.Source(domaingraph_br2_141.graphviz())

morsedecomposition = DSGRN.MorseDecomposition(domaingraph_br2_141.digraph())

graphviz.Source(morsedecomposition.graphviz())

morsegraph = DSGRN.MorseGraph()

morsegraph.assign(domaingraph_br2_141, morsedecomposition)

graphviz.Source(morsegraph.graphviz())

