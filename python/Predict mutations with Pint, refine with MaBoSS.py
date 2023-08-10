import biolqm

lqm = biolqm.load("https://cellcollective.org/#5884/tumour-cell-invasion-and-migration")

import maboss

wt_sim = biolqm.to_maboss(lqm)

wt_sim.network.set_output(('Metastasis', 'Migration', 'Invasion', 'Apoptosis', 'CellCycleArrest'))

wt_sim.network.set_istate("ECM", [0, 1]) # ECM is active
wt_sim.network.set_istate("DNAdamage", [0.5, 0.5]) # DNAdamage can start either active or inactive

wt_sim.update_parameters(max_time=50)

wt_res = wt_sim.run()

wt_res.plot_piechart()

import pypint

m = biolqm.to_pint(lqm)

m.initial_state["ECM"] = 1
m.initial_state["DNAdamage"] = {0,1}

mutants = m.oneshot_mutations_for_cut("Apoptosis=1", exclude={"ECM", "DNAdamage"})
mutants

from itertools import combinations
from functools import reduce

mutant_combinations = [combinations(m.items(), 2) for m in mutants if len(m) >= 2]
candidates = reduce(set.union, mutant_combinations, set())
candidates

import matplotlib.pyplot as plt # for customizing the plots

for mutant in sorted(candidates):
    mut_sim = wt_sim.copy()
    for (node, value) in mutant:
        mut_sim.mutate(node, "ON" if value else "OFF")
    mut_res = mut_sim.run()
    mut_res.plot_piechart(embed_labels=False, autopct=4)
    mutant_name = "/".join(["%s:%s"%m for m in mutant])
    plt.title("%s mutant" % mutant_name)



