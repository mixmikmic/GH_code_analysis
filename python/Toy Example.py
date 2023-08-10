import maboss

fast_sim = maboss.load("Four_cycle.bnd", "Four_cycle_FEscape.cfg")

#maboss.wg_set_istate(fast_sim)
maboss.set_nodes_istate(fast_sim, ["A", "B"], [0, 1])

fast_res = fast_sim.run()
fast_res.plot_trajectory()

slow_sim = fast_sim.copy()

slow_sim.param["$escape"] = 0.00001

slow_res = slow_sim.run()

slow_res.plot_trajectory()



