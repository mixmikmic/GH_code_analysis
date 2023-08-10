from ANM import Simulator
from case75 import case75
from numpy.random import RandomState

sim = Simulator(case75(), rng=RandomState(987654321))

print "The distribution network has %d buses (i.e. nodes) and %d branches (i.e. links)." % (sim.N_buses,sim.N_branches)
print "The network supplies %d loads and gathers the production of %d generators." % (sim.N_loads, sim.N_gens)
print "The control means consist in %d flexible loads (%.1f%%) and %d curtailable generators (%.1f%%)."       % (sim.N_flex, 100.0*sim.N_flex/sim.N_loads, sim.N_curt, 100.0*sim.N_curt/sim.N_gens)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.figsize"] = (10,6)

P_prod, P_cons = [], []
for _ in range(96): # One day long simulation
    P_prod.append(sum([sim.getPGen(gen) for gen in range(sim.N_gens)]))
    P_cons.append(sum([sim.getPLoad(load) for load in range(sim.N_loads)]))
    sim.transition() # Triggers a transition of the simulation (i.e. simulare next time step)

plt.plot(P_prod, label="production")
plt.plot(P_cons, label="consumption")
plt.ylabel("Active power [MW]")
plt.xlabel("Time")
plt.xticks([0, 24, 48, 72, 96], ["0h", "6h", "12h", "18h", "24h"])
plt.xlim([0,96])
_ = plt.legend()

from numpy import array

V = []
for _ in range(96): # One day long simulation
    V.append([sim.getV(bus) for bus in range(sim.N_buses)])
    sim.transition() # Triggers a transition of the simulation (i.e. simulates next time step)

plt.plot(array(V), "k")
plt.ylabel("Voltage magnitude [p.u.]")
plt.xlabel("Time")
plt.xticks([0, 24, 48, 72, 96], ["0h", "6h", "12h", "18h", "24h"])
_ = plt.xlim([0,96])

# Build the dataset
N_sim, L_sim = 5, 192 # 5 runs of 2 days are simulated to build the dataset
dataset = []
rng_dataset = RandomState(6576458)
for _ in range(N_sim):
    # A new instance is required for every simulation run.
    sim_dataset = Simulator(case75(),rng=rng_dataset)
    for _ in range(L_sim):
        # Compute averall active power balance.
        P_i = sum([sim_dataset.getPGen(gen) for gen in range(sim_dataset.N_gens)])             + sum([sim_dataset.getPLoad(load) for load in range(sim_dataset.N_loads)])
        # isSafe() returns True when operational constraints are met, False otherwise
        y_i = sim_dataset.isSafe()
        dataset.append([P_i,y_i])
        sim_dataset.transition()
print "Simulations led to %.1f%% of secure time steps."       % (100.0*sum([1 if data[1] else 0 for data in dataset])/len(dataset))

# Determine linear approximator using a 5% margin
P_bar = 0.95*min([d[0] for d in dataset if not d[1]])
# Show approximation
fig = plt.figure(figsize=(15,3))
plt.axvspan(xmin=P_bar, xmax=1.1*max([d[0] for d in dataset]), color="r", alpha=0.25)
plt.scatter([d[0] for d in dataset], [0.0]*len(dataset), c=[float(d[1]) for d in dataset], s=50, cmap="prism")
plt.yticks([])
plt.xlim([1.1*min([d[0] for d in dataset]),1.1*max([d[0] for d in dataset])])
plt.xlabel("Active power unbalance [MW]")
plt.title("Simple linear approximation")
fig.tight_layout()

from copy import copy, deepcopy

L_simu = 96
N_trajs = 10
rng_trajs = RandomState(3478765)
non_curt_gens = [gen for gen in range(sim.N_gens) if gen not in sim.curtIdInGens]

# record current state to compare simulation with/without the policy
sim_cloned = copy(sim)
sim_cloned = copy(sim)
sim_cloned.wind = copy(sim.wind)
sim_cloned.sun = copy(sim.sun)
sim_cloned.Ploads_fcts = deepcopy(sim.Ploads_fcts)
sim_cloned.rng = deepcopy(sim.rng) # 

P_prod = [] # Gather overall potential production during the simulation
P_curt = [] # Gather overall effictive production (incl. curt.) during the simulation
P_cons = [] # Gather overall consumption during the simulation
V = [] # Gather all voltage magnitudes during the simulation

for _ in range(L_simu):
    # Generate the N_trajs trajectories
    P_exo = []
    for _ in range(N_trajs):
        # Copy the Simulator's instance
        sampler = copy(sim)
        sampler.wind = copy(sim.wind)
        sampler.sun = copy(sim.sun)
        sampler.Ploads_fcts = deepcopy(sim.Ploads_fcts)
        sampler.rng = rng_trajs
        # Simulate a transition
        sampler.transition()
        P_exo.append(sum([sampler.getPGen(gen) for gen in non_curt_gens])+                     sum([sampler.getPLoad(load) for load in range(sampler.N_loads)]))

    # Determine the production limit of curtailable generators and apply the control actions
    P_max = min([(P_bar-P)/sim.N_curt for P in P_exo])
    for gen in sim.curtIdInGens:
        sim.setPmax(gen, P_max)
    
    # Simulate a transition
    sim.transition()
    P_prod.append(sum([sim.getPGen(gen) for gen in range(sim.N_gens)]))
    P_curt.append(sum([sim.getPCurtGen(gen) for gen in range(sim.N_gens)]))
    P_cons.append(sum([sim.getPLoad(load) for load in range(sim.N_loads)]))
    V.append([sim.getV(bus) for bus in range(1,sim.N_buses)]) # Range starts at 1 to ignore slack bus


# Simulate the same run without applying the policy
P_prod_free = [] # Gather overall potential production during the simulation
P_cons_free = [] # Gather overall consumption during the simulation
V_free = [] # Gather all voltage magnitudes during the simulation

for _ in range(L_simu):
    # Simulate a transition
    sim_cloned.transition()
    P_prod_free.append(sum([sim_cloned.getPGen(gen) for gen in range(sim_cloned.N_gens)]))
    P_cons_free.append(sum([sim_cloned.getPLoad(load) for load in range(sim_cloned.N_loads)]))
    V_free.append([sim_cloned.getV(bus) for bus in range(1,sim_cloned.N_buses)]) # Range starts at 1 to ignore slack bus

# Plot simulation results with policy
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,5))
fig.suptitle("Simulation with policy", fontsize=16)
ax1.fill_between(range(L_simu), P_prod, P_curt, color="r")
ax1.plot(P_prod, "k--", label=None)
ax1.plot(P_curt, "k-", lw="2", label="production")
ax1.plot(P_cons, "g", label="consumption")
ax1.legend(loc=3)
ax1.set_xticks([0, 24, 48, 72, 96])
ax1.set_xticklabels(["0h", "6h", "12h", "18h", "24h"])
ax1.set_xlim([0,L_simu])
ax2.plot(V,"k",label=None)
ax2.axhline(y=1.05, color="r", linestyle="--", label="$V_\max\,,\,V_\min$")
ax2.axhline(y=0.95, color="r", linestyle="--")
ax2.set_ylim([0.94,1.06])
ax2.set_xlim([0,L_simu])
ax2.legend(loc=3)
fig.tight_layout()
plt.subplots_adjust(top=0.9)

# Plot simulation results without policy
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,5))
fig.suptitle("Simulation without policy", fontsize=16)
ax1.plot(P_prod_free, "k-", lw="2", label="production")
ax1.plot(P_cons_free, "g", label="consumption")
ax1.legend(loc=3)
ax1.set_xlim([0,L_simu])
ax2.plot(V_free,"k",label=None)
ax2.axhline(y=1.05, color="r", linestyle="--", label="$V_\max\,,\,V_\min$")
ax2.axhline(y=0.95, color="r", linestyle="--")
ax2.set_ylim([0.94,1.06])
ax2.set_xlim([0,L_simu])
ax2.legend(loc=3)
fig.tight_layout()
plt.subplots_adjust(top=0.9)

