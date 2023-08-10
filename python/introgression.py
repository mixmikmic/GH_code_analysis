get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import random
import collections
import msprime
import numpy as np
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt

from IPython.display import SVG

# Population IDs: Africa, Eurasia, Neanderthal
AFR, EUR, NEA = 0, 1, 2
    
def run_simulation(random_seed=None):   
    time_units = 1000 / 25  # Conversion factor for kya to generations
    ts = msprime.simulate(
        Ne=10**4,  # The same for all populations; highly unrealistic!
        recombination_rate=1e-8,
        length=100*10**6,  # 100 Mb
        samples=[
            msprime.Sample(time=0, population=AFR),
            msprime.Sample(time=0, population=EUR),
            # Neanderthal sample taken 30 kya
            msprime.Sample(time=30 * time_units, population=NEA),
        ],
        population_configurations = [
            msprime.PopulationConfiguration(), # Africa
            msprime.PopulationConfiguration(), # Eurasia
            msprime.PopulationConfiguration(), # Neanderthal
        ],
        demographic_events = [
            msprime.MassMigration(
                # 2% introgression 50 kya
                time=50 * time_units,
                source=EUR, dest=NEA, proportion=0.02),
            msprime.MassMigration(
                # Eurasian & Africa populations merge 70 kya
                time=70 * time_units, 
                source=EUR, dest=AFR, proportion=1),
            msprime.MassMigration(
                # Neanderthal and African populations merge 300 kya
                time=300 * time_units,
                source=NEA, destination=AFR, proportion=1),
        ],
        record_migrations=True,  # Needed for tracking segments.
        random_seed=random_seed,
    )
    return ts

ts = run_simulation(1)

def get_migrating_tracts(ts):
    migrating_tracts = []
    # Get all tracts that migrated into the neanderthal population
    for migration in ts.migrations():
        if migration.dest == NEA:
            migrating_tracts.append((migration.left, migration.right))
    return np.array(migrating_tracts) 

def get_coalescing_tracts(ts):
    coalescing_tracts = []
    tract_left = None
    for tree in ts.trees():    
        # 1 is the Eurasian sample and 2 is the Neanderthal
        mrca_pop = tree.population(tree.mrca(1, 2))
        left = tree.interval[0]
        if mrca_pop == NEA and tract_left is None:
            # Start a new tract
            tract_left = left      
        elif mrca_pop != NEA and tract_left is not None:
            # End the last tract
            coalescing_tracts.append((tract_left, left))
            tract_left = None
    if tract_left is not None:
        coalescing_tracts.append((tract_left, ts.sequence_length))
    return np.array(coalescing_tracts)

def get_eur_nea_tracts(ts):
    tracts = []
    tract_left = None
    for tree in ts.trees():    
        # 1 is the Eurasian sample and 2 is the Neanderthal
        mrca = tree.mrca(1, 2)
        left = tree.interval[0]
        if mrca != tree.root and tract_left is None:
            # Start a new tract
            tract_left = left      
        elif mrca != tree.root and tract_left is not None:
            # End the last tract
            tracts.append((tract_left, left))
            tract_left = None
    if tract_left is not None:
        tracts.append((tract_left, ts.sequence_length))
    return np.array(tracts)

                                    
migrating = get_migrating_tracts(ts)
within_nea = get_coalescing_tracts(ts)
eur_nea = get_eur_nea_tracts(ts)

nea_total = np.sum(eur_nea[:,1] - eur_nea[:,0])
migrating_total = np.sum(migrating[:,1] - migrating[:,0])
within_nea_total = np.sum(within_nea[:,1] - within_nea[:,0])
print([nea_total, migrating_total, within_nea_total])

kb = 1 / 1000
plt.hist([
    (eur_nea[:,1] - eur_nea[:,0]) * kb,
    (migrating[:,1] - migrating[:,0]) * kb,   
    (within_nea[:,1] - within_nea[:,0]) * kb,],    
    label=["Migrating", "EUR-NEA", "Within NEA"]
)
plt.yscale('log')
plt.legend()
plt.xlabel("Tract length (KB)");

def simulate_mutation_times(ts, random_seed=None):
    rng = random.Random(random_seed)
    mutation_time = np.zeros(ts.num_mutations)
    for tree in ts.trees():
        for mutation in tree.mutations():
            a = tree.time(mutation.node)
            b = tree.time(tree.parent(mutation.node))
            mutation_time[mutation.id] = rng.uniform(a, b)
    return mutation_time

pop_configs = [
    msprime.PopulationConfiguration(sample_size=3),
    msprime.PopulationConfiguration(sample_size=1),
    msprime.PopulationConfiguration(sample_size=1)]
M = [
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]]
ts  = msprime.simulate(
    population_configurations=pop_configs, migration_matrix=M,
    record_migrations=True, mutation_rate=0.5, random_seed=25)
mutation_time = simulate_mutation_times(ts, random_seed=25)

def get_mutation_population(ts, mutation_time):
    node_migrations = collections.defaultdict(list)
    for migration in ts.migrations():
        node_migrations[migration.node].append(migration)
    mutation_population = np.zeros(ts.num_mutations, dtype=int)
    for tree in ts.trees():
        for site in tree.sites():
            for mutation in site.mutations:                
                mutation_population[mutation.id] = tree.population(mutation.node)
                for mig in node_migrations[mutation.node]:
                    # Stepping through all migations will be inefficient for large 
                    # simulations. Should use an interval tree (e.g. 
                    # https://pypi.python.org/pypi/intervaltree) to find all 
                    # intervals intersecting with site.position.
                    if mig.left <= site.position < mig.right:
                        # Note that we assume that we see the migration records in 
                        # increasing order of time!
                        if mig.time < mutation_time[mutation.id]:
                            assert mutation_population[mutation.id] == mig.source
                            mutation_population[mutation.id] = mig.dest
    return mutation_population

mutation_population = get_mutation_population(ts, mutation_time)

tree = ts.first()
colour_map = {0:"red", 1:"blue", 2: "green"}
node_colours = {u: colour_map[tree.population(u)] for u in tree.nodes()}
mutation_colours = {mut.id: colour_map[mutation_population[mut.id]] for mut in tree.mutations()}
SVG(tree.draw(node_colours=node_colours, mutation_colours=mutation_colours))
    

