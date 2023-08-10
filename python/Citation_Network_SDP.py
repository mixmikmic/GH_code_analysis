import metaknowledge as mk
import ncpol2sdpa as nc
import numpy as np
import re

def Fp(chain, T):
    return sum((1+chain[t])*(1-chain[t+1]*chain[t]) for t in range(T-1))//4


def Fn(chain, T):
    return sum((1-chain[t])*(1-chain[t+1]*chain[t]) for t in range(T-1))//4


def Sp(chain, T):
    return sum((1+chain[t])*(1+chain[t+1]*chain[t]) for t in range(T-1))//4


def Sn(chain, T):
    return sum((1-chain[t])*(1+chain[t+1]*chain[t]) for t in range(T-1))//4


def generate_chains(T, At=None, chain=None):
    if At is None:
        Atl = [1, -1]
    else:
        Atl = [At]
        self_chain = [c for c in chain]
    sum_ = []
    for Ai in Atl:
        if chain is None:
            self_chain = [Ai]
        else:
            self_chain.append(Ai)
        for Aj in [1, -1]:
            if T > 2:
                sum_ += generate_chains(T-1, Aj, self_chain)
            else:
                sum_.append(self_chain + [Aj])
    return sum_


class Probability(object):

    def __init__(self, T):
        chains = generate_chains(T)
        self.combinations = []
        for chain1 in chains:
            for chain2 in chains:
                self.combinations.append([chain1, chain2, 0])

    def __getitem__(self, chains):
        for combination in self.combinations:
            if combination[0] == chains[0] and combination[1] == chains[1]:
                return combination[2]
        raise Exception("Not found")

    def __setitem__(self, chains, value):
        for combination in self.combinations:
            if combination[0] == chains[0] and combination[1] == chains[1]:
                combination[2] = value
                return
        raise Exception("Not found")

    def normalize(self, Z):
        for combination in self.combinations:
            combination[2] /= Z

def normalize_name(name):
    w = re.sub("([a-z]{2,}) ", "\g<1>X", re.sub("[\.,]", " ", name.lower()))
    return w.replace(" ", "").replace("X", " ")


def get_authors_in_period(RC, year1, year2):
    RC_period = RC.yearSplit(year1, year2)
    authors_period = {}
    for R in RC_period:
        if R.citations is not None:
            reference_IDs = [reference.ID() for reference in R.citations]
            for author in R.authors:
                author = normalize_name(author)
                if author in authors_period:
                    authors_period[author] += reference_IDs
                else:
                    authors_period[author] = reference_IDs
    return authors_period


def create_directed_coauthor_graph(RC):
    coauthors_undirected = RC.coAuthNetwork()
    coauthors = coauthors_undirected.to_directed()
    weights = coauthors_undirected.degree()
    for edge in coauthors_undirected.edges_iter():
        if weights[edge[0]] > weights[edge[1]]:
            coauthors.remove_edge(edge[1], edge[0])
        else:
            coauthors.remove_edge(edge[0], edge[1])
    return coauthors


def get_all_references(author, epochs):
    references = set()
    for epoch in epochs:
        if author in epoch:
            for reference in epoch[author]:
                references.add(reference)
    return references

RC = mk.RecordCollection("./savedrecs.txt")
# RC = mk.RecordCollection("./InfSci20JournBase.hci")

years = []
for R in RC:
    if isinstance(R.year, int):
        years.append(R.year)
histogram = np.histogram(np.array(years), bins=3)
coauthors0 = create_directed_coauthor_graph(RC.yearSplit(histogram[1][0],
                                                         histogram[1][1]))

authors_period0 = get_authors_in_period(RC, histogram[1][0], histogram[1][1])
authors_period1 = get_authors_in_period(RC, histogram[1][1], histogram[1][2])
authors_period2 = get_authors_in_period(RC, histogram[1][2], histogram[1][3])
epochs = [authors_period0, authors_period1, authors_period2]

T = len(epochs)
p = Probability(T)
M = 0
for pair in coauthors0.edges():
    coauthor1 = normalize_name(pair[0])
    coauthor2 = normalize_name(pair[1])
    references = get_all_references(coauthor1, epochs)
    references.union(get_all_references(coauthor1, epochs))
    for reference in references:
        chain1, chain2 = [], []
        for epoch in epochs:
            if coauthor1 in epoch:
                if reference in epoch[coauthor1]:
                    chain1.append(+1)
                else:
                    chain1.append(-1)
            else:
                chain1.append(-1)
            if coauthor2 in epoch:
                if reference in epoch[coauthor2]:
                    chain2.append(+1)
                else:
                    chain2.append(-1)
            else:
                chain2.append(-1)
        p[(chain1, chain2)] += 1
        M += 1
p.normalize(M)

A1 = 1
alpha0 = nc.generate_variables(name='alpha_0')[0]
alphap = nc.generate_variables(name='alpha_p')[0]
alpham = nc.generate_variables(name='alpha_m')[0]
beta0 = nc.generate_variables(name='beta_0')[0]
betap = nc.generate_variables(name='beta_p')[0]
betam = nc.generate_variables(name='beta_m')[0]
chains = generate_chains(T)
moments = []
for chain1 in chains:
    for chain2 in chains:
        if p[(chain1, chain2)] > 0:
            P_A = alphap**Fp(chain1, T) * alpham**Fn(chain1, T) * (1-alpham)**Sn(chain1, T) * (1-alphap)**Sp(chain1, T) *                     alpha0**((1+A1)//2) * (1-alpha0)**((1-A1)//2)
            P_B = betap**Fp(chain2, T) * betam**Fn(chain2, T) * (1-betam)**Sn(chain2, T) * (1-betap)**Sp(chain2, T) *                     beta0**((1+A1)//2) * (1-beta0)**((1-A1)//2)
            moments.append(P_A*P_B - p[(chain1, chain2)])

inequalities = [alpha0, alphap, alpham, beta0, betap, betam,
                1-alpha0, 1-alphap, 1-alpham, 1-beta0, 1-betap, 1-betam]

sdp = nc.SdpRelaxation([alpha0, alphap, alpham, beta0, betap, betam])
sdp.get_relaxation(3, inequalities=inequalities, momentequalities=moments)
sdp.solve(solver="mosek")

print(sdp.primal, sdp.dual, sdp.status)

