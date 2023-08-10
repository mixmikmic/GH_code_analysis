from bayespy.nodes import Dirichlet
K = 3
initial_state = Dirichlet(1e-3*np.ones(K))
transmat = Dirichlet(1e-3*np.ones((K,K)))

from bayespy.nodes import Gamma
D = 5
rate_prior = Gamma(1e-3, 1e-3, plates=(D,1,K))

from bayespy.nodes import Mixture, CategoricalMarkovChain, Poisson
Y = []
Z = []
TrainingData = [
    [np.random.poisson(lam=5, size=(D, np.random.poisson(lam=30)))]
]
for i in range(len(TrainingData[0])):
    [D, sequence_length] = TrainingData[0][i].shape
    Z.append(CategoricalMarkovChain(initial_state, transmat, states=sequence_length))
    Y.append(Mixture(Z[i], Poisson, rate_prior))
    Y[i].observe(TrainingData[0][i])

# Would like to do this:
#nodes = Y + [rate_prior] + Z + [transmat, initial_state]
#for z in Z:
#    z.initialize_from_random()
# But can't until issue number 30 has been fixed.

# Thus, use this:
nodes = Y + Z + [rate_prior, transmat, initial_state]
rate_prior.initialize_from_value(Gamma(10, 10, plates=(D,1,K)).random())

from bayespy.inference import VB
Q = VB(*nodes)

Q.update(repeat=100)

