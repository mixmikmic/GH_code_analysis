from Environment import *
from ExactRL import *
from Plot_utilities import *
from ApproximateRL import *

sim = simulator()
RLMCobj = MC_ExactRL(config(), sim.init, sim.step)
RLMCobj.Monte_Carlo_Control(6000000)

RLSobj = ApproxRL(config(), sim.init, sim.step, 0.5, coarse_featurizer)
RLSobj.Apply_SARSA(100000)

plot(RLSobj, "Optimal State Value Function for Approximate SARSA(Lambda = 0.5)")

def MSE_Compute(Q2):
    la = np.linspace(0,1,11)
    Y = []
    for l1 in list(la):
        RLSobj =ApproxRL(config(), sim.init, sim.step, l1, coarse_featurizer)
        RLSobj.Apply_SARSA(1000)
        Y.append(MSE(RLSobj.getQtable(), Q2))    
    plt.subplot(2, 1, 1)
    plt.plot(la, np.asarray(Y), 'r')
    plt.title('MSE(Q,Q*) as function of Lambda')
    plt.show()     

MSE_Compute(RLMCobj.Q)

