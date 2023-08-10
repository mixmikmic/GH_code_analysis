from Environment import *
from ExactRL import *
from Plot_utilities import *

sim = simulator()

RLMCobj = MC_ExactRL(config(), sim.init, sim.step)
RLMCobj.Monte_Carlo_Control(6000000)

plot(RLMCobj , "Optimal State Value Function for Monte-Carlo Control (GLIE)")

RLSobj = SarsaL_ExactRL(config(), sim.init, sim.step, lamda = 0.5)
RLSobj.Apply_SARSA(episodes = 100000)

plot(RLSobj, "Optimal State Value Function for SARSA(Lambda = 0.5)")

def MSE_Compute(Q2):
    la = np.linspace(0,1,11)
    Y = []
    for l1 in list(la):
        RLSobj = SarsaL_ExactRL(config(), sim.init, sim.step, lamda =l1)
        RLSobj.Apply_SARSA(1000)
        Y.append(MSE(RLSobj.Q, Q2))    
    plt.subplot(2, 1, 1)
    plt.plot(la, np.asarray(Y), 'r')
    plt.title('MSE(Q,Q*) as function of Lambda')
    plt.show()     

MSE_Compute(RLMCobj.Q)

