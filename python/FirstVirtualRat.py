import numpy as np
import matplotlib.pyplot as plt

from RNN import FirstRNN
from VirtualRatFunctions import *
from RNN_solver import RNNsolver

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

allRatsData = getData(5)

preprocessedData = preProcess(allRatsData,ratnames=["A099"])

RNNs = {}
solvers = []
choices = {}
probabilities = {}
logical_accuracies = {}
real_accuracies = {}
train_choices = {}
train_probabilities = {}
train_logical_accuracies = {}
train_real_accuracies = {}
test_choices = {}
test_probabilities = {}
test_logical_accuracies = {}
test_real_accuracies = {}
#learning_rates = 1e-5 * np.arange(8,13)
learning_rates = [1e-4]
for lr in learning_rates:
    print lr
    for ratname in preprocessedData.keys():
        print ratname
        ratData = preprocessedData[ratname]
        RNN = FirstRNN(hidden_dim = 4)
        RNNs[ratname] = RNN
        solver = RNNsolver(RNN, ratData['trainX'], ratData['trainY'],
                           update_rule='sgd',
                           optim_config={'learning_rate': lr,
                       }, num_epochs = 300,
                           lr_decay = 1,
                           verbose = True)
        #solvers[ratname] = solver
        solvers.append(solver)
        solver.train()
        choice, probs = RNN.predict(ratData['valX'])
        probabilities[ratname] = probs
        choices[ratname] = choice
        acc = np.mean(choice == ratData['valTrueY'])
        accReal = np.mean(choice == ratData['valY'])
        logical_accuracies[ratname] = acc
        real_accuracies[ratname] = accReal
        
        train_choice, train_probs = RNN.predict(ratData['trainX'])
        train_probabilities[ratname] = train_probs
        train_choices[ratname] = train_choice
        train_acc = np.mean(train_choice == ratData['trainTrueY'])
        train_accReal = np.mean(train_choice == ratData['trainY'])
        train_logical_accuracies[ratname] = train_acc
        train_real_accuracies[ratname] = train_accReal
        
        test_choice, test_probs = RNN.predict(ratData['testX'])
        test_probabilities[ratname] = test_probs
        test_choices[ratname] = test_choice
        test_acc = np.mean(test_choice == ratData['testTrueY'])
        test_accReal = np.mean(test_choice == ratData['testY'])
        test_logical_accuracies[ratname] = test_acc
        test_real_accuracies[ratname] = test_accReal
        print "Logical training accuracy: %s, Real sequence training accuracy: %s" % (train_acc, train_accReal)
        print "Logical validation accuracy: %s, Real sequence validation accuracy: %s" % (acc, accReal)
        print "Logical test accuracy: %s, Real sequence test accuracy: %s" % (test_acc, test_accReal)

sample = 50
for ratname in probabilities.keys():
    probs = probabilities[ratname]
    plt.plot(probs[0,:sample,0],'bo')
    plt.plot(probs[0,:sample,1],'ro')
    plt.plot(probs[0,:sample,2],'go')
    plt.xlabel('Trials')
    plt.ylabel('probs')
    plt.title('Probabilities of '+ ratname)
    plt.show()

sample = 50
for ratname in preprocessedData.keys():
    ratchoice = preprocessedData[ratname]['valY'][0,:]
    print ratchoice[:sample]
    

# Plot the training losses
for solver in solvers:
    plt.plot(solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history of '+ ratname)
    plt.show()

# Plot the training losses
#for ratname in solvers.keys():
#    plt.plot(solvers[ratname].loss_history)
#    plt.xlabel('Iteration')
#    plt.ylabel('Loss')
#    plt.title('Training loss history of '+ ratname)
#    plt.show()

trial_window = 3
postData = postProcess(choices, probabilities, preprocessedData,trial_window=trial_window)
p2a, a2p, p2a2, a2p2 = meanPerformance(postData) 
real_p2a, real_a2p, normalized_real_rat = realRatMeanPerformance(preprocessedData) 

for ratname in normalized_real_rat.keys():
    print "%s rat: %f, logical: %f, sequencial: %f" % (ratname,normalized_real_rat[ratname]['normalized_accuracy'],
                                                      logical_accuracies[ratname], real_accuracies[ratname])

for ratname in postData:
    plt.plot(range(500), postData[ratname]['pro_prob'][:500],color='b')
    plt.plot(range(500), postData[ratname]['anti_prob'][:500],color='r')
    plt.xlabel('Trials')
    plt.ylabel('%Correct')
    plt.title('Correction rate')
    plt.show()

# Plot for normalization
p2aplot, = plt.plot(range(-trial_window, 0), p2a[:trial_window], color='b')
a2pplot, = plt.plot(range(-trial_window, 0), a2p[:trial_window], color='r')
plt.plot(range(trial_window+1), p2a[trial_window:], color='r')
plt.plot(range(trial_window+1), a2p[trial_window:], color='b')
plt.plot([-1,0],p2a[trial_window - 1:trial_window + 1],'k--')
plt.plot([-1,0],a2p[trial_window - 1:trial_window + 1],'k--')
plt.scatter(range(-trial_window, 0), p2a[:trial_window], color='b')
plt.scatter(range(-trial_window, 0), a2p[:trial_window], color='r')
plt.scatter(range(trial_window+1), p2a[trial_window:], color='r')
plt.scatter(range(trial_window+1), a2p[trial_window:], color='b')

# Plot for excluding cpv
plt.plot(range(-trial_window, 0), p2a2[:trial_window], color='c')
plt.plot(range(-trial_window, 0), a2p2[:trial_window], color='m')
plt.plot(range(trial_window+1), p2a2[trial_window:], color='c')
plt.plot(range(trial_window+1), a2p2[trial_window:], color='m')
plt.plot([-1,0],p2a2[trial_window - 1:trial_window + 1],'y--')
plt.plot([-1,0],a2p2[trial_window - 1:trial_window + 1],'y--')
plt.scatter(range(-trial_window, 0), p2a2[:trial_window], color='c')
plt.scatter(range(-trial_window, 0), a2p2[:trial_window], color='m')
plt.scatter(range(trial_window+1), p2a2[trial_window:], color='m')
plt.scatter(range(trial_window+1), a2p2[trial_window:], color='c')

realp2aplot = plt.plot(range(-trial_window, 0), real_p2a[:trial_window], 'b--')
reala2pplot = plt.plot(range(-trial_window, 0), real_a2p[:trial_window], 'r--')
plt.plot(range(trial_window+1), real_p2a[trial_window:], 'r--')
plt.plot(range(trial_window+1), real_a2p[trial_window:], 'b--')
plt.plot([-1,0],real_p2a[trial_window - 1:trial_window + 1],'g--')
plt.plot([-1,0],real_a2p[trial_window - 1:trial_window + 1],'g--')
plt.scatter(range(-trial_window, 0), real_p2a[:trial_window], color='b')
plt.scatter(range(-trial_window, 0), real_a2p[:trial_window], color='r')
plt.scatter(range(trial_window+1), real_p2a[trial_window:], color='r')
plt.scatter(range(trial_window+1), real_a2p[trial_window:], color='b')

plt.legend([p2aplot, a2pplot],["pro","anti"])
plt.xlabel('Trial from switch')
plt.ylabel('Probability of correct')
plt.title('Performance around switches')
plt.show()

#uploadRNN(solvers['Z009'], 'Z009', "High acc but no switch cost feature", None, 5e-5, 4, 0.993295019157)

