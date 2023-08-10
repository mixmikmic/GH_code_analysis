get_ipython().system('pip install mdp')
get_ipython().system('pip install Oger-1.2.tar.gz')

import Oger
import mdp
from matplotlib import pyplot as plt
import numpy as np

# Get the dataset
[x, y] = Oger.datasets.narma10(n_samples=10, sample_len=10000)

washable = True # washout of 200 steps gives better performance
input_scaling = 0.1 # best parameter for most models found through grid-search
bias_scaling = 0.0 # best parameter for most models found through grid-search
number_of_neurons = 100 # for the test run and the grid-search
number_of_neurons_list = [100, 200, 300] # for the evaluation
number_of_runs = 10 # number of random runs per model

# construct individual nodes
#reservoir = Oger.nodes.ReservoirNode(output_dim=100, input_scaling=0.5)
reservoir = Oger.nodes.SparseAndOrthogonalMatricesReservoir(input_dim=1, output_dim=number_of_neurons, input_scaling=input_scaling, bias_scaling=bias_scaling)
readout = Oger.nodes.RidgeRegressionNode(verbose=True, plot_errors=True, other_error_measure = Oger.utils.nmse)

# build network with MDP framework
flow = mdp.Flow([reservoir, readout], verbose=1)
Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)
if washable:
    Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode, 200)

data = [[], zip(x[0:-1], y[0:-1])]

# train the flow 
flow.train(data)

#apply the trained flow to the training data and test data
trainout = flow(x[0])
testout = flow(x[-1])

# print errors for all narma time serie
print "===== Full time serie error ====="
print "NRMSE: " + str(Oger.utils.nrmse(y[-1], testout))
print "NMSE: " + str(Oger.utils.nmse(y[-1], testout))
print "MSE: " + str(Oger.utils.mse(y[-1], testout))

# print errors for first 500 steps
print "===== first 500 steps error ====="
print "NRMSE: " + str(Oger.utils.nrmse(y[-1][:500], testout[:500]))
print "NMSE: " + str(Oger.utils.nmse(y[-1][:500], testout[:500]))
print "MSE: " + str(Oger.utils.mse(y[-1][:500], testout[:500]))

# print errors for first 500 steps after 200 steps
print "===== 200-700 steps error ====="
print "NRMSE: " + str(Oger.utils.nrmse(y[-1][200:700], testout[200:700]))
print "NMSE: " + str(Oger.utils.nmse(y[-1][200:700], testout[200:700]))
print "MSE: " + str(Oger.utils.mse(y[-1][200:700], testout[200:700]))

# plot the full predicition
plt.figure()
plt.plot(y[-1], 'r', ls='-')
plt.plot(testout, 'b', ls='--')
plt.legend(['Predicted signal', 'Target signal'])
plt.title('NARMA30 example run for ESN (full prediction)')
plt.xlabel('steps')
plt.ylabel('narma30 y value')
plt.show()

# plot the first 200 steps
plt.figure()
plt.plot(y[-1][:200], 'r', ls='-')
plt.plot(testout[:200], 'b', ls='--')
plt.legend(['Target signal', 'Predicted signal'], loc='best')
plt.title('NARMA10 example run for SORM (200 steps prediction)')
plt.xlabel('steps')
plt.ylabel('narma value')
plt.savefig('NARMA10_SORM_example.png')
plt.show()

string_names_dict_short = {
    0: 'ESN',
    1: 'DLR',
    2: 'DLRB',
    3: 'SCR',
    4: 'CRJ',
    5: 'FF-ESN',
    6: 'SORM',
    7: 'CyclicSORM'
}

string_names_dict_full = {
    0: 'original ESN',
    1: 'DelayLineReservoir',
    2: 'DelayLineReservoirWithFeedback',
    3: 'SimpleCycleReservoir',
    4: 'CycleReservoirWithJumps',
    5: 'FeedForwardESN',
    6: 'SparseAndOrthogonalMatrices',
    7: 'CyclicSparseAndOrthogonalMatrices'
}

nodes_dict = {
    0: Oger.nodes.ReservoirNode,
    1: Oger.nodes.DelayLineReservoirNode,
    2: Oger.nodes.DelayLineWithFeedbackReservoirNode,
    3: Oger.nodes.SimpleCycleReservoirNode,
    4: Oger.nodes.CycleReservoirWithJumpsNode,
    5: Oger.nodes.FeedForwardESNReservoir,
    6: Oger.nodes.SparseAndOrthogonalMatricesReservoir,
    7: Oger.nodes.CyclicSORMsReservoir
}

data = [[], zip(x[0:-1], y[0:-1])]

#Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)
if washable:
    Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode, 200)

# set the range of input and bias scaling
input_scaling_range = np.arange(0.1, 1.1, 0.1)
bias_scaling_range = np.arange(0.0, 1.0, 0.1)

count = 0
max_count = input_scaling_range.shape[0] * bias_scaling_range.shape[0] * len(nodes_dict.keys())
    

# grid-search for every model:
for key, reservoir_node in mdp.utils.progressinfo(nodes_dict.iteritems(), style='timer', 
                                                               length=len(nodes_dict.keys())):
    
    errors_mnse = np.zeros([input_scaling_range.shape[0], bias_scaling_range.shape[0]])
    errors_mse = np.zeros([input_scaling_range.shape[0], bias_scaling_range.shape[0]])
    
    for i, input_scaling in enumerate(input_scaling_range):
        for j, bias_scaling in enumerate(bias_scaling_range):
            # construct individual nodes
            reservoir = reservoir_node(input_dim=1, output_dim=number_of_neurons, input_scaling=input_scaling, 
                                                  bias_scaling=bias_scaling)
            readout = Oger.nodes.RidgeRegressionNode(verbose=False, plot_errors=False, 
                                                     other_error_measure = Oger.utils.nmse)

            # build network with MDP framework
            flow = mdp.Flow([reservoir, readout], verbose=0)

            # train the flow 
            flow.train(data)

            #apply the trained flow to the training data and test data
            testout = flow(x[-1])

            errors_mnse[i, j] = Oger.utils.nmse(y[-1], testout)
            errors_mse[i, j] = Oger.utils.mse(y[-1], testout)

            count += 1
            print "Progress: ", str(float(count) / max_count)
            
    # save the results for later processing
    np.savetxt('NMSE_' + string_names_dict_short[key] + '.txt', errors_mnse, delimiter=',')
    np.savetxt('MSE_' + string_names_dict_short[key] + '.txt', errors_mse, delimiter=',')

data = [[], zip(x[0:-1], y[0:-1])]

#Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)
if washable:
    Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode, 200)

# set the range of input and bias scaling
input_scaling_range = np.arange(0.1, 1.1, 0.1)
bias_scaling_range = np.arange(0.0, 1.0, 0.1)

count = 0
max_count = input_scaling_range.shape[0] * bias_scaling_range.shape[0] * len(nodes_dict.keys())
    

# grid-search for every model:
for key, reservoir_node in mdp.utils.progressinfo(nodes_dict.iteritems(), style='timer', 
                                                               length=len(nodes_dict.keys())):
    
    errors_mnse = np.zeros([input_scaling_range.shape[0], bias_scaling_range.shape[0]])
    errors_mse = np.zeros([input_scaling_range.shape[0], bias_scaling_range.shape[0]])
    
    # get the initial arrays to use
    reservoir_temp = reservoir_node(input_dim=1, output_dim=number_of_neurons,                                 input_scaling=1, bias_scaling=1)
    w_in = reservoir_temp.w_in
    w_bias = reservoir_temp.w_bias
    w = reservoir_temp.w
    
    for i, input_scaling in enumerate(input_scaling_range):
        for j, bias_scaling in enumerate(bias_scaling_range):
            # construct individual nodes
            reservoir = reservoir_node(input_dim=1, output_dim=number_of_neurons, 
                                      w=w, w_in=w_in*input_scaling, w_bias=w_bias*bias_scaling)
            
            readout = Oger.nodes.RidgeRegressionNode(verbose=False, plot_errors=False, 
                                                     other_error_measure = Oger.utils.nmse)

            # build network with MDP framework
            flow = mdp.Flow([reservoir, readout], verbose=0)

            # train the flow 
            flow.train(data)

            #apply the trained flow to the training data and test data
            testout = flow(x[-1])

            errors_mnse[i, j] = Oger.utils.nmse(y[-1], testout)
            errors_mse[i, j] = Oger.utils.mse(y[-1], testout)

            count += 1
            print "Progress: ", str(float(count) / max_count)
            
    # save the results for later processing
    np.savetxt('NMSE_' + string_names_dict_short[key] + '.txt', errors_mnse, delimiter=',')
    np.savetxt('MSE_' + string_names_dict_short[key] + '.txt', errors_mse, delimiter=',')

data = [[], zip(x[0:-1], y[0:-1])]

narma_nmse = np.zeros([len(number_of_neurons_list), len(nodes_dict), number_of_runs])
narma_mse = np.zeros([len(number_of_neurons_list), len(nodes_dict), number_of_runs])

count = 0
max_count = narma_nmse.shape[0] * narma_nmse.shape[1] * narma_nmse.shape[2]

mdp.utils.progressinfo(enumerate(number_of_neurons_list), style='timer', length=len(number_of_neurons_list))

# run the test for every reservoir
for i, reservoir_size in mdp.utils.progressinfo(enumerate(number_of_neurons_list), style='timer', length=len(number_of_neurons_list)):
    for key, reservoir_node in nodes_dict.iteritems():
        for j in range(number_of_runs):
            # construct individual nodes
            reservoir = reservoir_node(input_dim=1, output_dim=reservoir_size, input_scaling=input_scaling, bias_scaling=bias_scaling)
            readout = Oger.nodes.RidgeRegressionNode(verbose=False, plot_errors=False, other_error_measure = Oger.utils.nmse)

            # build network with MDP framework
            flow = mdp.Flow([reservoir, readout], verbose=0)

            # train the flow 
            flow.train(data)

            #apply the trained flow to the training data and test data
            testout = flow(x[-1])

            narma_nmse[i, key, j] = Oger.utils.nmse(y[-1], testout)
            narma_mse[i, key, j] = Oger.utils.mse(y[-1], testout)
            
            count += 1
            print "Progress: ", str(float(count) / max_count)
            
# save the errors for later processing
np.save("NARMA30_nmse", narma_nmse)
np.save("NARMA30_mse", narma_mse)

