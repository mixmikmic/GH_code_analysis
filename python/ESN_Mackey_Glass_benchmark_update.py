get_ipython().system('pip install mdp')
get_ipython().system('pip install Oger-1.2.tar.gz')

#%matplotlib nbagg
get_ipython().run_line_magic('matplotlib', 'inline')

import Oger
import mdp
import numpy as np
from matplotlib import pyplot as plt

# load the data
trainLen = 2000
testLen = 2000

initLen = 100

data = np.loadtxt('MackeyGlass_t17.txt')

# plot some of it
plt.figure(10).clear()
plt.plot(data[0:1000])

# generate the ESN reservoir
inSize = outSize = 1
resSize = 100
#a = 0.3


#np.random.seed(142)
reservoir = Oger.nodes.ReservoirNode(output_dim=resSize,     input_scaling=0.5, bias_scaling=0.0, spectral_radius=0.9, reset_states=False)

# Tell the reservoir to save its states for later plotting 
Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)
Oger.utils.make_inspectable(Oger.nodes.SparseReservoirNode)

# create the output   
reg = 1e-8   
#readout = Oger.nodes.RidgeRegressionNode( reg )
readout = Oger.nodes.RidgeRegressionNode(reg, verbose=True, plot_errors=True )

# enable washout
Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode, 200)

# connect them into ESN
flow = Oger.nodes.FreerunFlow([reservoir, readout], freerun_steps=testLen)

# train
flow.train([[], [[data[0:trainLen+1,None]]]])

# save states for plotting
X = reservoir.inspect()[0]



# run in a generative mode
Y = flow.execute(np.array(data[trainLen-initLen:trainLen+testLen+1,None]))
# discard the first elements (just a numbering convention)
Y = Y[initLen+1:] 

# compute MSE for the first errorLen time steps
errorLen = 500
mse = sum( np.square( data[trainLen+1:trainLen+errorLen+1] - Y[0:errorLen,0] ) ) / errorLen

print 'MSE = ' + str( mse )

# plot some signals
plt.figure(1).clear()
plt.plot( data[trainLen+1:trainLen+testLen+1], 'g' )
plt.plot( Y , 'b')
plt.legend(['Target signal', 'Free-running predicted signal'])
plt.title('Target and generated signals $y(n)$')

plt.figure(2).clear()
plt.plot( X[initLen:initLen+200,0:20] )
plt.title('Some reservoir activations $\mathbf{x}(n)$')

plt.figure(3).clear()
#plt.bar( range(1+resSize), readout.beta[:,0] )
plt.bar( range(1+resSize), np.hstack((readout.b, readout.w[:,0])) )
plt.title('Output weights $\mathbf{W}^{out}$')

plt.figure(4).clear()
plt.plot( data[trainLen+1:trainLen+testLen+1][:500], 'g' )
plt.plot( Y[:500] , 'b')
plt.legend(['Target signal', 'Free-running predicted signal'])
plt.title('Target and generated signals $y(n)$')

plt.figure(5).clear()
plt.plot( data[trainLen+1:trainLen+testLen+1][:200], 'g' )
plt.plot( Y[:200] , 'b')
plt.legend(['Target signal', 'Free-running predicted signal'])
plt.title('Target and generated signals $y(n)$')

plt.show()

# save some signals
plt.figure(1).clear()
plt.plot( data[trainLen+1:trainLen+testLen+1], 'r', ls='-' )
plt.plot( Y , 'b', ls='--')
plt.legend(['Target signal', 'Free-running predicted signal'], loc='best')
plt.title('Mackey-Glass ESN target and generated signals $y(n)$')
plt.xlabel('steps')
plt.ylabel('Mackey-Glass value')
plt.savefig('mg_prediction_2000.png')

plt.figure(2).clear()
plt.plot( X[initLen:initLen+200,0:20] )
plt.title('Mackey-Glass ESN some reservoir activations $\mathbf{x}(n)$')
plt.xlabel('steps')
plt.ylabel('unit internal value')
plt.savefig('mg_activations.png')

plt.figure(3).clear()
#plt.bar( range(1+resSize), readout.beta[:,0] )
plt.bar( range(1+resSize), np.hstack((readout.b, readout.w[:,0])) )
plt.title('Mackey-Glass ESN output weights $\mathbf{W}^{out}$')
plt.xlabel('weights')
plt.ylabel('weight value')
plt.savefig('mg_weights.png')

plt.figure(4).clear()
plt.plot( data[trainLen+1:trainLen+testLen+1][:500], 'r' , ls='-' )
plt.plot( Y[:500] , 'b', ls='--' )
plt.legend(['Target signal', 'Free-running predicted signal'], loc='best')
plt.title('Mackey-Glass ESN target and generated signals $y(n)$')
plt.xlabel('steps')
plt.ylabel('Mackey-Glass value')
plt.savefig('mg_prediction_500.png')

plt.figure(5).clear()
plt.plot( data[trainLen+1:trainLen+testLen+1][:200], 'g' )
plt.plot( Y[:200] , 'b')
plt.legend(['Target signal', 'Free-running predicted signal'])
plt.title('Target and generated signals $y(n)$')

plt.show()

test_signals = data[trainLen+1:trainLen+testLen+1][:, np.newaxis]
print "====== 500 steps error ======"
print "NMSE: " + str(Oger.utils.nmse(test_signals[:500], Y[:500]))
print "MSE: " + str(Oger.utils.mse(test_signals[:500], Y[:500]))

print "====== 200 steps error ======"
print "NMSE: " + str(Oger.utils.nmse(test_signals[:200], Y[:200]))
print "MSE: " + str(Oger.utils.mse(test_signals[:200], Y[:200]))

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

washable = True # washout of 200 steps gives better performance
input_scaling = 0.5 # best parameter for most models found through grid-search
bias_scaling = 0.0 # best parameter for most models found through grid-search
number_of_neurons_list = [100, 200, 300, 400, 500] # for the evaluation
number_of_runs = 10 # number of random runs per model

trainLen = 2000
testLen = 2000
initLen = 100
data = np.loadtxt('MackeyGlass_t17.txt')
inSize = outSize = 1
reg = 1e-8  

jump_size = 9

# define the test signals
test_signals = data[trainLen+1:trainLen+testLen+1][:, np.newaxis]

# enable washout
if washable:
    Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode, 200)

# error lists for 500 steps
mg_nmse_500 = np.zeros([len(number_of_neurons_list), len(nodes_dict), number_of_runs])
mg_mse_500 = np.zeros([len(number_of_neurons_list), len(nodes_dict), number_of_runs])
# error lists for 200 steps
mg_nmse_200 = np.zeros([len(number_of_neurons_list), len(nodes_dict), number_of_runs])
mg_mse_200 = np.zeros([len(number_of_neurons_list), len(nodes_dict), number_of_runs])

# run the test for every reservoir
for i, reservoir_size in mdp.utils.progressinfo(enumerate(number_of_neurons_list), style='timer', length=len(number_of_neurons_list)):
    for key, reservoir_node in nodes_dict.iteritems():
        for j in range(number_of_runs):
            # construct individual nodes
            if key == 4:
                reservoir = reservoir_node(input_dim=inSize, output_dim=reservoir_size, input_scaling=input_scaling,                                            bias_scaling=bias_scaling, jump_size=jump_size, reset_states=False)
            else:
                reservoir = reservoir_node(input_dim=inSize, output_dim=reservoir_size, input_scaling=input_scaling,                                            bias_scaling=bias_scaling, reset_states=False)

            readout = Oger.nodes.RidgeRegressionNode(reg, verbose=True, plot_errors=True )
            
            # connect them into ESN
            flow = Oger.nodes.FreerunFlow([reservoir, readout], freerun_steps=testLen)

            # train
            flow.train([[], [[data[0:trainLen+1,None]]]])
            
            # run in a generative mode
            Y = flow.execute(np.array(data[trainLen-initLen:trainLen+testLen+1,None]))
            # discard the first elements (just a numbering convention)
            Y = Y[initLen+1:]

            mg_nmse_500[i, key, j] = Oger.utils.nmse(test_signals[:500], Y[:500])
            mg_mse_500[i, key, j] = Oger.utils.mse(test_signals[:500], Y[:500])
            
            mg_nmse_200[i, key, j] = Oger.utils.nmse(test_signals[:200], Y[:200])
            mg_mse_200[i, key, j] = Oger.utils.mse(test_signals[:200], Y[:200])
            
# save the errors for later processing
np.save("mg_nmse_500", mg_nmse_500)
np.save("mg_mse_500", mg_mse_500)
np.save("mg_nmse_200", mg_nmse_200)
np.save("mg_mse_200", mg_mse_200)

# define the test signals
test_signals = data[trainLen+1:trainLen+testLen+1][:, np.newaxis]

# enable washout
if washable:
    Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode, 200)

# error lists for 500 steps
mg_nmse_500 = np.zeros([len(number_of_neurons_list), len(nodes_dict), number_of_runs])
mg_mse_500 = np.zeros([len(number_of_neurons_list), len(nodes_dict), number_of_runs])
# error lists for 200 steps
mg_nmse_200 = np.zeros([len(number_of_neurons_list), len(nodes_dict), number_of_runs])
mg_mse_200 = np.zeros([len(number_of_neurons_list), len(nodes_dict), number_of_runs])

count = 0
max_count = mg_nmse_500.shape[0] * mg_nmse_500.shape[1] * mg_nmse_500.shape[2]

# run the test for every reservoir
for i, reservoir_size in mdp.utils.progressinfo(enumerate(number_of_neurons_list), style='timer', length=len(number_of_neurons_list)):
    for key, reservoir_node in nodes_dict.iteritems():
        for j in range(number_of_runs):
            # construct individual nodes
            if key == 4:
                reservoir = reservoir_node(input_dim=inSize, output_dim=reservoir_size, input_scaling=input_scaling,                                            bias_scaling=bias_scaling, jump_size=jump_size, reset_states=False)
            else:
                reservoir = reservoir_node(input_dim=inSize, output_dim=reservoir_size, input_scaling=input_scaling,                                            bias_scaling=bias_scaling, reset_states=False)
            
            if key == 4 or key == 2:
                readout = Oger.nodes.RidgeRegressionNode(reg, verbose=True, plot_errors=True )

                # connect them into ESN
                flow = Oger.nodes.FreerunFlow([reservoir, readout], freerun_steps=testLen)

                # train
                flow.train([[], [[data[0:trainLen+1,None]]]])

                # run in a generative mode
                Y = flow.execute(np.array(data[trainLen-initLen:trainLen+testLen+1,None]))
                # discard the first elements (just a numbering convention)
                Y = Y[initLen+1:]

                mg_nmse_500[i, key, j] = Oger.utils.nmse(test_signals[:500], Y[:500])
                mg_mse_500[i, key, j] = Oger.utils.mse(test_signals[:500], Y[:500])

                mg_nmse_200[i, key, j] = Oger.utils.nmse(test_signals[:200], Y[:200])
                mg_mse_200[i, key, j] = Oger.utils.mse(test_signals[:200], Y[:200])
                
            else:
                max_error = 10.0
                
                while max_error > 1.0:
                    reservoir = reservoir_node(input_dim=inSize, output_dim=reservoir_size, input_scaling=input_scaling,                                            bias_scaling=bias_scaling, reset_states=False)
                    readout = Oger.nodes.RidgeRegressionNode(reg, verbose=True, plot_errors=True )

                    # connect them into ESN
                    flow = Oger.nodes.FreerunFlow([reservoir, readout], freerun_steps=testLen)

                    # train
                    flow.train([[], [[data[0:trainLen+1,None]]]])

                    # run in a generative mode
                    Y = flow.execute(np.array(data[trainLen-initLen:trainLen+testLen+1,None]))
                    # discard the first elements (just a numbering convention)
                    Y = Y[initLen+1:]

                    mg_nmse_500[i, key, j] = Oger.utils.nmse(test_signals[:500], Y[:500])
                    mg_mse_500[i, key, j] = Oger.utils.mse(test_signals[:500], Y[:500])

                    mg_nmse_200[i, key, j] = Oger.utils.nmse(test_signals[:200], Y[:200])
                    mg_mse_200[i, key, j] = Oger.utils.mse(test_signals[:200], Y[:200])
                    
                    max_error = Oger.utils.nmse(test_signals[:500], Y[:500])
                    #print max_error
                    
            count += 1
            print "Progress: ", str(float(count) / max_count)
            
# save the errors for later processing
np.save("mg_nmse_500_limit", mg_nmse_500)
np.save("mg_mse_500_limit", mg_mse_500)
np.save("mg_nmse_200_limit", mg_nmse_200)
np.save("mg_mse_200_limit", mg_mse_200)

