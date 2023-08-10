import numpy as np
import matplotlib.pyplot as plt
import Oger as oger
import scipy as sp
import mdp

get_ipython().magic('matplotlib inline')

[inputs, outputs] = oger.datasets.analog_speech(indir="Lyon_decimation_128")

plt.plot(range(len(inputs[-1][:])),inputs[-1][:])
plt.show()
print outputs[-1][0]

input_dim = inputs[0].shape[1]
reservoir = oger.nodes.LeakyReservoirNode(input_dim=input_dim, output_dim=100, input_scaling=1, leak_rate=0.1)
readout = oger.nodes.RidgeRegressionNode(0.001)
mnnode = oger.nodes.MeanAcrossTimeNode()
flow = mdp.Flow([reservoir, readout, mnnode])

train_frac = .8
n_samples = len(inputs)
n_train_samples = int(round(n_samples * train_frac))
n_test_samples = int(round(n_samples * (1 - train_frac))) 
flow.train([None,                 zip(inputs[0:n_train_samples - 1],                     outputs[0:n_train_samples - 1]),                 [None]])

ytest = []
for xtest in inputs[n_train_samples:]:
        ytest.append(flow(xtest))

ymean = sp.array([sp.argmax(sample) for sample in outputs[n_train_samples:]])
ytestmean = sp.array([sp.argmax(sample) for sample in ytest])

confusion_matrix = oger.utils.ConfusionMatrix.from_data(10, ytestmean, ymean)
print "Error rate: %.4f" % confusion_matrix.error_rate

oger.utils.plot_conf(confusion_matrix.balance())



