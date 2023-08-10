import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

D = np.random.randn(1000,500)
hidden_layer_sizes = [500]*10
nonlinearities = ['tanh']*len(hidden_layer_sizes)

act = {'relu':lambda x:np.maximum(0,x), 'tanh':lambda x:np.tanh(x)}

Hs_collapsed = {}
Hs_blown = {}
for i in range(len(hidden_layer_sizes) * 2):
    if i < 10:
        ###: H updates on collapsed weight values
        if i == 0:
            X = D
        else:
            X = Hs_collapsed[i-1]
        n_in = X.shape[1] 
        n_out = hidden_layer_sizes[i] 
        W = np.random.randn(n_in, n_out) * 0.01
        H = np.dot(X, W)
        H = act[nonlinearities[i]](H)
        Hs_collapsed[i] = H
    else:
        ###: H updates on blown weight values
        ix = np.abs(10-(20-i-1))-1
        if ix == 0:
            X = D
        else:
            X = Hs_blown[ix-1]
        n_in = X.shape[1] 
        n_out = hidden_layer_sizes[ix] 
        W = np.random.randn(n_in, n_out) * 1.0
        H = np.dot(X, W)
        H = act[nonlinearities[ix]](H)
        Hs_blown[ix] = H 
        
print("Saturation -> 0 results: \n")
print("Input layer mean {mu} and standard deviation {sd}".format(mu=np.mean(D), sd=np.mean(D)))
layer_means_c = [np.mean(H) for i, H in Hs_collapsed.items()]
layer_stds_c = [np.std(H) for i, H in Hs_collapsed.items()]

for i, H in Hs_collapsed.items():
    print("Hidden layer {layer} had mean {mu} and standard deviation {sd}".format(layer=i+1, mu=layer_means_c[i], sd=layer_stds_c[i]))

print("\nSaturation -> 1 results: \n")
print("Input layer mean {mu} and standard deviation {sd}".format(mu=np.mean(D), sd=np.mean(D)))
layer_means_b = [np.mean(H) for i, H in Hs_blown.items()]
layer_stds_b = [np.std(H) for i, H in Hs_blown.items()]

for i, H in Hs_blown.items():
    print("Hidden layer {layer} had mean {mu} and standard deviation {sd}".format(layer=i+1, mu=layer_means_b[i], sd=layer_stds_b[i]))

plt.figure(figsize=(20,10))
keys = np.array([key for key in Hs_collapsed.keys()])

plt.subplot(121)
l_means = np.array(layer_means_c).T
plt.plot(keys, l_means, 'ob-')
plt.title('layer mean')

plt.subplot(122)
l_stds = np.array(layer_stds_c).T
plt.plot(keys, l_stds, 'ob-')
plt.title('layer std')

plt.show()

plt.figure(figsize=(20,10))
keys = np.array([key for key in Hs_blown.keys()])

plt.subplot(121)
l_means = np.array(layer_means_b).T
plt.plot(keys, l_means, 'ob-')
plt.title('layer mean')

plt.subplot(122)
l_stds = np.array(layer_stds_b).T
plt.plot(keys, l_stds, 'ob-')
plt.title('layer std')

plt.show()

Hs_Xavier = {}
for i in range(len(hidden_layer_sizes)):
    if i == 0:
        X = D
    else:
        X = Hs_Xavier[i-1]
    n_in = X.shape[1] 
    n_out = hidden_layer_sizes[i] 
    W = np.random.randn(n_in, n_out) / np.sqrt(n_in)
    H = np.dot(X, W)
    H = act[nonlinearities[i]](H)
    Hs_Xavier[i] = H
    
print("Xavier Initialization results: \n")
print("Input layer mean {mu} and standard deviation {sd}".format(mu=np.mean(D), sd=np.mean(D)))
layer_means_X = [np.mean(H) for i, H in Hs_Xavier.items()]
layer_stds_X = [np.std(H) for i, H in Hs_Xavier.items()]

for i, H in Hs_Xavier.items():
    print("Hidden layer {layer} had mean {mu} and standard deviation {sd}".format(layer=i+1, mu=layer_means_X[i], sd=layer_stds_X[i]))

plt.figure(figsize=(20,10))
keys = np.array([key for key in Hs_Xavier.keys()])
plt.subplot(121)
l_means = np.array(layer_means_X).T
plt.plot(keys, l_means, 'ob-')
plt.title('layer mean')

plt.subplot(122)
l_stds = np.array(layer_stds_X).T
plt.plot(keys, l_stds, 'ob-')
plt.title('layer std')

plt.show()

nonlinearities = ['relu']*len(hidden_layer_sizes)
Hs_Xavier = {}
for i in range(len(hidden_layer_sizes)):
    if i == 0:
        X = D
    else:
        X = Hs_Xavier[i-1]
    n_in = X.shape[1] 
    n_out = hidden_layer_sizes[i] 
    W = np.random.randn(n_in, n_out) / np.sqrt(n_in)
    H = np.dot(X, W)
    H = act[nonlinearities[i]](H)
    Hs_Xavier[i] = H

layer_means_X = [np.mean(H) for i, H in Hs_Xavier.items()]
layer_stds_X = [np.std(H) for i, H in Hs_Xavier.items()]

plt.figure(figsize=(20,10))
keys = np.array([key for key in Hs_Xavier.keys()])
plt.subplot(121)
l_means = np.array(layer_means_X).T
plt.plot(keys, l_means, 'ob-')
plt.title('layer mean')
plt.subplot(122)
l_stds = np.array(layer_stds_X).T
plt.plot(keys, l_stds, 'ob-')
plt.title('layer std')
plt.show()

nonlinearities = ['relu']*len(hidden_layer_sizes)
Hs_Xavier = {}
for i in range(len(hidden_layer_sizes)):
    if i == 0:
        X = D
    else:
        X = Hs_Xavier[i-1]
    n_in = X.shape[1] 
    n_out = hidden_layer_sizes[i] 
    W = np.random.randn(n_in, n_out) / np.sqrt(n_in/2)
    H = np.dot(X, W)
    H = act[nonlinearities[i]](H)
    Hs_Xavier[i] = H

layer_means_X = [np.mean(H) for i, H in Hs_Xavier.items()]
layer_stds_X = [np.std(H) for i, H in Hs_Xavier.items()]

plt.figure(figsize=(20,10))
keys = np.array([key for key in Hs_Xavier.keys()])
plt.subplot(121)
l_means = np.array(layer_means_X).T
plt.plot(keys, l_means, 'ob-')
plt.title('layer mean')
plt.subplot(122)
l_stds = np.array(layer_stds_X).T
plt.plot(keys, l_stds, 'ob-')
plt.title('layer std')
plt.show()



