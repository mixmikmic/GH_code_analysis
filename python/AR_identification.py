import numpy as np
import matplotlib.pylab as plt

import signalz
import padasip as pa

get_ipython().magic('matplotlib inline')
plt.style.use('ggplot') # nicer plots
np.random.seed(52102) # always use the same random seed to make results comparable

# AR model parameters
h = [-0.41, 1.27, -1.85, 1.79]

# pass white noise to AR model (3000 samples)
N = 3000
d = signalz.autoregressive_model(N, h, noise="white")

# plot data
plt.figure(figsize=(12.5,3))
plt.plot(d)
plt.title("Used data - all samples")
plt.xlabel("Discrete time index [/]")
plt.ylabel("System output [/]")
plt.tight_layout(); plt.show()

plt.figure(figsize=(12.5,3))
plt.plot(d[500:600])
plt.title("Used data - detail (samples 500-600)")
plt.xlabel("Discrete time index [/]")
plt.ylabel("System output [/]")
plt.tight_layout(); plt.show()

# input matrix for filters (4 taps)
n = 4
x = pa.preprocess.input_from_history(d, n)[:-1]
d = d[n:]

# list of all filters (with other values like names, and positions in figures)
filters = [
    {"name": "LMS", "mu_s": 0.001, "mu_e": 0.05, "filter": pa.filters.FilterLMS(n), "plot_position": 221 },
    {"name": "NLMS", "mu_s": 0.01, "mu_e": 2., "filter": pa.filters.FilterNLMS(n), "plot_position": 222 },
    {"name": "GNGD", "mu_s": 0.01, "mu_e": 4., "filter": pa.filters.FilterGNGD(n), "plot_position": 223 },
    {"name": "RLS", "mu_s": 0.001, "mu_e": 1., "filter": pa.filters.FilterRLS(n), "plot_position": 224 },    
]

plt.figure(figsize=(12.5,8))

# iterate over all filters
for i in range(len(filters)):
    # make sure that initial weights are zeros
    filters[i]["filter"].init_weights("zeros")
    # get mean error for learning rates in given range
    errors_e, mu_range = filters[i]["filter"].explore_learning(d, x, mu_start=filters[i]["mu_s"],
                                        mu_end=filters[i]["mu_e"],
                                        steps=100, ntrain=0.5, epochs=1, criteria="MSE")
    # get deviation of weights from parameters of AR model
    errors_w, mu_range = filters[i]["filter"].explore_learning(d, x, mu_start=filters[i]["mu_s"],
                                        mu_end=filters[i]["mu_e"],
                                        steps=100, ntrain=0.5, epochs=1, criteria="MSE", target_w=h)
    
    # save the best learning rate for later use
    filters[i]["filter"].mu = mu_range[np.argmin(errors_w)]
    # plot it
    plt.subplot(filters[i]["plot_position"])
    plt.plot(mu_range, errors_e, label="MSE of system prediction")
    plt.plot(mu_range, errors_w, label="MSE of parameters")
    plt.title(filters[i]["name"]); plt.xlabel("Learning rate (mu) [/]")
    plt.ylabel("MSE [/]"); plt.ylim(0,3); plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12.5, 12))

# iterate over all filters
for i in range(len(filters)):
    # make sure that initial weights are zeros
    filters[i]["filter"].init_weights("zeros")
    # run the identification
    filters[i]["filter"].run(d, x)
    # get the history of weights
    w_history = filters[i]["filter"].w_history        
    # plot it
    plt.subplot(filters[i]["plot_position"])
    plt.plot(w_history)  
    plt.title(filters[i]["name"]); plt.xlabel("Number of iteration [/]")
    plt.ylabel("Weights [/]"); plt.ylim(-2,2)
    # draw lines representing system parameters
    for coef in h:
        plt.axhline(y=coef, color="black", linestyle=':')
    
plt.tight_layout(); plt.show()

plt.figure(figsize=(12.5, 8))

# iterate over all filters
for i in range(len(filters)):
    # make sure that initial weights are zeros
    filters[i]["filter"].init_weights("zeros")
    # run the identification
    filters[i]["filter"].run(d, x)
    # get the history of weights
    w_history = filters[i]["filter"].w_history     
    deviation = pa.misc.logSE(np.mean((w_history - h)**2, axis=1))
    # plot it
    plt.plot(deviation, label=filters[i]["name"])  
    plt.title("Speed of identification"); plt.xlabel("Number of iteration [/]")
    plt.ylabel("Deviation of weights [dB]")
    
plt.tight_layout(); plt.legend(); plt.show()

