import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

def plotdata(filename='./sim_perf.txt', ttl=''):
    sd = pd.read_csv(filename, sep=';')
    sd['loss'] = sd['loss'].astype(float)
    sd['epsilon'] = sd['epsilon'].astype(float)
    plt.figure(figsize=(16,8))
    plt.title(ttl)
    plt.plot(sd['epoch'], sd['rougeF1'], c='blue', label='F1')
    plt.plot(sd['epoch'], sd['rougePrecision'], c='green', label='Precision')
    plt.plot(sd['epoch'], sd['rougeRecall'], c='purple', label='Recall')
    plt.ylim([0,1])
    plt.ylabel("Rouge, Loss, & Epsilon value")
    plt.xlabel("Training Epoch")
    plt.grid()
    
#     plt.plot(sd['epoch'], sd['loss'], c='red')
#     plt.plot(sd['epoch'], sd['epsilon'], c='gray')
    plt.legend()
    plt.show()
    return sd

get_ipython().system(' time th DQN_Simulation.lua --nepochs 1000 --gamma 0.8             --learning_rate 1e-4 --cuts 5 --n_rand 100             --edim 50 --mem_size 50 --metric f1 ')

_ = plotdata('./sim_perf.txt','Rouge, Loss, Epsilon across training Epochs - BOW')

get_ipython().system(' time th DQN_Simulation.lua --nepochs 1000 --gamma 0.8             --learning_rate 1e-4 --cuts 5 --n_rand 100             --edim 50 --mem_size 50 --metric f1 --nnmod lstm')

_ = plotdata('./sim_perf.txt', 'Rouge, Loss, Epsilon across training Epochs - LSTM')

get_ipython().system(' time th DQN_Simulation.lua --nepochs 1000 --gamma 0.8             --learning_rate 1e-4 --cuts 5 --n_rand 100             --edim 50 --mem_size 6 --metric f1 --nnmod bow')

_ = plotdata('./sim_perf.txt', 'Rouge, Loss, Epsilon across training Epochs - BOW')

get_ipython().system(' time th DQN_Simulation.lua --nepochs 1000 --gamma 0             --learning_rate 1e-4 --cuts 5 --n_rand 100             --edim 50 --mem_size 6 --metric f1 --nnmod bow')

_ = plotdata('./sim_perf.txt', 'Rouge, Loss, Epsilon across training Epochs - BOW')



