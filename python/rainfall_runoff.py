import numpy as np
import matplotlib.pyplot as plt
import random

from keras import Model
from keras.layers import Input, Dense, Dropout, LSTM, SimpleRNN, GRU

get_ipython().magic('matplotlib inline')

MIN_EVENT_TIME = 5
MAX_EVENT_TIME = 20
MIN_EVENT_POWER = 0.05
MAX_EVENT_POWER = 0.5

BASINS = [
    (0.02, 20),
    (0.02, 30),
    (0.01, 35),
    (0.03, 41),
    (0.02, 45),
    (0.01, 54),
    (0.02, 65),
    (0.01, 75),
]

def gen_precip(num_samples, num_events):  
    
    result = np.zeros(num_samples)
    
    for event_start in random.sample(range(num_samples - MAX_EVENT_TIME), num_events):
        event = np.zeros((num_samples,))
        event_time = random.randint(MIN_EVENT_TIME, MAX_EVENT_TIME)
        event_power = random.random() * (MAX_EVENT_POWER - MIN_EVENT_POWER) + MIN_EVENT_POWER
        event[event_start: event_start + event_time] = (np.cos(np.linspace(-np.pi, np.pi, num=event_time)) * 0.5 + 0.5) * event_power
        result += event
    
    return result

def gen_discharge(basins, precip, runoff):            
    all_precip = np.zeros(precip.shape)
    
    for basin in basins:
        lagged = np.roll(precip * basin[0], basin[1])
        lagged[:basin[1]] = 0
        all_precip += lagged
        
    discharge = np.zeros(precip.shape)
        
    for i in range(1, len(discharge)):
        discharge[i] = discharge[i - 1]  * (1 - runoff) + all_precip[i]
        
    return discharge, precip

precip = gen_precip(500, 20)
discharge, _ = gen_discharge(BASINS, precip, 0.05)

plt.figure(figsize=(30,6))
plt.plot(precip)
plt.plot(discharge)
plt.show()

import random

NUM_SAMPLES = 2000
SEQ_LEN = 500

raw_train_Y, raw_train_X = zip(*[gen_discharge(BASINS, gen_precip(SEQ_LEN, 10), 0.01) for i in range(NUM_SAMPLES)])

#TODO : Test various models:

#fixed window feed-forward network
#simple RNN & LSTMs one step ahead
#test on longer sequences
#figure out how to iuse feed-forward network to longer predictions

