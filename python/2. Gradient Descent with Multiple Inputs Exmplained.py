# An empty nerual network
def nerual_network(inputs,weight):
    prediction=inputs*weight
    return prediction

# Inputs information

weight=0.1
number_of_toes=[8.5]
win_or_lose_binary=[1]

inputs=number_of_toes[0]
true=win_or_lose_binary[0]

pred=nerual_network(inputs,weight)

error=(pred-true)**2
delta=pred-true

print error
print delta

# Multi input: making a prediction and calculating error and delta
toes=[8.5,9.5,9.9,9.0]
wlrec=[0.65,0.8,0.8,0.9]
nfans=[1.2,1.3,0.5,1.0]

win_or_lose_binary=[1,1,0,1]
true=win_or_lose_binary[0]
inputs=[toes[0],wlrec[0],nfans[0]]
weights=[0.1,0.2,-0.1]

def ele_mut(inputs,weights):
    sum=0
    for i,v in enumerate(inputs):
        sum+=v*weights[i]
    return sum

# Update nerual network

def nerual_network(inputs,wegiths):
    prediction=ele_mut(inputs,weights)
    return prediction

pred=nerual_network(inputs,weights)

error=(pred-true)**2
delta=pred-true

print pred
print error
print delta

import numpy as np

inputs=np.array(inputs)
weight_delta=np.dot(delta,inputs)
print weight_delta

alpha=0.01

weight-=weight_delta*alpha
print weight



