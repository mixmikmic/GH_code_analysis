# Inputs data 
toes=[8.5,9.5,9.9,9.0]
wlrec=[0.65,0.8,0.8,0.9]
nfans=[1.2,1.3,0.5,1.0]

inputs=[toes[0],wlrec[0],nfans[0]]
win_or_lose_binary=[1,1,0,1]
true=win_or_lose_binary[0]
weights=[0.1,0.2,-0.1]
alpha=0.3

# Create an empty network

def nerual_network(inputs,weights):
    prediction=mul_ele(inputs,weights)
    return prediction

def mul_ele(inputs,weights):
    output=0
    for i,v in enumerate(inputs):
        output+=inputs[i]*weights[i]
    return output        

# Run the model
import numpy as np
for iteration in range(3):
    pred=nerual_network(inputs,weights)
    error=(pred-true)**2
    delta=pred-true
    weight_delta=delta*np.array(inputs)
    weight_delta[0]=0
    print 'Iteration:'+str(iteration+1)
    print 'Prediction:'+str(pred)
    print 'Error:'+str(error)
    print 'Delta:'+str(delta)
    print 'Weights:'+str(weights)
    print 'Weight_Delta:'+str(weight_delta)
    
    weights-=alpha*weight_delta
    print

