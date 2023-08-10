# Step 1: An empty network with multiple inputs

weights=[0.1,0.2,-0.1]

def nerual_network(inputs,weight):
    pred=w_sum(inputs,weight)
    return pred

def w_sum(inputs,weight):
    sum=0
    for i,v in enumerate(inputs):
        sum+=inputs[i]*weight[i]
    return sum

        

# Step 2: Predict and Compare

toes=[8.5,9.5,9.9,9.0]
wlrec=[0.65,0.8,0.8,0.9]
nfans=[1.2,1.3,0.5,1.0]

win_or_lose_binary=[1,1,0,1]
true=win_or_lose_binary[0]

inputs=[toes[0],wlrec[0],nfans[0]]

pred=nerual_network(inputs,weights)
error=(pred-true)**2
delta=pred-true

print 'Prediction: '+str(pred)
print 'Error: '+str(error)
print 'Delta: '+str(delta)

# Step 3:Learn: Calculating Each 'Weight Delta' and putting it on Each weight
import numpy as np
def ele_mul(delta,inputs):
    inputs=np.array(inputs)
    weight_delta=delta*inputs
    return weight_delta

weight_delta=ele_mul(delta,inputs)

print weight_delta

# Step 4: UPdate the weights

alpha=0.01
 
for i in range(len(weights)):
    weights[i]-=alpha*weight_delta[i]
    print weights[i]



