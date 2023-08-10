import numpy as np
np.random.seed(1)
import sys

def relu(x):
    return (x>0)*x

def relu2deriv(output):
    return (output>0)

# Input data

alpha,iterations=(0.1,100)
pixels_per_image,num_labels,hidden_size=(784,10,100)

weights_0_1=0.2*np.random.random((pixels_per_image,hidden_size))-0.1
weights_1_2=0.2*np.random.random((hidden_size,num_labels))-0.1

for j in xrange(iteratins):
    error=0
    correct_cnt=0
    for i in xrange(len(images)/batch_size):
        batch_start,batch_end=((i*batch_size),((i+1)*batch_size))
        layer_0=images[batch_start:batch_end]
        layer_1=relu(np.dot(layer_0,weights_0_1))
        dropout_mask=np.random.randint(2,size=layer_1.shape)
        layer_1*=dropout_mask
        layer_2=np.dot(layer_1,weights_1_2)
        
        error+=np.sum((labels[batch_start:batch_end]-layer_2)**2)
        
        for k in xrange(batch_size):
            correct_cnt+=int(np.argmax(layer_2[k:k+1])==np.argmax(labels[batch_start+k:batch_start+k+1]))
        
        delta_layer_2=(layer_2-labels[batch_start:batch_end])/batch_size
        delta_layer_1=delta_layer_2.dot(weights_1_2.T)*relu2deriv(layer_1)
        
        delta_layer_1*=dropout_mask
        
        weights_1_2-=alpha*layer_1.T*dot(delta_layer_2)
        weights_0_1-=alpha*layer_0.T*dot(delta_layer_1)
        


