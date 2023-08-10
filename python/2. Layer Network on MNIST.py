import numpy as np
np.random.seed(1)

# Input data
alpha=0.005
iteration=300
hidden_size=40
pixels_per_image=784
num_labels=10

weights_0_1=0.2*np.random.random((pixels_per_image,hidden_size))-0.1
weights_1_2=0.2*np.random.random((hidden_size,num_labels))-0.1

print weights_0_1.shape
print weights_1_2.shape

def relu(x):
    return (x>=0)*x

def relu2deriv(output):
    return output>=0

for j in xrange(iteration):
    error=0
    correct_cnt=0
    
    for i in xrange(len(images)):
        layer_0=images[i:i+1]
        layer_1=relu(np.dot(layer_0,weights_0_1))
        layer_2=np.dot(layer_1,weights_1_2)
        
        error+=np.sum((layer_2-label[i:i+1])**2)
        correct_cnt+=int(np.argmax(layer_2)==np.argmax(labels[i:i+1]))
        
        layer_2_delta=layer_2-label[i:i+1]
        layer_1_delta=layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer-1)
        
        weights_1_2-=alpha*layer_1.T.dot(layer_2_delta)
        weights_0_1-=alpha*layer_0.T.dot(layer_1_delta)
        
    sys.stdout.write('\r'+
                    'Iteration:'+str(j)+
                    'Error:'+str(error/float(len(images)))[0:5]+
                    'Correct:'+str(correct_cnt/float(len(images))))

