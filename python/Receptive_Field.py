def compute_N(out,f,s):
    return s*(out-1)+f if s>0.5 else ((out+(f-2))/2)+1#

def compute_RF(layers):
    out=1
    for f,s in reversed(layers):
        out=compute_N(out,f,s)
    return out

layers=[(9,1),(3,1),(3,1),(3,1),(9,1),(3,1),(3,1),(7,1),(3,1)]
compute_RF(layers)

import numpy as np
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F

def compute_RF_numerical(net,img_np):
    '''
    @param net: Pytorch network
    @param img_np: numpy array to use as input to the networks, it must be full of ones and with the correct
    shape.
    '''
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)
    net.apply(weights_init)
    img_ = Variable(torch.from_numpy(img_np).float(),requires_grad=True)
    out_cnn=net(img_)
    out_shape=out_cnn.size()
    ndims=len(out_cnn.size())
    grad=torch.zeros(out_cnn.size())
    l_tmp=[]
    for i in xrange(ndims):
        if i==0 or i ==1:#batch or channel
            l_tmp.append(0)
        else:
            l_tmp.append(out_shape[i]/2)
            
    grad[tuple(l_tmp)]=1
    out_cnn.backward(gradient=grad)
    grad_np=img_.grad[0,0].data.numpy()
    idx_nonzeros=np.where(grad_np!=0)
    RF=[np.max(idx)-np.min(idx)+1 for idx in idx_nonzeros]
    
    return RF

class CNN(nn.Module):
    def __init__(self,layer_list):
        #layers is a list of tuples [(f,s)]
        super(CNN, self).__init__()
        f_ini,s_ini=layer_list[0]
        f_end,s_end=layer_list[-1]
        self.layers=[]
        self.layers.append(nn.Conv2d(1, 16, kernel_size=f_ini, padding=1,stride=s_ini,dilation=1))
        for f,s in layer_list[1:-1]:
            self.layers.append(nn.Conv2d(16, 16, kernel_size=f, padding=1,stride=s,dilation=1))
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(16, 1, kernel_size=f_end, padding=1,stride=s_end,dilation=1))
        self.all_layers=nn.Sequential(*self.layers)
        
        
    def forward(self, x):
        out = self.all_layers(x)
        return out

###########################################################
print 'analytical RF:',compute_RF(layers)

mycnn=CNN(layers)


img_np=np.ones((1,1,100,100))
print 'numerical RF',compute_RF_numerical(mycnn,img_np)



