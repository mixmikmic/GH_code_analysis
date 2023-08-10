import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision import transforms
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
import pickle

from __future__ import print_function

import numpy as np

from random import random
from math import log, ceil
from time import time, ctime



class Hyperband:
    
    def __init__( self, get_params_function, try_params_function ):
        self.get_params = get_params_function
        self.try_params = try_params_function
        
        self.max_iter = 81      # maximum iterations per configuration
        self.eta = 3            # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log( x ) / log( self.eta )
        self.s_max = int( self.logeta( self.max_iter ))
        self.B = ( self.s_max + 1 ) * self.max_iter

        self.results = []    # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1
        

    # can be called multiple times
    def run( self, skip_last = 0, dry_run = False ):
        
        for s in reversed( range( self.s_max + 1 )):
            
            # initial number of configurations
            n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))    
            
            # initial number of iterations per config
            r = self.max_iter * self.eta ** ( -s )        

            # n random configurations
            T = [ self.get_params() for i in range( n )] 
            
            for i in range(( s + 1 ) - int( skip_last )):    # changed from s + 1
                
                # Run each of the n configs for <iterations> 
                # and keep best (n_configs / eta) configurations
                
                n_configs = n * self.eta ** ( -i )
                n_iterations = r * self.eta ** ( i )
                
                print("\n*** {} configurations x {:.1f} iterations each".format( n_configs, n_iterations ))
                
                val_losses = []
                early_stops = []
                
                for t in T:
                    
                    self.counter += 1
                    print("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format( self.counter, ctime(), self.best_loss, self.best_counter ))
                    
                    start_time = time()
                    
                    if dry_run:
                        result = { 'loss': random(), 'log_loss': random(), 'auc': random()}
                    else:
                        result = self.try_params( n_iterations, t )        # <---
                        
                    assert( type( result ) == dict )
                    assert( 'loss' in result )
                    
                    seconds = int( round( time() - start_time ))
                    print("\n{} seconds.".format( seconds))
                    
                    loss = result['loss']    
                    val_losses.append( loss )
                    
                    early_stop = result.get( 'early_stop', False )
                    early_stops.append( early_stop )
                    
                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter
                    
                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations
                    
                    self.results.append( result )
                
                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort( val_losses )
                T = [ T[i] for i in indices if not early_stops[i]]
                T = T[ 0:int( n_configs / self.eta )]
        
        return self.results

class Net(nn.Module):
    def __init__(self, dropout, conv1_n, conv2_n, fc_n, act_fn, init_fn):
        assert act_fn in ['relu', 'lrelu', 'elu']
        assert init_fn in ['xavier_uniform', 'xavier_normal', 'he_normal', 'he_uniform']
        
        super(Net, self).__init__()
        
        if act_fn == 'relu':
            self.act_fn = F.relu
            gain = init.calculate_gain('relu')
        elif act_fn == 'lrelu':
            self.act_fn = F.leaky_relu
            gain = init.calculate_gain('leaky_relu')
        else:
            self.act_fn = F.elu
            gain = init.calculate_gain('leaky_relu')
            
        if init_fn == 'xavier_uniform':
            init_layers = lambda tensor: init.xavier_uniform(tensor, gain=gain)
            init_last = lambda tensor: init.xavier_uniform(tensor, gain=1)
        elif init_fn == 'xavier_normal':
            init_layers = lambda tensor: init.xavier_normal(tensor, gain=gain)
            init_last = lambda tensor: init.xavier_normal(tensor, gain=1)
        elif init_fn == 'he_uniform':
            init_layers = init.kaiming_uniform
            init_last = init.kaiming_uniform
        else:
            init_layers = init.kaiming_normal
            init_last = init.kaiming_normal
        
        self.conv1 = nn.Conv2d(1, conv1_n, kernel_size=5, bias=False)
        init_layers(self.conv1.weight)
        self.conv2 = nn.Conv2d(conv1_n, conv2_n, kernel_size=5, bias=False)
        init_layers(self.conv2.weight)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.dropout = dropout
        self.n_flat = 4*4*conv2_n
        self.fc1 = nn.Linear(self.n_flat, fc_n)
        init_layers(self.fc1.weight)
        self.fc2 = nn.Linear(fc_n, 10)
        init_last(self.fc2.weight)
        
    def forward(self, x):
        x = self.act_fn(F.max_pool2d(self.conv1(x), 2))
        x = self.act_fn(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.n_flat)
        x = self.act_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

space = {
    'dropout': hp.quniform('dropout', 0, 0.5, 0.1),
    'batchsize': hp.choice('batchsize', (128, 256)),
    'fc_n': hp.choice('fc_n', (32, 64, 128)),
    'conv1_n': hp.choice('fc_n', (16, 32, 64)),
    'conv2_n': hp.choice('fc_n', (16, 32, 64)),
    'init_fn': hp.choice('init_fn', ('xavier_uniform', 'xavier_normal', 'he_normal', 'he_uniform')),
    'act_fn': hp.choice('act_fn', ('relu', 'lrelu', 'elu')),
    'lr': hp.loguniform('lr', -10, -2),
    'l2': hp.loguniform('l2', -10, -2),
}

def get_params():
    return sample(space)

def try_params(n_iterations, params):
    n_iterations = int(round(n_iterations))
    print("iterations: ", n_iterations)
    print("params: ", params)
    
    train_loader = torch.utils.data.DataLoader(MNIST('data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=params['batchsize'])
    test_loader = torch.utils.data.DataLoader(MNIST('data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=params['batchsize'])
    
    model = Net(params['dropout'], params['conv1_n'], params['conv2_n'], params['fc_n'], params['act_fn'], params['init_fn'])
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['l2'])
    
    model.train()
    for epoch in range(n_iterations):
        for data, target in train_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            loss = F.nll_loss(model(data), target)
            loss.backward()
            optimizer.step()
    
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return {'loss': test_loss, 'accuracy': accuracy}

hyperband = Hyperband(get_params, try_params)
results = hyperband.run()

with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)

sorted(results, key=lambda r: r['loss'])[:5]

