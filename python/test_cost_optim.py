get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import numpy as np
import theano
from theano import tensor as T
import pickle as pk
import re
from copy import deepcopy
import sys

# image processing
from skimage.io import imread
from skimage.transform import resize

import lasagne
from lasagne.layers import Pool2DLayer, Conv2DLayer, dropout,     DenseLayer, InputLayer, get_output, get_all_params
    
import bnr_ml.objectdetect.yolo as yolo
from bnr_ml.utils.helpers import meshgrid2D, softmax, bitwise_not

import pdb

reload(yolo)

S = (6,6)
B = 2
C = 4

N = 100
input = 1000*np.random.randn(N,3,200,200).astype(np.float32)
output = np.random.rand(N,B*5 + C, S[0], S[1]).astype(np.float32)
truth = np.random.rand(N,4 + C).astype(np.float32)

input = theano.shared(input)
truth = theano.shared(truth)

net = {}
net['input'] = InputLayer((None,3,200,200))
net['conv1'] = Conv2DLayer(net['input'], 16, (3,3))
net['conv2'] = Conv2DLayer(net['conv1'], 16, (3,3))
net['pool1'] = Pool2DLayer(net['conv2'], (2,2))
net['conv3'] = Conv2DLayer(net['pool1'], 32, (3,3))
net['conv4'] = Conv2DLayer(net['conv3'], 32, (3,3))
net['pool2'] = Pool2DLayer(net['conv4'], (2,2))
net['conv5'] = Conv2DLayer(net['pool2'], 64, (3,3))
net['conv6'] = Conv2DLayer(net['conv5'], 64, (3,3))
net['pool3'] = Pool2DLayer(net['conv6'], (2,2))
net['conv7'] = Conv2DLayer(net['pool3'], 64, (3,3))
net['conv8'] = Conv2DLayer(net['conv7'], 64, (3,3))
net['pool4'] = Pool2DLayer(net['conv8'], (2,2))
net['dense1'] = DenseLayer(dropout(net['pool4'], p=.8), 1000)
net['dense2'] = DenseLayer(dropout(net['dense1'], p=.8), 1000)
net['output'] = DenseLayer(dropout(net['dense2'], p=.5), 5, nonlinearity=lasagne.nonlinearities.softmax)

with open('pretrained_weights.pkl', 'rb') as f:
    weights = pk.load(f)
    lasagne.layers.set_all_param_values(net['output'], weights)

net['dense3'] = DenseLayer(dropout(net['pool4'], p=.8), 2048)
net['output'] = DenseLayer(dropout(net['dense3'], p=.8), (S[0] * S[1]) * (5 * B + C), nonlinearity=None)

yl = yolo.YoloObjectDetector(net, (None, 3, 200, 200), C, S, B) 

def _get_cost_optim_multi(self, output, truth, S, B, C,lmbda_coord=5., lmbda_noobj=0.5, iou_thresh=0.05):
    '''
    Calculates cost for multiple objects in a scene without for loops or scan (so reduces the amount of variable
    created in the theano computation graph).  A cell is associated with a certain object if the iou of that cell
    and the object is higher than any other ground truth object. and the rest of the objectness scores are pushed
    towards zero.
    '''
    
    # calculate height/width of individual cell
    block_height, block_width = 1. / S[0], 1./ S[1]

    # get the offset of each cell
    offset_x, offset_y = meshgrid2D(T.arange(0,1,block_width), T.arange(0,1,block_height))

    # get indices for x,y,w,h,object-ness for easy access
    x_idx, y_idx = T.arange(0,5*B,5), T.arange(1,5*B, 5)
    w_idx, h_idx = T.arange(2,5*B,5), T.arange(3,5*B,5)
    conf_idx = T.arange(4,5*B,5)

    # Get position predictions with offsets.
    pred_x = (output[:,x_idx] + offset_x.dimshuffle('x','x',0,1)).dimshuffle(0,'x',1,2,3)
    pred_y = (output[:,y_idx] + offset_y.dimshuffle('x','x',0,1)).dimshuffle(0,'x',1,2,3)
    pred_w, pred_h = output[:,w_idx].dimshuffle(0,'x',1,2,3), output[:,h_idx].dimshuffle(0,'x',1,2,3)
    pred_conf = output[:,conf_idx].dimshuffle(0,'x',1,2,3)
    pred_class = output[:,-C:].dimshuffle(0,'x',1,2,3)
    
    pred_w, pred_h = T.maximum(pred_w, 0.), T.maximum(pred_h, 0.)

    x_idx, y_idx = T.arange(0,truth.shape[1],4+C), T.arange(1,truth.shape[1],4+C)
    w_idx, h_idx = T.arange(2,truth.shape[1],4+C), T.arange(3,truth.shape[1],4+C)
    class_idx,_ = theano.scan(
        lambda x: T.arange(x,x+C,1),
        sequences = T.arange(4,truth.shape[1],4+C)
    )

    truth_x, truth_y = truth[:,x_idx], truth[:,y_idx]
    truth_w, truth_h = truth[:,w_idx], truth[:,h_idx]
    truth_class = truth[:, class_idx]
    

    # Get intersection region bounding box coordinates
    xi = T.maximum(pred_x, truth_x.dimshuffle(0,1,'x','x','x'))
    xf = T.minimum(pred_x + pred_w, (truth_x + truth_w).dimshuffle(0,1,'x','x','x'))
    yi = T.maximum(pred_y, truth_y.dimshuffle(0,1,'x','x','x'))
    yf = T.minimum(pred_y + pred_h, (truth_y + truth_h).dimshuffle(0,1,'x','x','x'))
    w, h = T.maximum(xf - xi, 0.), T.maximum(yf - yi, 0.)

    # Calculate iou score for predicted boxes and truth
    isec = w * h
    union = (pred_w * pred_h) + (truth_w * truth_h).dimshuffle(0,1,'x','x','x') - isec
    iou = T.maximum(isec/union, 0.)

    # Get index matrix representing max along the 1st dimension for the iou score (reps 'responsible' box).
    maxval_idx, _ = meshgrid2D(T.arange(B), T.arange(truth.shape[0]))
    maxval_idx = maxval_idx.dimshuffle(0,'x',1,'x','x')
    maxval_idx = T.repeat(T.repeat(maxval_idx,S[0],3),S[1],4)

    box_is_resp = T.eq(maxval_idx, iou.argmax(axis=2).dimshuffle(0,1,'x',2,3))

    # Get matrix for the width/height of each cell
    width, height = T.ones(S) / S[1], T.ones(S) / S[0]
    width, height = width.dimshuffle('x','x',0,1), height.dimshuffle('x','x',0,1)
    offset_x, offset_y = offset_x.dimshuffle('x','x',0,1), offset_y.dimshuffle('x','x',0,1)

    # Get bounding box for intersection between CELL and ground truth box.
    xi = T.maximum(offset_x, truth_x.dimshuffle(0,1,'x','x'))
    xf = T.minimum(offset_x + width, (truth_x + truth_w).dimshuffle(0,1,'x','x'))
    yi = T.maximum(offset_y, truth_y.dimshuffle(0,1,'x','x'))
    yf = T.minimum(offset_y + height, (truth_y + truth_h).dimshuffle(0,1,'x','x'))
    w, h = T.maximum(xf - xi, 0.), T.maximum(yf - yi, 0.)

    # Calculate iou score for the cell.
    isec = w * h
    union = (width * height) + (truth_w* truth_h).dimshuffle(0,1,'x','x') - isec
    iou_cell = T.maximum(isec/union, 0.).dimshuffle(0,1,'x',2,3)
    
    maxval_idx, _ = meshgrid2D(T.arange(iou_cell.shape[1]), T.arange(iou_cell.shape[0]))
    maxval_idx = maxval_idx.dimshuffle(0,1,'x','x','x')
    maxval_idx = T.repeat(T.repeat(T.repeat(maxval_idx, B, 2), S[0], 3), S[1], 4)
    
    obj_for_cell = T.eq(maxval_idx, iou_cell.argmax(axis=1).dimshuffle(0,'x',1,2,3))
        
    # Get logical matrix representing minimum iou score for cell to be considered overlapping ground truth.
    cell_intersects = (iou_cell > iou_thresh)
        
    obj_in_cell_and_resp = T.bitwise_and(T.bitwise_and(cell_intersects, box_is_resp), obj_for_cell)
    conf_is_zero = T.bitwise_and(
        bitwise_not(T.bitwise_and(cell_intersects, box_is_resp)),
        obj_for_cell
    )
    conf_is_zero = conf_is_zero.sum(axis=1, keepdims=True)
    
    # repeat "cell overlaps" logical matrix for the number of classes.
    pred_class = T.repeat(pred_class, truth.shape[1] // (4 + C), axis=1)

    # repeat the ground truth for class probabilities for each cell.
    truth_class_rep = T.repeat(T.repeat(truth_class.dimshuffle(0,1,2,'x','x'), S[0], axis=3), S[1], axis=4)

    # calculate cost
    cost = T.sum((pred_conf - iou)[obj_in_cell_and_resp.nonzero()]**2) +         lmbda_noobj * T.sum((pred_conf[conf_is_zero.nonzero()])**2) +         lmbda_coord * T.sum((pred_x - truth_x.dimshuffle(0,1,'x','x','x'))[obj_in_cell_and_resp.nonzero()]**2) +         lmbda_coord * T.sum((pred_y - truth_y.dimshuffle(0,1,'x','x','x'))[obj_in_cell_and_resp.nonzero()]**2) +         lmbda_coord * T.sum((pred_w.sqrt() - truth_w.dimshuffle(0,1,'x','x','x').sqrt())[obj_in_cell_and_resp.nonzero()]**2) +         lmbda_coord * T.sum((pred_h.sqrt() - truth_h.dimshuffle(0,1,'x','x','x').sqrt())[obj_in_cell_and_resp.nonzero()]**2) +         T.sum(((pred_class - truth_class_rep)[cell_intersects.nonzero()])**2)
    
    return cost / T.maximum(1., truth.shape[0])

S = (2,2)
B = 2
C = 2

truth = np.asarray([[0.,0.,0.5,0.5, 1.,0.,   .6,.6,.3,.3,0.,1.,     -10.,-10.,0.1,0.1, 1.,0.,                       -10.,-10.,0.1,0.1, 1.,0.,    -10.,-10.,0.1,0.1, 1.,0.,    -10.,-10.,0.1,0.1, 1.,0.]])

pred = np.asarray([[0.,0.,0.5,0.5,1., -10.,-10.,.1,.1,.1  ,1.,0.]]) 
pred = pred.reshape((1,B*5 + C,1,1))
pred = np.repeat(np.repeat(pred,2,axis=2),2,axis=3)

output = T.tensor4('output')
target = T.matrix('target')

N, M = 10, 2
X,y = np.random.rand(N, 5*B + C, S[0], S[1]), np.random.rand(N, M * (4 + C))

X,y = theano.shared(pred), theano.shared(truth)

cost = _get_cost_optim_multi(None, X,y,S,B,C)

cost.eval()

cost_fn(pred, truth)







_get_cost_optim(yl, yl.output_test, truth, S, B, C)

def manual_cost(output, truth, C, S, B, lmbda_coord=5., lmbda_noobj=.5, thresh = .05):
    output, truth = np.copy(output), np.copy(truth)
    def calc_iou(b1, b2):
        xi = np.maximum(b1[0], b2[0])
        xf = np.minimum(b1[0]+b1[2], b2[0]+b2[2])
        yi = np.maximum(b1[1], b2[1])
        yf = np.minimum(b1[1]+b1[3], b2[1]+b2[3])
        w, h = np.maximum(0., xf - xi), np.maximum(0., yf - yi)
        isec = np.maximum(w * h, 0.)
        union = (b1[2]*b1[3]) + (b2[2]*b2[3]) - isec
        iou = isec / union
        return iou
    
    cost = 0.
    xshift, yshift = 1./S[1], 1./S[0]
    for i in range(S[0]):
        for j in range(S[1]):
            iou_score_per_box = np.zeros((B,))
            for k in range(B):
                reg_truth = truth[:4]
                reg_box = output[k*5:k*5 + 4, i, j]
                reg_box[0] += j * xshift
                reg_box[1] += i * yshift
                iou_score_per_box[k] = calc_iou(reg_truth, reg_box)
            
            idx_resp = np.argmax(iou_score_per_box)
            
            for k in range(B):
                reg_fact = lmbda_noobj
                if k == idx_resp:
                    reg_fact = 1.
                cost += reg_fact * (output[k*B + 4, i, j] - iou_score_per_box[k])**2
        
            reg_cell = np.asarray([j * xshift, i * yshift, xshift, yshift])
            iou_cell = calc_iou(reg_cell, truth[:4])
            
            if iou_cell > thresh:
                cost += ((output[-C:, i, j] - truth[-C:])**2).sum()
                
            cost += lmbda_coord * ((output[idx_resp*5:idx_resp*5+2,i,j] - truth[:2])**2).sum()
            cost += lmbda_coord * ((np.sqrt(output[idx_resp*5+2:idx_resp*5+4,i,j]) - np.sqrt(truth[2:4]))**2).sum()
    
    return cost
            

N = 100
input = 10000*np.random.randn(N,3,200,200).astype(np.float32)
output = np.random.rand(N,B*5 + C, S[0], S[1]).astype(np.float32)
truth = np.random.rand(N,4 + C).astype(np.float32)

target = T.matrix('target')
art_output = T.tensor4('art_output')

out_fn = theano.function([yl.input], yl.output_test, allow_input_downcast=True)

fn = theano.function([yl.input, target], yl._get_cost_optim(yl.output_test, target, S, B, C), allow_input_downcast=True)

fn(input, truth)

# fn2 = theano.function([art_output, target], get_cost_optim(yl, art_output, target, S, B, C), allow_input_downcast=True)

art_output = iput

fn(iput, truth)

manual_cost(out_fn(iput)[0], truth[0], C, S, B)

# Create artificially correct answer
truth = np.random.rand(1, 4 + C)
true_column = np.concatenate((truth[[0],:4], [[1.]], [[0.,0.,.0,.0]], [[0.]], truth[[0],-C:]), axis=1).reshape((1,5*B + C, 1, 1))
true_column = np.repeat(true_column, S[0], axis=2)
true_column = np.repeat(true_column, S[1], axis=3)

offset_x, offset_y = np.meshgrid(np.arange(0.,1.,1./9), np.arange(0.,1.,1./9))
true_column[:,[0,5]] -= offset_x.reshape((-1,1,9,9))
true_column[:,[1,6]] -= offset_y.reshape((-1,1,9,9))

fn2(true_column, truth)

manual_cost(true_column[0], truth[0], C, S, B)



