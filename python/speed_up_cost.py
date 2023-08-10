import numpy as np
from bnr_ml.utils.helpers import meshgrid2D
import theano
from theano import tensor as T
from time import time

import pdb

S = (7,7)
B = 3
C = 10

N = 2
y_pred= np.random.rand(N,5 * B + C,S[0], S[1]).astype(theano.config.floatX)
# y_truth = np.random.rand(N,4 + C).astype(theano.config.floatX)
y_truth = np.concatenate(
    (
        .25*np.random.rand(N,1),
        .25*np.random.rand(N,1),
        .75*np.random.rand(N,1),
        .75*np.random.rand(N,1),
        np.random.rand(N,C),
    ),
    axis=1
).astype(theano.config.floatX)
# ypred= np.random.rand(10,5 * B + C,S[0], S[1])
# ytruth = np.random.rand(10,4 + C)

output = T.as_tensor(y_pred.astype(theano.config.floatX))
ytruth = T.as_tensor(y_truth.astype(theano.config.floatX))
lmbda_coord = T.as_tensor_variable(5.)
lmbda_noobj = T.as_tensor_variable(.5)

block_height, block_width = 1. / S[0], 1./ S[1]

offsetx, offsety = np.meshgrid(np.arange(0,1,block_height), np.arange(0,1,block_width))

# theano version
offsetx, offsety = meshgrid2D(T.arange(0,1,block_height), T.arange(0,1,block_width))

x_idx, y_idx = [i*5 for i in range(B)], [i*5 + 1 for i in range(C)]
w_idx, h_idx = [i*5 + 2 for i in range(B)], [i*5 + 3 for i in range(C)]
conf = [i*5 + 4 for i in range(B)]

# theano version
x_idx, y_idx = T.arange(0,5*B,5), T.arange(1,5*B+ 1, 5)
w_idx, h_idx = T.arange(2,5*B+2,5), T.arange(3,5*B+3,5)
conf_idx = T.arange(4,5*B+4,5)

ypred[:,x_idx,:,:] += offsetx
ypred[:,y_idx,:,:] += offsety

# theano version
ypred = T.set_subtensor(output[:,x_idx,:,:],output[:,x_idx,:,:] + offsetx)
ypred = T.set_subtensor(output[:,y_idx,:,:],output[:,y_idx,:,:] + offsety)

# calculate the IOU score for each region and box
xi = np.maximum(ypred[:,x_idx,:,:], ytruth[:,0].reshape((10,1,1,1)))
xf = np.minimum(ypred[:,x_idx,:,:] + ypred[:,w_idx,:,:], (ytruth[:,0] + ytruth[:,2]).reshape((10,1,1,1)))

yi = np.maximum(ypred[:,y_idx,:,:], ytruth[:,0].reshape((10,1,1,1)))
yf = np.minimum(ypred[:,y_idx,:,:] + ypred[:,h_idx,:,:], (ytruth[:,0] + ytruth[:,2]).reshape((10,1,1,1)))

# theano version
xi = T.maximum(ypred[:,x_idx,:,:], ytruth[:,0].dimshuffle(0,'x','x','x'))
xf = T.minimum(ypred[:,x_idx,:,:] + ypred[:,w_idx,:,:], (ytruth[:,0] + ytruth[:,2]).dimshuffle(0,'x','x','x'))

yi = T.maximum(ypred[:,y_idx,:,:], ytruth[:,1].dimshuffle(0,'x','x','x'))
yf = T.minimum(ypred[:,y_idx,:,:] + ypred[:,h_idx,:,:], (ytruth[:,1] + ytruth[:,3]).dimshuffle(0,'x','x','x'))

isec = (xf - xi) * (yf - yi)
union = (ypred[:,w_idx] * ypred[:,h_idx]) + (ytruth[:,2] * ytruth[:,3]).reshape((10,1,1,1)) - isec
iou = np.maximum(isec/union, 0.)

# theano version
isec = (xf - xi) * (yf - yi)
union = (ypred[:,w_idx] * ypred[:,h_idx]) + (ytruth[:,2] * ytruth[:,3]).dimshuffle(0,'x','x','x') - isec
iou = T.maximum(isec/union, 0.)

cidx, _ = np.meshgrid(range(B),range(10))
# cidx = cidx.reshape((10,2,1,1))
# cidx = np.repeat(np.repeat(cidx,S[0],2),S[1],3)

# theano version
maxval_idx, a = meshgrid2D(T.arange(B), T.arange(ypred.shape[0]))
maxval_idx = maxval_idx.reshape((ypred.shape[0],-1))
maxval_idx = maxval_idx.dimshuffle(0,1,'x','x')
maxval_idx = T.repeat(T.repeat(maxval_idx,S[0],2),S[1],3)

ismax = np.equal(cidx, iou.argmax(axis=1).reshape((10,1,2,2)))

# theano version
ismax = T.eq(maxval_idx, iou.argmax(axis=1).dimshuffle(0,'x',1,2))

((iou[ismax] - ypred[:,conf,:,:][ismax])**2).reshape((10,2,2)).shape

width, height = np.ones(S) / S[1], np.ones(S) / S[0]
width, height = width.reshape((1,2,2)), height.reshape((1,2,2))

# theano version
width, height = T.ones(S) / S[1], T.ones(S) / S[0]
width, height = width.dimshuffle('x',0,1), height.dimshuffle('x',0,1)

xi = np.maximum(width, ytruth[:,0].reshape((10,1,1)))
xf = np.minimum(width + offsetx, (ytruth[:,0] + ytruth[:,2]).reshape((10,1,1)))

yi = np.maximum(height, ytruth[:,1].reshape((10,1,1)))
yf = np.minimum(height + offsety, (ytruth[:,1] + ytruth[:,3]).reshape((10,1,1)))

# theano version
xi = T.maximum(offsetx, ytruth[:,0].dimshuffle(0,'x','x'))
xf = T.minimum(offsetx + width, (ytruth[:,0] + ytruth[:,2]).dimshuffle(0,'x','x'))

yi = T.maximum(offsety, ytruth[:,1].dimshuffle(0,'x','x'))
yf = T.minimum(offsety + height, (ytruth[:,1] + ytruth[:,3]).dimshuffle(0,'x','x'))

isec = (xf - xi) * (yf - yi)
union = width * height + (ytruth[:,2] * ytruth[:,3]).reshape((10,1,1)) - isec
ioucell = np.maximum(isec / union, 0.)

# theano version
isec = (xf - xi) * (yf - yi)
union = width * height + (ytruth[:,2] * ytruth[:,3]).dimshuffle(0,'x','x') - isec
iou_cell = T.maximum(isec/union, 0.)

isinter = np.reshape(ioucell > .1, (10,-1,2,2))

# theano version
isinter = (iou_cell > .1).dimshuffle(0,'x',1,2)

tmp = np.bitwise_and(ismax, isinter)

# theano version
isbox_andcell = T.bitwise_and(ismax, isinter)

# theano
isinter = T.repeat(isinter, C, axis=1)

ypred[:,-C:][isinter.nonzero()].shape.eval()

clspred_truth = T.repeat(T.repeat(ytruth[:,-C:].dimshuffle(0,1,'x','x'), S[0], axis=2), S[1], axis=3)

clspred_truth.shape.eval()

tmp = T.sum((ypred[:,-C:,:,:][isinter.nonzero()] - clspred_truth[isinter.nonzero()])**2)
print tmp.eval()





cost = lmbda_coord * T.sum((ypred[:,conf_idx,:,:][isbox_andcell.nonzero()] - iou[isbox_andcell.nonzero()])**2) +         lmbda_noobj * T.sum((ypred[:,conf_idx,:,:][T.bitwise_not(isbox_andcell.nonzero())])**2) +         T.sum((ypred[:,x_idx,:,:][ismax.nonzero()].reshape((ytruth.shape[0],-1)) - ytruth[:,[0]])**2) +         T.sum((ypred[:,y_idx,:,:][ismax.nonzero()].reshape((ytruth.shape[0],-1)) - ytruth[:,[1]])**2) +         T.sum((ypred[:,w_idx,:,:][ismax.nonzero()].reshape((ytruth.shape[0],-1)).sqrt() - ytruth[:,[2]].sqrt())**2) +         T.sum((ypred[:,h_idx,:,:][ismax.nonzero()].reshape((ytruth.shape[0],-1)).sqrt() - ytruth[:,[3]].sqrt())**2) +         T.sum((ypred[:,-C:,:,:][isinter.nonzero()] - clspred_truth[isinter.nonzero()])**2)
        
        
        
        
        
        
        

def get_cost(output, truth, S, B, C, lmbda_coord=5., lmbda_noobj=0.5, iou_thresh=0.1):
    # calculate height/width of individual cell
    block_height, block_width = 1. / S[0], 1./ S[1]

    # get the offset of each cell
    offset_x, offset_y = meshgrid2D(T.arange(0,1,block_height), T.arange(0,1,block_width))
    
    # get indices for x,y,w,h,object-ness for easy access
    x_idx, y_idx = T.arange(0,5*B,5), T.arange(1,5*B, 5)
    w_idx, h_idx = T.arange(2,5*B,5), T.arange(3,5*B,5)
    conf_idx = T.arange(4,5*B+4,5)
    
    # Get position predictions with offsets.
    pred_x = output[:,x_idx] + offset_x.dimshuffle('x','x',0,1)
    pred_y = output[:,y_idx] + offset_y.dimshuffle('x','x',0,1)
    pred_w, pred_h, pred_conf = output[:,w_idx], output[:,h_idx], output[:,conf_idx]
    
    # Get intersection region bounding box coordinates
    xi = T.maximum(pred_x, truth[:,0].dimshuffle(0,'x','x','x'))
    xf = T.minimum(pred_x + pred_w, (truth[:,0] + truth[:,2]).dimshuffle(0,'x','x','x'))
    yi = T.maximum(pred_y, truth[:,1].dimshuffle(0,'x','x','x'))
    yf = T.minimum(pred_y + pred_h, (truth[:,1] + truth[:,3]).dimshuffle(0,'x','x','x'))
    
    # Calculate iou score for predicted boxes and truth
    isec = (xf - xi) * (yf - yi)
    union = (pred_w * pred_h) + (truth[:,2] * truth[:,3]).dimshuffle(0,'x','x','x') - isec
    iou = T.maximum(isec/union, 0.)

    # Get index matrix representing max along the 1st dimension for the iou score (reps 'responsible' box).
    maxval_idx, a = meshgrid2D(T.arange(B), T.arange(truth.shape[0]))
    maxval_idx = maxval_idx.reshape((truth.shape[0],-1))
    maxval_idx = maxval_idx.dimshuffle(0,1,'x','x')
    maxval_idx = T.repeat(T.repeat(maxval_idx,S[0],2),S[1],3)
    is_max = T.eq(maxval_idx, iou.argmax(axis=1).dimshuffle(0,'x',1,2))
    
    # Get matrix for the width/height of each cell
    width, height = T.ones(S) / S[1], T.ones(S) / S[0]
    width, height = width.dimshuffle('x',0,1), height.dimshuffle('x',0,1)
    
    # Get bounding box for intersection between CELL and ground truth box.
    xi = T.maximum(offset_x, truth[:,0].dimshuffle(0,'x','x'))
    xf = T.minimum(offset_x + width, (truth[:,0] + truth[:,2]).dimshuffle(0,'x','x'))
    yi = T.maximum(offset_y, truth[:,1].dimshuffle(0,'x','x'))
    yf = T.minimum(offset_y + height, (truth[:,1] + truth[:,3]).dimshuffle(0,'x','x'))

    # Calculate iou score for the cell.
    isec = (xf - xi) * (yf - yi)
    union = width * height + (truth[:,2] * truth[:,3]).dimshuffle(0,'x','x') - isec
    iou_cell = T.maximum(isec/union, 0.)
    
    # Get logical matrix representing minimum iou score for cell to be considered overlapping ground truth.
    is_inter = (iou_cell > iou_thresh).dimshuffle(0,'x',1,2)
    
    # Get logical matrix for cells and boxes which overlap and are responsible for prediction.
    isbox_andcell = T.bitwise_and(is_max, is_inter)
    
    # repeat "cell overlaps" logical matrix for the number of classes.
    is_inter = T.repeat(is_inter, C, axis=1)
    
    # repeat the ground truth for class probabilities for each cell.
    clspred_truth = T.repeat(T.repeat(truth[:,-C:].dimshuffle(0,1,'x','x'), S[0], axis=2), S[1], axis=3)
    
    # calculate cost
    cost = lmbda_coord * T.sum((pred_conf - iou)[isbox_andcell.nonzero()]**2) +         lmbda_noobj * T.sum((pred_conf[T.bitwise_not(isbox_andcell).nonzero()])**2) +         T.sum((pred_x[is_max.nonzero()].reshape((truth.shape[0],-1)) - truth[:,[0]])**2) +         T.sum((pred_y[is_max.nonzero()].reshape((truth.shape[0],-1)) - truth[:,[1]])**2) +         T.sum((pred_w[is_max.nonzero()].reshape((truth.shape[0],-1)).sqrt() - truth[:,[2]].sqrt())**2) +         T.sum((pred_h[is_max.nonzero()].reshape((truth.shape[0],-1)).sqrt() - truth[:,[3]].sqrt())**2) +         T.sum((output[:,-C:][is_inter.nonzero()] - clspred_truth[is_inter.nonzero()])**2)
    
    return cost

output = T.tensor4('output')
truth = T.matrix('truth')
lmbda_coord = T.scalar('lambda_coord')
lmbda_noobj = T.scalar('lambda_noobj')

output = T.as_tensor(y_pred)
truth = T.as_tensor(y_truth)
lmbda_coord = T.as_tensor_variable(5.)
lmbda_noobj = T.as_tensor_variable(0.5)

cost = get_cost(output, truth, S, B, C)

# cost_fn = theano.function([output,truth], cost)
cost_fn = theano.function([output, truth], cost)

grad = T.grad(cost, output)

grad_fn = theano.function([output, truth], grad, on_unused_input='ignore')

grad_fn(y_pred, y_truth).shape



