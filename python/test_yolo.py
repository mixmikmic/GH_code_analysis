get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import theano
from theano import tensor as T
from theano.tensor.signal.pool import pool_2d
import numpy as np

from bnr_ml.nnet import layers
from bnr_ml.utils import helpers

import pdb

range(0,10,2)

reload(layers)

reload(helpers)

input = T.tensor4('input')

l1 = layers.ConvolutionalLayer2D(
    (None,3,200,200),
    ((3,3)),
    16,
    input=input
)

l2 = layers.ConvolutionalLayer2D(
    l1.get_output_shape(),
    (3,3),
    16,
    input=layers.NonLinLayer(T.nnet.relu, input=l1.output).output
)

l3 = layers.PoolLayer2D(
    l2.get_output_shape(),
    (7,7),
    input=layers.NonLinLayer(T.nnet.relu, input=l2.output).output
)

l4 = layers.ConvolutionalLayer2D(
    l3.get_output_shape(),
    (3,3),
    16,
    input=l3.output
)
l5 = layers.ConvolutionalLayer2D(
    l4.get_output_shape(),
    (3,3),
    16,
    input=layers.NonLinLayer(T.nnet.relu, input=l4.output).output
)

l6 = layers.PoolLayer2D(
    l5.get_output_shape(),
    (12,12),
    input=layers.NonLinLayer(T.nnet.relu, input=l5.output).output
)

l7 = layers.FCLayer(
    l6.get_output_shape(),
    1024,
    input=layers.NonLinLayer(T.nnet.sigmoid, input=l6.output).output
)

l8 = layers.FCLayer(
    l7.get_output_shape(),
    2*2*13,
    input=layers.NonLinLayer(T.nnet.sigmoid, input=l7.output).output
)

l8 = l8.output.reshape((-1, 13, 2,2))

net = {}
net['l1'] = l1
net['l2'] = l2
net['l3'] = l3
net['l4'] = l4
net['l5'] = l5
net['l6'] = l6
net['l7'] = l7
net['output'] = l8

net

fun = theano.function([input], l6.output)

X = np.random.randn(10,3,200,200)

tmp = fun(X)

tmp.shape


class YoloObjectDetectorError(Exception):
		pass


class YoloObjectDetector(object):
	'''

	'''
	def __init__(
		self,
		network,
		input_shape,
		num_classes,
		S,
		B,
		input=None):
		'''
		network:
		--------
			Dict with the entire network defined, must have a "feature_map" and "output" layer.
			You must be able to call .get_output() on these layers.
		'''
		self.network = network
		self.num_classes = num_classes
		self.S = S
		self.B = B
		if input is None:
			input = T.tensor4('input')
		self.input = input
		self.input_shape = input_shape

	def _get_cost(self, output, probs, dims, lmbda_coord=10., lmbda_noobj = .1, iou_thresh = .1):
		lmbda_coord = T.as_tensor_variable(lmbda_coord)
		lmbda_noobj = T.as_tensor_variable(lmbda_noobj)
		iou_thresh = T.as_tensor_variable(iou_thresh)
# 		output = network['output']
# 		if isinstance(output, AbstractNNetLayer):
# 			output = output.get_output()

		w1, w2 = np.ceil(float(self.input_shape[2]) / self.S[0]), np.ceil(float(self.input_shape[3]) / self.S[1])

		def scale_dims(dims):
			newdims = T.set_subtensor(dims[:,0], (dims[:,0] - i * w1) / self.input_shape[2])
			newdims = T.set_subtensor(newdims[:,1], (newdims[:,1] - j * w2) / self.input_shape[3])
			newdims = T.set_subtensor(newdims[:,2], (newdims[:,2] / self.input_shape[2]))
			newdims = T.set_subtensor(newdims[:,3], (newdims[:,3] / self.input_shape[3]))
			return newdims
		def unscale_dims(dims):
			newdims = T.set_subtensor(dims[:,0], dims[:,0] * self.input_shape[2] + i * w1)
			newdims = T.set_subtensor(newdims[:,1], newdims[:,1] * self.input_shape[3] + j * w2)
			newdims = T.set_subtensor(newdims[:,2], newdims[:,2] * self.input_shape[2])
			newdims = T.set_subtensor(newdims[:,3], newdims[:,3] * self.input_shape[3])
			return newdims

		cost = T.as_tensor_variable(0.)
		for i in range(self.S[0]):
			for j in range(self.S[1]):
				preds_ij = []
				ious = []

				newdims = scale_dims(dims)

				for k in range(self.B):
					pred_ijk = output[:,k*5:(k+1) * 5,i,j] # single prediction for cell and box

					# get intersecion box coordinates relative to boxes
					isec_xi = T.maximum(newdims[:,0], pred_ijk[:,0])
					isec_yi = T.maximum(newdims[:,1], pred_ijk[:,1])
					isec_xf = T.minimum(newdims[:,0] + newdims[:,2], pred_ijk[:,0] + pred_ijk[:,2])
					isec_yf = T.minimum(newdims[:,1] + newdims[:,3], pred_ijk[:,1] + pred_ijk[:,3])

					isec = T.maximum((isec_xf - isec_xi) * (isec_yf - isec_yi), 0.)

					union = newdims[:,2] * newdims[:,3] + pred_ijk[:,2] * pred_ijk[:,3] - isec

					iou = isec / union

					preds_ij.append(pred_ijk.dimshuffle(0,1,'x'))
					ious.append(iou.dimshuffle(0,'x'))

				# Determine if the image intersects with the cell
				isec_xi = T.maximum(newdims[:,0], 0.)
				isec_yi = T.maximum(newdims[:,1], 0.)
				isec_xf = T.minimum(newdims[:,0] + newdims[:,2], 1. / self.S[0])
				isec_yf = T.minimum(newdims[:,1] + newdims[:,3], 1. / self.S[1])

				isec = T.maximum((isec_xf - isec_xi) * (isec_yf - isec_yi), 0.)

				union = newdims[:,2] * newdims[:,3] + pred_ijk[:,2] * pred_ijk[:,3] - isec

				iou = isec / union

				is_not_in_cell = (iou < iou_thresh).nonzero()

				preds_ij = T.concatenate(preds_ij, axis=2)
				ious = T.concatenate(ious, axis=1)

				iou_max = T.argmax(ious, axis=1)

				# get final values for predictions
				row,col = meshgrid2D(T.arange(preds_ij.shape[0]), T.arange(preds_ij.shape[1]))
				dep,col = meshgrid2D(iou_max, T.arange(preds_ij.shape[1]))

				preds_ij = preds_ij[row,col,dep].reshape(preds_ij.shape[:2])

				# get final values for IoUs
				row = T.arange(preds_ij.shape[0])
				ious = ious[row, iou_max]

				is_box_not_in_cell = (ious < iou_thresh).nonzero()

				cost_ij_t1 = (preds_ij[:,0] - newdims[:,0])**2 + (preds_ij[:,1] - newdims[:,1])**2
				cost_ij_t1 += (T.sqrt(preds_ij[:,2]) - T.sqrt(newdims[:,2]))**2 + (T.sqrt(preds_ij[:,3]) - T.sqrt(newdims[:,3]))**2
				cost_ij_t1 *= lmbda_coord

				cost_ij_t1 += lmbda_noobj * (preds_ij[:,4] - ious)**2

				cost_ij_t2 = lmbda_noobj * T.sum((probs - output[:,-self.num_classes:,i,j])**2, axis=1)

				cost_ij_t1 = T.set_subtensor(cost_ij_t1[is_box_not_in_cell], 0.)
				cost_ij_t2 = T.set_subtensor(cost_ij_t2[is_not_in_cell], 0.)

				cost += cost_ij_t1 + cost_ij_t2

				dims = unscale_dims(newdims)

		cost = cost.mean()

		return cost

yolo = YoloObjectDetector(
    net,
    (None,3,200,200),
    3,
    (2,2),
    2,
    input=input
)

output = T.tensor4('output')

proba = T.matrix('probs')
dims = T.matrix('dims')

cost = yolo._get_cost(output, proba, dims)

fun = theano.function([output, proba, dims], cost)

fun(out, probdat, dimdat)

prob_dat = np.random.rand(10, 3)
prob_dat /= prob_dat.sum(axis=1, keepdims=True)

dim_dat = np.round(200 * np.random.rand(10,5))

tmp = fun(X, prob_dat, dim_dat)

tmp

def get_cost(output, probs, dims, S, B, input_shape, num_classes, lmbda_coord=10., lmbda_noobj = .1, iou_thresh = .1):
#     lmbda_coord = T.as_tensor_variable(lmbda_coord)
#     lmbda_noobj = T.as_tensor_variable(lmbda_noobj)
#     iou_thresh = T.as_tensor_variable(iou_thresh)
#     output = self.network['output']

    w1, w2 = np.ceil(float(input_shape[2]) / S[0]), np.ceil(float(input_shape[3]) / S[1])

    cost = 0.
    for i in range(S[0]):
        for j in range(S[1]):
            preds_ij = []
            ious = []

            newdims = np.copy(dims)
#             newdims[:,0] = newdims[:,0] / input_shape[2] # CHANGES HERE
#             newdims[:,1] = newdims[:,1] / input_shape[3]
            newdims[:,2] = newdims[:,2] / input_shape[2]
            newdims[:,3] = newdims[:,3] / input_shape[3]
            newdims[:,0] = (newdims[:,0] - i * w1) / input_shape[2] # CHANGE HERE
            newdims[:,1] = (newdims[:,1] - j * w2) / input_shape[3]
            
            for k in range(B):
                pred_ijk = output[:,k*5:(k+1) * 5,i,j] # single prediction for cell and box

                # get intersection box coordinates relative to boxes
                isec_xi = np.maximum(newdims[:,0], pred_ijk[:,0])
                isec_yi = np.maximum(newdims[:,1], pred_ijk[:,1])
                isec_xf = np.minimum(newdims[:,0] + newdims[:,2], pred_ijk[:,0] + pred_ijk[:,2]) #CHANGE HERE
                isec_yf = np.minimum(newdims[:,1] + newdims[:,3], pred_ijk[:,1] + pred_ijk[:,3])

                isec = np.max((isec_xf - isec_xi) * (isec_yf - isec_yi), 0.)

                union = newdims[:,2] * newdims[:,3] + pred_ijk[:,2] * pred_ijk[:,3] - isec # CHANGE HERE

                iou = isec / union

                preds_ij.append(pred_ijk.reshape(pred_ijk.shape + (1,)))
                ious.append(iou.reshape((-1,1)))
                
            # determine if intersects with cell i,j  # CHANGE HERE
            isec_xi = np.maximum(newdims[:,0], 0.)
            isec_yi = np.maximum(newdims[:,1], 0.)
            isec_xf = np.minimum(newdims[:,0] + newdims[:,2], 1. / S[0]) #CHANGE HERE
            isec_yf = np.minimum(newdims[:,1] + newdims[:,3], 1. / S[1])

            isec = np.max((isec_xf - isec_xi) * (isec_yf - isec_yi), 0.)

            union = newdims[:,2] * newdims[:,3] + pred_ijk[:,2] * pred_ijk[:,3] - isec # CHANGE HERE

            iou = isec / union
            
            is_not_in_cell = (iou < iou_thresh).nonzero()

            preds_ij = np.concatenate(preds_ij, axis=2)
            ious = np.concatenate(ious, axis=1)

            iou_max = np.argmax(ious, axis=1)

            # get final values for predictions
            row,col = np.meshgrid(np.arange(preds_ij.shape[0]), np.arange(preds_ij.shape[1]))
            dep,col = np.meshgrid(iou_max, np.arange(preds_ij.shape[1]))
            
            preds_ij = preds_ij[row.flatten(),col.flatten(),dep.flatten()].reshape(preds_ij.shape[:2])

            # get final values for IoUs
            row = np.arange(preds_ij.shape[0]) # CHANGES HERE
            ious = ious[row, iou_max]

            is_not_valid = (ious < iou_thresh).nonzero()

            # calc cost for term 1 involving bounding box predictions
            cost_ij_t1 = (preds_ij[:,0] - newdims[:,0])**2 + (preds_ij[:,1] - newdims[:,1])**2
            cost_ij_t1 += (np.sqrt(preds_ij[:,2]) - np.sqrt(newdims[:,2]))**2 + (np.sqrt(preds_ij[:,3]) - np.sqrt(newdims[:,3]))**2
            cost_ij_t1 *= lmbda_coord

            cost_ij_t1 += lmbda_noobj * (preds_ij[:,4] - ious)**2

            
            cost_ij_t2 = lmbda_noobj * np.sum((probs - output[:,-num_classes:,i,j])**2, axis=1)
            
            cost_ij_t1[np.bitwise_not(is_not_valid)] = 0.
            cost_ij_t2[np.bitwise_not(is_not_in_cell)] = 0.

            
            cost += cost_ij_t1 + cost_ij_t2

    cost = cost.mean()

    return cost

out = fun(X)

tmp = np.concatenate()

get_cost(out, probdat, dimdat, (2,2), 2, (None, 3, 200, 200), 3)

probdat = np.asarray([[1.,0.,0.]])
dimdat = np.array([[50.,50.,100.,100.,1./6]])

tmp = np.asarray([.25,.25,.5,.5,1.,0,0,0,0,0,1.,0.,0.,])
out = np.zeros((1,13,2,2))
out[0,:,0,0] = tmp



