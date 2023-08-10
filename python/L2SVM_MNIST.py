import zipml
import numpy as np
import time

Z = zipml.ZipML_SGD(on_pynq=1, bitstreams_path=zipml.BITSTREAMS, ctrl_base=0x41200000, dma_base=0x40400000, dma_buffer_size=32*1024*1024)

get_ipython().system('./get_mnist.sh')

start = time.time()
Z.load_libsvm_data('./mnist/mnist', 10000, 784)
print('Data loaded, time: ' + str(time.time()-start) )
Z.a_normalize(to_min1_1=0, row_or_column='r') # Normalize features of the data set

# Set training related parameters
num_epochs = 10
step_size = 1.0/(1 << 12)
cost_pos = 1.0
cost_neg = 1.5

import scipy.misc
for i in range(0,10):
    scipy.misc.toimage( np.reshape(Z.a[i,1:785], (28,28)) ).save('out'+ str(i) +'.jpg')

from PIL import Image
from IPython.display import display

for i in range(0,10):
    im = Image.open('./out'+str(i)+'.jpg')
    display(im)

xs_CPU = np.zeros((Z.num_features, 10))
for c in range(0, 10):
	start = time.time()
	Z.b_binarize(c) # Binarize the labels of the data set
	print('b binarized for ' + str(c) + ", time: " + str(time.time()-start) )
	start = time.time()
    
    # Train model on the CPU
	x_history = Z.L2SVM_SGD(num_epochs, step_size, cost_pos, cost_neg, regularize=0, use_binarized=1)
    
	print('Training time: ' + str(time.time()-start) )
    # Print losses after each epoch
	initial_loss = Z.calculate_L2SVM_loss(np.zeros(Z.num_features), cost_pos, cost_neg, 0, 1)
	print('Initial loss: ' + str(initial_loss))
	for e in range(0, num_epochs):
		loss = Z.calculate_L2SVM_loss(x_history[:,e], cost_pos, cost_neg, 0, 1)
		print('Epoch ' + str(e) + ' loss: ' + str(loss) )

	xs_CPU[:,c] = x_history[:,num_epochs-1]

xs_floatFSGD = np.zeros((Z.num_features, 10))
for c in range(0, 10):
	start = time.time()
	Z.b_binarize(c)
	print('b binarized for ' + str(c) + ", time: " + str(time.time()-start) )
	start = time.time()
	
    # Train model on the FPGA
	Z.configure_SGD_FPGA(num_epochs, step_size, cost_pos, cost_neg, 1, c)
	x_history = Z.SGD_FPGA(num_epochs)

	print('Training time: ' + str(time.time()-start) )
    # Print losses after each epoch
	initial_loss = Z.calculate_L2SVM_loss(np.zeros(Z.num_features), cost_pos, cost_neg, 0, 1)
	print('Initial loss: ' + str(initial_loss))
	for e in range(0, num_epochs):
		loss = Z.calculate_L2SVM_loss(x_history[:,e], cost_pos, cost_neg, 0, 1)
		print('Epoch ' + str(e) + ' loss: ' + str(loss) )

	xs_floatFSGD[:,c] = x_history[:,num_epochs-1]

# Quantize the features of the data set
Z.a_quantize(quantization_bits=1)

xs_qFSGD = np.zeros((Z.num_features, 10))
for c in range(0, 10):
	start = time.time()
	Z.b_binarize(c)
	print('b binarized for ' + str(c) + ", time: " + str(time.time()-start) )
	start = time.time()
	
    # Train model on the FPGA
	Z.configure_SGD_FPGA(num_epochs, step_size, cost_pos, cost_neg, 1, c)
	x_history = Z.SGD_FPGA(num_epochs)

	print('Training time: ' + str(time.time()-start) )
    # Print losses after each epoch
	initial_loss = Z.calculate_L2SVM_loss(np.zeros(Z.num_features), cost_pos, cost_neg, 0, 1)
	print('Initial loss: ' + str(initial_loss))
	for e in range(0, num_epochs):
		loss = Z.calculate_L2SVM_loss(x_history[:,e], cost_pos, cost_neg, 0, 1)
		print('Epoch ' + str(e) + ' loss: ' + str(loss) )

	xs_qFSGD[:,c] = x_history[:,num_epochs-1]

Z.load_libsvm_data('./mnist/mnist.t', 10000, 784)
Z.a_normalize(to_min1_1=0, row_or_column='r');

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
matches_CPU = Z.multi_classification(xs_CPU, classes)
matches_floatFSGD = Z.multi_classification(xs_floatFSGD, classes)
matches_qFSGD = Z.multi_classification(xs_qFSGD, classes)

print('CPU training -> Matches ' + str(matches_CPU) + ' out of ' + str(Z.num_samples) + ' samples.')
print('FPGA training -> Matches ' + str(matches_floatFSGD) + ' out of ' + str(Z.num_samples) + ' samples.')
print('1-bit FPGA training -> Matches ' + str(matches_qFSGD) + ' out of ' + str(Z.num_samples) + ' samples.')



