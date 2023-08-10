get_ipython().system('./get_galaxymerger.sh')

from PIL import Image
from IPython.display import display

merger = Image.open('./galaxymerger/merger.jpeg')
non_merger = Image.open('./galaxymerger/non_merger.jpeg')
print('merging galaxies:')
display(merger.resize((300,300), Image.ANTIALIAS) )
print('no merging:')
display(non_merger.resize((300,300), Image.ANTIALIAS) )

import zipml
import numpy as np
import time

Z = zipml.ZipML_SGD(on_pynq=1, bitstreams_path=zipml.BITSTREAMS, ctrl_base=0x41200000, dma_base=0x40400000, dma_buffer_size=32*1024*1024)

start = time.time()
Z.load_libsvm_data('./galaxymerger/galaxy_train.dat', 3000, 2048)
print('a loaded, time: ' + str(time.time()-start) )
Z.a_normalize(to_min1_1=0, row_or_column='r')
Z.b_binarize(1)
print('b binarized for ' + str(1.0) + ", time: " + str(time.time()-start) )

# Set training related parameters
num_epochs = 50
step_size = 1.0/(1 << 8)
cost_pos = 1.0
cost_neg = 1.0

start = time.time()
x_history_CPU = Z.L2SVM_SGD(num_epochs, step_size, cost_pos, cost_neg, 0, 1)
print('Training time: ' + str(time.time()-start) )
initial_loss = Z.calculate_L2SVM_loss(np.zeros(Z.num_features), cost_pos, cost_neg, 0, 1)
print('Initial loss: ' + str(initial_loss))
for e in range(0, num_epochs):
	loss = Z.calculate_L2SVM_loss(x_history_CPU[:,e], cost_pos, cost_neg, 0, 1)
	print('Epoch ' + str(e) + ' loss: ' + str(loss) )

start = time.time()
Z.configure_SGD_FPGA(num_epochs, step_size, cost_pos, cost_neg, 1, 1.0)
x_history_FPGA = Z.SGD_FPGA(num_epochs)
print('Training time: ' + str(time.time()-start) )
initial_loss = Z.calculate_L2SVM_loss(np.zeros(Z.num_features), cost_pos, cost_neg, 0, 1)
print('Initial loss: ' + str(initial_loss))
for e in range(0, num_epochs):
	loss = Z.calculate_L2SVM_loss(x_history_FPGA[:,e], cost_pos, cost_neg, 0, 1)
	print('Epoch ' + str(e) + ' loss: ' + str(loss) )

Z.load_libsvm_data('./galaxymerger/galaxy_test.dat', 1000, 2048)
Z.a_normalize(to_min1_1=0, row_or_column='r')
Z.b_binarize(1)

matches = Z.binary_classification(x_history_FPGA[:,e])
print('matches ' + str(matches) + ' out of ' + str(Z.num_samples) + ' samples.')

