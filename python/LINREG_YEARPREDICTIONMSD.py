import zipml
import numpy as np
import time

Z = zipml.ZipML_SGD(on_pynq=1, bitstreams_path=zipml.BITSTREAMS, ctrl_base=0x41200000, dma_base=0x40400000, dma_buffer_size=32*1024*1024)

get_ipython().system('./get_yearpredictionMSD.sh')

start = time.time()
Z.load_libsvm_data('./YearPredictionMSD/YearPredictionMSD', 50000, 90)
print('a loaded, time: ' + str(time.time()-start) )
Z.a_normalize(to_min1_1=1, row_or_column='c');
Z.b_normalize(to_min1_1=0)

# Set training related parameters
num_epochs = 10
step_size = 1.0/(1 << 12)

start = time.time()

# Train on the CPU
x_history = Z.LINREG_SGD(num_epochs=num_epochs, step_size=step_size, regularize=0)

print('Performed linear regression on cadata. Training time: ' + str(time.time()-start))
# Print losses after each epoch
initial_loss = Z.calculate_LINREG_loss(np.zeros(Z.num_features), 0)
print('Initial loss: ' + str(initial_loss))
for e in range(0, num_epochs):
	loss = Z.calculate_LINREG_loss(x_history[:,e], 0)
	print('Epoch ' + str(e) + ' loss: ' + str(loss) )

start = time.time()

# Train on FPGA
Z.configure_SGD_FPGA(num_epochs, step_size, -1, -1, 0, 0)
x_history = Z.SGD_FPGA(num_epochs)

print('FPGA train time: ' + str(time.time()-start) )
# Print losses after each epoch
initial_loss = Z.calculate_LINREG_loss(np.zeros(Z.num_features), 0)
print('Initial loss: ' + str(initial_loss))
for e in range(0, num_epochs):
	loss = Z.calculate_LINREG_loss(x_history[:,e], 0)
	print('Epoch ' + str(e) + ' loss: ' + str(loss) )

Z.a_quantize(quantization_bits=8)

start = time.time()

# Train on FPGA
Z.configure_SGD_FPGA(num_epochs, step_size, -1, -1, 0, 0)
x_history = Z.SGD_FPGA(num_epochs)

print('FPGA train time: ' + str(time.time()-start) )
# Print losses after each epoch
initial_loss = Z.calculate_LINREG_loss(np.zeros(Z.num_features), 0)
print('Initial loss: ' + str(initial_loss))
for e in range(0, num_epochs):
	loss = Z.calculate_LINREG_loss(x_history[:,e], 0)
	print('Epoch ' + str(e) + ' loss: ' + str(loss) )



