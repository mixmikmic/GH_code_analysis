import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl
get_ipython().run_line_magic('matplotlib', 'inline')

num_dataset = np.array([[0.22, 0.34, 0], [0.21, 0.37, 0], [0.25, 0.31, 0], [0.76, 0.19, 1], [0.84, 0.14, 1]])

features = num_dataset[:, :2]
labels = num_dataset[:, 2].reshape((num_dataset.shape[0], 1))

plt.scatter(features[:,0], features[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input Data')

dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1

num_output = labels.shape[1]

dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]

perceptron = nl.net.newp([dim1, dim2], num_output)

error_progress = perceptron.train(features, labels, epochs=100, show=20, lr=0.03)

plt.plot(error_progress)
plt.xlabel('Number of Epochs')
plt.ylabel('Training Error')
plt.title('Training Error Progress')
plt.grid()

perceptron.sim([[0.81, 0.23]])

perceptron.sim([[0.27, 0.42]])

text = np.loadtxt('data_simple_nn.txt')
data = text[:, 0:2]
labels = text[:, 2:]

text

data

labels

plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input Data')

dim1_min, dim1_max = data[:,0].min(), data[:,0].max()
dim2_min, dim2_max = data[:,1].min(), data[:,1].max()
num_output = labels.shape[1]

dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
snn = nl.net.newp([dim1, dim2], num_output)

error_progress = snn.train(data, labels, epochs=100, show=20, lr=0.03)

plt.plot(error_progress)
plt.xlabel('Number of Epochs')
plt.ylabel('Training Error')
plt.title('Training Error Progress')
plt.grid()

print('Testing Data:\n')
testing_data = [[0.3, 4.2], [4.3, 0.5], [4.6, 8]]
for i in testing_data:
    print(i, '==>', snn.sim([i])[0])

min_vals = -20
max_vals = 20
num_points = 140

x = np.linspace(min_vals, max_vals, num_points)
y = 3*np.square(x)+5
y /= np.linalg.norm(y)

data = x.reshape(num_points,1)
labels = y.reshape(num_points, 1)

plt.scatter(data, labels)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data Points')

mlnn = nl.net.newff([[min_vals, max_vals]], [10,6,1])
mlnn.trainf = nl.train.train_gd
error_progress = mlnn.train(data, labels, epochs=2000, show=100, goal=0.01)

output = mlnn.sim(data)
y_pred = output.reshape(num_points)

x_dense = np.linspace(min_vals, max_vals, num_points*2)
y_dense_pred = mlnn.sim(x_dense.reshape(x_dense.size,1)).reshape(x_dense.size)

plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('Actual vs. Predicted')

def get_data(num_points):
    w_one = 0.6 * np.sin(np.arange(0, num_points))
    w_two = 3.5 * np.sin(np.arange(0, num_points))
    w_three = 1.2 * np.sin(np.arange(0, num_points))
    w_four = 4.6 * np.sin(np.arange(0, num_points))
    
    a_one = np.ones(num_points)
    a_two = 2.2 + np.zeros(num_points)
    a_three = 3.1 * np.ones(num_points)
    a_four = 0.9 + np.zeros(num_points)
    
    wave = np.array([w_one, w_two, w_three, w_four]).reshape(num_points*4,1)
    amp = np.array([a_one, a_two, a_three, a_four]).reshape(num_points*4,1)
    
    return wave, amp

def visualize_output(nn, num_points_test):
    wave, amp = get_data(num_points_test)
    output = nn.sim(wave)
    plt.plot(amp.reshape(num_points_test*4))
    plt.plot(output.reshape(num_points_test*4))
    
if __name__ == '__main__':
    num_points = 50
    wave, amp = get_data(num_points)

nn = nl.net.newelm([[-2,2]], [10,1], [nl.trans.TanSig(), nl.trans.PureLin()])

nn.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
nn.layers[1].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
nn.init()

error_progress = nn.train(wave, amp, epochs=1200, show=100, goal=0.01)

output = nn.sim(wave)

plt.subplot(211)
plt.plot(error_progress)
plt.xlabel('#Epochs')
plt.ylabel('Error (MSE)')

plt.subplot(212)
plt.plot(amp.reshape(num_points*4))
plt.plot(output.reshape(num_points*4))
plt.legend(['Original', 'Predicted'])

plt.figure()
plt.subplot(211)
visualize_output(nn, 82)
plt.xlim([0, 300])

plt.subplot(212)
visualize_output(nn, 49)
plt.xlim([0, 300])
plt.show()







