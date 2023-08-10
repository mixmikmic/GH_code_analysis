import mxnet as mx
import numpy as np

#Training data
train_data = np.array([[1,2],[3,4],[5,6],[3,2],[7,1],[6,9]])
train_label = np.array([5.1,10.9,17.1,6.9,9.1,24])
batch_size = 1

#Evaluation Data
eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([11.1,25.9,16.2])

train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size, shuffle=True, label_name='lin_reg_label') 
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)

# Defining the Model Structure

X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label')
fully_connected_layer  = mx.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)
linear_regression_output = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="linear_regression_output")

mx.viz.plot_network(symbol=linear_regression_output)

model = mx.mod.Module(
    symbol = linear_regression_output ,
    data_names=['data'], 
    label_names = ['lin_reg_label']
)

model.fit(train_iter, eval_iter,
            optimizer_params={'learning_rate':0.01},
            num_epoch=1000,
            batch_end_callback = mx.callback.Speedometer(batch_size, 2))

#Inference - predicting y for new unseen values of (x1, x2)
model.predict(eval_iter).asnumpy()

#Evaluation - Calculate metrics of how well our model is performing on unseen evaluation data
metric = mx.metric.MSE()
model.score(eval_iter, metric)

#Evaluation Data
eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([14.1,27.1,19.1]) #Adding 0.1 to each of the values 
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)

model.score(eval_iter, metric)

