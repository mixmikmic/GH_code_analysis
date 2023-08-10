import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  # Config the logging
np.random.seed(777)

xy = np.loadtxt(os.path.join('data-03-diabetes.csv'), delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

train_x_data, test_x_data = x_data[:700], x_data[700:]
train_y_data, test_y_data = y_data[:700], y_data[700:]
print(train_x_data.shape, test_x_data.shape, train_y_data.shape, test_y_data.shape)

# hyper-parameters
sample_num = x_data.shape[0]
dimension = x_data.shape[1]
batch_size = 32

data = mx.sym.Variable("data")
target = mx.sym.Variable("target")
fc = mx.sym.FullyConnected(data=data, num_hidden=1, name='fc')
pred = mx.sym.LogisticRegressionOutput(data=fc, label=target)

net = mx.mod.Module(symbol=pred,
                    data_names=['data'],
                    label_names=['target'],
                    context=mx.gpu(0))
net.bind(data_shapes = [mx.io.DataDesc(name='data', shape=(batch_size, dimension), layout='NC')],
         label_shapes= [mx.io.DataDesc(name='target', shape=(batch_size, 1), layout='NC')])
net.init_params(initializer=mx.init.Normal(sigma=0.01))
net.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': 1E-2, 'momentum': 0.9})

# First constructing the training iterator and then fit the model
train_iter = mx.io.NDArrayIter(train_x_data, train_y_data, batch_size, shuffle=True, label_name='target')
metric = mx.metric.CustomMetric(feval=lambda labels, pred: ((pred > 0.5) == labels).mean(),
                                name="acc")
net.fit(train_data=train_iter, eval_metric=metric, num_epoch=15)

test_iter = mx.io.NDArrayIter(test_x_data, None, batch_size, shuffle=False, label_name=None)

pred_class = (fc > 0)
test_net = mx.mod.Module(symbol=pred_class,
                         data_names=['data'],
                         label_names=None,
                         context=mx.gpu(0))
test_net.bind(data_shapes=[mx.io.DataDesc(name='data', shape=(batch_size, dimension), layout='NC')],
              label_shapes=None,
              for_training=False,
              shared_module=net)
out = test_net.predict(eval_data=test_iter)
print(out.asnumpy())

acc = np.equal(test_y_data, out.asnumpy()).mean()
print("Accuracy: %.4g" %acc)



