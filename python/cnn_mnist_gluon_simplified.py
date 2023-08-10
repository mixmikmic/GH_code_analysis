import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)

class BaseCNNClassifier(mx.gluon.Block):
    def __init__(self, ctx):
        super(BaseCNNClassifier, self).__init__()
        self.ctx = ctx
        self.net = None
        
    #@override
    def build_model(self, convs, num_fc, num_classes):
        '''
        Default activation is relu
        '''
        # convs = [(channel, kernel_sz, pool_siz)triplets *N]
        cnn_layers = gluon.nn.HybridSequential(prefix='')
        for ch, k_sz, p_sz in convs:
            cnn_layers.add(gluon.nn.Conv2D(channels=ch, kernel_size=k_sz, activation='relu'))
            cnn_layers.add(gluon.nn.MaxPool2D(pool_size=p_sz, strides=2)) # strides fixed for now
            
        net = gluon.nn.HybridSequential()
        with net.name_scope():
            net.add(cnn_layers)
            # Flatten and apply fully connected layers
            net.add(gluon.nn.Flatten())
            net.add(gluon.nn.Dense(num_fc, activation="relu"))
            net.add(gluon.nn.Dense(num_classes))

        # speed up execution with hybridization
        net.hybridize()
        self.net = net
    
    def forward(self):
        pass

    def compile_model(self, loss=None, optimizer='sgd', lr=1E-3, init_mg=2.24):
        print self.net
        self.net.collect_params().initialize(mx.init.Xavier(magnitude=init_mg), ctx=self.ctx)
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss() if loss is None else loss
        self.optimizer = mx.gluon.Trainer(self.net.collect_params(), 
                                          optimizer, {'learning_rate': lr})
    
    def evaluate_accuracy(self, data_iterator):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            output = self.net(data)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
        return acc.get()[1]
    
    def fit(self, train_data, test_data, epochs):
        
        smoothing_constant = .01
        ctx = self.ctx
        
        for e in range(epochs):
            for i, (data, label) in enumerate(train_data):
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)
                with autograd.record(train_mode=True):
                    output = self.net(data)
                    loss = self.loss(output, label)
                loss.backward()
                self.optimizer.step(data.shape[0])

                ##########################
                #  Keep a moving average of the losses
                ##########################
                curr_loss = nd.mean(loss).asscalar()
                moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                               else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

            test_accuracy = self.evaluate_accuracy(test_data)
            train_accuracy = self.evaluate_accuracy(train_data)
            print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))       

batch_size = 64
num_inputs = 784
num_outputs = 10

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

train_data.__dict__

num_fc = 512
num_classes = 10 #num_outputs
convs = [(20,5,2), (50,5,2)]

ctx = mx.gpu()
cnn = BaseCNNClassifier(ctx)
cnn.build_model(convs, num_fc, num_classes)
cnn.compile_model(optimizer='adam')
cnn.fit(train_data, test_data, epochs=10)

batch_size = 32

def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2,0,1))
    data = data.astype(np.float32)
    return data, label

train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train=False, transform=transformer),
    batch_size=batch_size, shuffle=False, last_batch='discard')

num_fc = 512
num_classes = 10 #num_outputs
convs = [(50,3,2), (50,3,2), (100,3,2), (100,3,2)]

ctx = mx.gpu()
cnn = BaseCNNClassifier(ctx)
cnn.build_model(convs, num_fc, num_classes)
cnn.compile_model(optimizer='adam')
cnn.fit(train_data, test_data, epochs=5)

