import hdfs3

hdfs = hdfs3.HDFileSystem(host='54.86.159.15')

from distributed import Executor, progress, wait

from distributed.hdfs import read_bytes, write_bytes

e = Executor('54.86.159.15:8786')

e.restart()

from sklearn.grid_search import ParameterGrid

import numpy as np

param_grid = {'learning_rate': np.linspace(0.1, 0.8, num=8), 'steps': [1000]}
params_ = list(ParameterGrid(param_grid))

params = e.scatter(params_)

def load_train():
    import numpy as np
    from io import StringIO
    
    with hdfs.open('/tmp/mnist_train.csv') as f:
        train = f.read()

    train = StringIO(train.decode('utf-8'))
    mnist_train = np.loadtxt(train, delimiter=',')
    return mnist_train

def load_test():
    import numpy as np
    from io import StringIO
        
    with hdfs.open('/tmp/mnist_test.csv') as f:
        test = f.read()
    
    test = StringIO(test.decode('utf-8'))
    mnist_test = np.loadtxt(test, delimiter=',')
    return mnist_test

def train(params, train_data, test_data):
    import skflow
    
    mnistX_train = train_data[:, 1:]
    mnistY_train = train_data[:, 0]
    mnistX_test = test_data[:, 1:]
    mnistY_test = test_data[:, 0]
    
    classifier = skflow.TensorFlowLinearClassifier(
            n_classes=10, batch_size=100, **params)
    classifier.fit(mnistX_train, mnistY_train)
    
    from sklearn.metrics import accuracy_score
    score = accuracy_score(classifier.predict(mnistX_test), mnistY_test)
    return score

results = e.map(train, params, traindata=future_traindata, testdata=future_testdata)

progress(results)

results

[a for a in results]

