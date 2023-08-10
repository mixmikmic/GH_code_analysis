import conx as cx

net = cx.Network("XOR", 2, 2, 1, activation="sigmoid")
net.compile(error="mse", optimizer="sgd", lr=0.1, momentum=0.1)

net.summary()

def make_dataset(unbalanced=False):
    dataset = [
        [[0, 0], [0], ["0,0"]],
        [[1, 0], [1], ["1,0"]],
        [[1, 1], [0], ["1,1"]],
        [[0, 1], [1], ["0,1"]],
    ]
    if unbalanced:
        for i in range(99):
            dataset.append([[0, 1], [1], ["0,1"]])
    return dataset

net.dataset.load(make_dataset())
net.reset()
net.train(15000, accuracy=1.0, report_rate=100)

net.dataset.load(make_dataset(unbalanced=True))
net.reset()
net.train(15000, accuracy=1.0, report_rate=100)

net.dataset.load(make_dataset(unbalanced=True))
net.reset()
net.train(15000, accuracy=1.0, report_rate=100, sample_weight=cx.np.array([100, 100, 100] + ([1] * 100)))

net.plot_activation_map()

net = cx.Network("XOR", 2, 2, 1, activation="sigmoid")
net.compile(error="mse", optimizer="sgd", lr=10.0, momentum=0.1)

net.dataset.load(make_dataset(unbalanced=False))
net.reset()
net.train(15000, accuracy=1.0, report_rate=100)

from sklearn.utils.extmath import softmax

class PatternWeights():
    def __init__(self, network, recompute):
        self.count = 0
        self.network = network
        self.recompute = recompute
        self.length = len(self.network.dataset.inputs)
    def __len__(self):
        return self.length
    def __getitem__(self, indices):
        if self.count % self.recompute == 0:
            diffs = cx.np.abs(self.network.model.predict(net.dataset._inputs) - self.network.dataset._targets)
            self.diffs = [1] * self.length
            self.diffs[cx.argmax([x[0] for x in softmax(diffs)[0].tolist()])] = self.length
        self.count += 1
        return [self.diffs[i] for i in indices]
    @property
    def shape(self):
        return (self.length,)
    @property
    def ndim(self):
        return 1

mylist = PatternWeights(net, 100)

net.dataset.load(make_dataset(unbalanced=True))
net.reset()
net.train(15000, accuracy=1.0, report_rate=100, sample_weight=mylist)

net.plot_activation_map()



