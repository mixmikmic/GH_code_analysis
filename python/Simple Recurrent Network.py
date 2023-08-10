import conx as cx

net = cx.Network("Polka in Time")
net.add(
    cx.Layer("input", 1),
    cx.Layer("context", 5),
    cx.Layer("hidden", 5, activation="sigmoid"),
    cx.Layer("output", 1, activation="sigmoid"),
)
net.connect("input", "hidden")
net.connect("context", "hidden")
net.connect("hidden", "output")
net.compile(error="mse", optimizer="sgd", lr=0.1, momentum=0.1)

net.picture([[0], [0.5 for i in range(5)]])

net.propagate_to("hidden", (0, [0.5 for i in range(5)]))

dataset = [
    [[0], [0]],
    [[0], [1]],
    [[1], [0]],
]

def propagate_sequence(context, dataset):
    for inputs,targets in dataset:
        context = net.propagate_to("hidden", [inputs, context])
    return context

propagate_sequence([0.5 for i in range(5)], dataset)

def build_batch_dataset(context, dataset):
    batch_dataset = []
    for inputs,targets in dataset:
        batch_dataset.append([[inputs,context], targets])
        context = net.propagate_to("hidden", [inputs, context])
    return batch_dataset

build_batch_dataset([0.5 for i in range(5)], dataset)

net.dataset.load(
    build_batch_dataset(context=[0.5 for i in range(5)],
                        dataset=dataset))

net.summary()

net.reset()
net.train(1)

from IPython.display import clear_output

def train(net, epochs, dataset, clear_context=True, report_rate=1, *args, **kwargs):
    context = initial_context = [0.5 for i in range(5)]
    for epoch in range(epochs):
        net.dataset.load(build_batch_dataset(context, dataset))
        total_epochs, results = net.train(1, plot=False, verbose=0, *args, **kwargs)
        if epoch % report_rate == 0:
            clear_output(wait=True)
            net.plot_results()
        if clear_context:
            context = initial_context
        else:
            context = propagate_sequence(context, dataset)
        if "accuracy" in kwargs:
            if results["acc"] >= kwargs["accuracy"]:
                break
    clear_output(wait=True)
    net.plot_results()

net.reset()

train(net, 25000, dataset, accuracy=1.0, report_rate=100)

net.dashboard()

import conx as cx

net = cx.Network("Polka SimpleRNN v1")
net.add(
    cx.Layer("Input", (None, 1,)), 
    cx.SimpleRNNLayer("SimpleRNNLayer", 5, return_sequences=True),
    cx.Layer("Output", 1, activation="sigmoid", time_distributed=True),
)

net.connect()

net.compile(error="mse", optimizer="sgd", lr=0.1, momentum=0.1)

net.propagate([[0]])

net.summary()

net.dataset.load([
    [[[0], [0], [1], [0], [0], [1]],  # inputs
     [[0], [1], [0], [0], [1], [0]]], # targets
])

net.dashboard()

net.reset()

net.train(1000, accuracy=1.0, report_rate=100)

net.propagate([[0], [0], [1], [0], [0], [1]])

net = cx.Network("Polka SimpleRNN v2")
net.add(
    cx.Layer("Input", (3, 1), batch_shape=(1,3,1)), 
    cx.SimpleRNNLayer("SimpleRNNLayer", 5, stateful=True),
    cx.Layer("Output", 1, activation="sigmoid"),
)

net.connect()

net.compile(error="mse", optimizer="sgd", lr=0.1, momentum=0.1)

net.propagate([[0], [0], [1]])

net.summary()

net.dataset.load([
    [[[0], [0], [1]], [0]], 
    [[[0], [1], [0]], [1]], 
    [[[1], [0], [0]], [0]], 
])

net.dashboard()

net.reset()

net.train(3000, batch_size=1, shuffle=False, report_rate=100, accuracy=1.0)

net.propagate([[0], [0], [1]])

net.propagate([[0], [1], [0]])

net.propagate([[1], [0], [0]])

net = cx.Network("Polka SimpleRNN v3")
net.add(
    cx.Layer("Input", (None, 1)), 
    cx.SimpleRNNLayer("SimpleRNNLayer", 5),
    cx.Layer("Output", 1, activation="sigmoid"),
)

net.connect()

net.compile(error="mse", optimizer="sgd", lr=0.1, momentum=0.1)

net.propagate([[0]])

net.summary()

net.dataset.load([
    [[[0], [0], [1]], [0]],
    [[[0], [1], [0]], [1]],
    [[[1], [0], [0]], [0]],
    [[[0], [0], [1]], [0]],
    [[[0], [1], [0]], [1]],
    [[[1], [0], [0]], [0]],
])

net.dashboard()

net.train(3000, report_rate=100, accuracy=1.0)

net.propagate([[0], [1], [0]])

net.propagate([[0]])

net.propagate([[0], [1]])

net.propagate([[0], [1], [0]])



