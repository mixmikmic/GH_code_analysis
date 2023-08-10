import conx as cx

net = cx.Network("XOR Network", 2, 3, 1, activation="sigmoid")

net.compile(loss='mean_squared_error', optimizer='sgd', lr=0.3, momentum=0.9)

XOR = [
    ([0, 0], [0], "0"),
    ([0, 1], [1], "1"),
    ([1, 0], [1], "1"),
    ([1, 1], [0], "0")
]

net.dataset.load(XOR)
net.dataset.summary()

dash = net.dashboard()
dash

net.propagate([0, 1])

net.train(epochs=1000, accuracy=1, report_rate=25, record=True)

zeros = net.dataset.inputs.select(lambda i,ds: ds.labels[i] == "0")
ones = net.dataset.inputs.select(lambda i,ds: ds.labels[i] == "1")

net.playback(lambda net,epoch: (
    net.plot_activation_map(scatter=[["0", zeros], ["1", ones]],
                            symbols={"0": "ko", "1": "k+"},
                            title="Epoch %s" % epoch,
                            format="image"),
    net.plot('all', end=epoch+1, format="image")))

net.set_weights_from_history(-1)

states = [net.propagate_to("hidden", pattern) for pattern in net.dataset.inputs]
pca = cx.PCA(states)

symbols = {
    "0 (correct)": "bo",
    "0 (wrong)": "bx",
    "1 (correct)": "ro",
    "1 (wrong)": "rx",
}
net.playback(lambda net,epoch: cx.scatter(**pca.transform_network_bank(net, "hidden"),
                                          symbols=symbols,
                                          format='svg'))

net.set_weights_from_history(-1)

net.propagate_to("input", [0, 1])

net.propagate([0.5, 0.5])

net.propagate_to("hidden", [1, 0])

net.propagate_to("output", [1, 1])

net.propagate_to("input", [0.25, 0.25])

net.propagate_from("input", [1.0, 1.0])

net.propagate_from("hidden", [1.0, 0.0, -1.0])

net2 = cx.Network("XOR2 Network")

net2.add(cx.Layer("input1", 1),
         cx.Layer("input2", 1),
         cx.Layer("hidden1", 10, activation="sigmoid"),
         cx.Layer("hidden2", 10, activation="sigmoid"),
         cx.Layer("shared-hidden", 5, activation="sigmoid"),
         cx.Layer("output1", 1, activation="sigmoid"),
         cx.Layer("output2", 1, activation="sigmoid"))

net2.connect("input1", "hidden1")
net2.connect("input2", "hidden2")
net2.connect("hidden1", "shared-hidden")
net2.connect("hidden2", "shared-hidden")
net2.connect("shared-hidden", "output1")
net2.connect("shared-hidden", "output2")

net2.picture()

net2.layers[2].incoming_connections

net2.compile(loss='mean_squared_error', optimizer='SGD', lr=0.3, momentum=0.9)

net2.config["hspace"] = 200
dash = net2.dashboard()
dash

net2.propagate_to("hidden1", [[1], [1]])

net2.propagate([[1], [1]])

XOR2 = [
    ([[0],[0]], [[0],[0]]),
    ([[0],[1]], [[1],[1]]),
    ([[1],[0]], [[1],[1]]),
    ([[1],[1]], [[0],[0]])
]

net2.dataset.load(XOR2)

net2.get_weights("hidden2")

net2.propagate([[1], [1]])

import time
net2.reset()
for i in range(20):
    (epoch_count, results) = net2.train(epochs=100, verbose=0, report_rate=25)
    for index in range(4):
        net2.propagate(XOR2[index][0])
        time.sleep(0.1)

net2.reset()
net2.train(epochs=2000, accuracy=1.0, report_rate=25)

net2.propagate_from("shared-hidden", [0.0] * 5)

net2.propagate_to("hidden1", [[1], [1]])

net2.dataset.slice(2)

net2.train(epochs=2000, accuracy=1.0, report_rate=25)

net2.plot('all')



