import conx as cx

net = cx.Network("Bad Output", 2, 3, 1, activation="sigmoid")

net.compile(error="mse", optimizer="sgd")

net.picture()

net.dataset.append([0, 1], [1])

net.dataset.append([1, 0], [-1])

net.picture()



